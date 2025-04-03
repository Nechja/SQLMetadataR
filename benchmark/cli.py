"""
Command-line interface for the SQLMetadataR benchmarking module.

This module provides a CLI for running benchmarks, evaluating AI-generated
SQL queries, and generating benchmark reports.
"""

import argparse
import json
import os
import sys
import logging
import time
import random
import re
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path

import anthropic
from .benchmark import Benchmarker, mock_ai_query_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Maximum token limit for Anthropic API (conservative estimate)
MAX_TOKENS = 180000  # Setting lower than actual limit to leave room for the response

def load_api_key(keys_file: str = "keys.info") -> Optional[str]:
    """Load API key from keys.info file."""
    if not os.path.exists(keys_file):
        logger.warning(f"API keys file '{keys_file}' not found.")
        return None
        
    try:
        with open(keys_file, 'r') as f:
            for line in f:
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    return line.strip().split("=")[1].strip()
    except Exception as e:
        logger.error(f"Error reading API key: {e}")
        
    return None

def load_additional_context(base_name: str) -> Dict[str, Any]:
    """
    Load additional context files based on the base name of the database.
    
    Args:
        base_name: Base name of the database (e.g., 'dvd' for 'dvd.db')
        
    Returns:
        Dictionary containing additional context data
    """
    context = {}
    
    # List of potential context files
    context_files = [
        f"{base_name}_metadata.json",
        f"{base_name}_schema.json",
        f"{base_name}_table_dict.json",
        f"{base_name}_embedding_data.json"
    ]
    
    for file_path in context_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Use filename without extension as the key
                    key = os.path.splitext(os.path.basename(file_path))[0]
                    context[key] = data
                    logger.info(f"Loaded additional context from {file_path}")
            except Exception as e:
                logger.warning(f"Could not load context from {file_path}: {e}")
    
    return context

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a string.
    This is a simple approximation - tokens are roughly 4 characters on average.
    
    Args:
        text: The text string to estimate token count for
        
    Returns:
        Estimated token count
    """
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4 + 1

def truncate_context_to_fit_tokens(metadata: Dict[str, Any], additional_context: Dict[str, Any], natural_language: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Truncate context data to fit within token limits.
    
    Args:
        metadata: Database metadata
        additional_context: Additional context data
        natural_language: The natural language query
        
    Returns:
        Tuple of (truncated metadata, truncated additional context)
    """
    # Start with essential content
    base_prompt = f"""
    You are an expert SQL query generator. Given the database metadata and a natural language question,
    generate the most appropriate SQL query to answer the question.
    
    NATURAL LANGUAGE QUESTION:
    {natural_language}
    
    SQL QUERY:
    """
    
    token_budget = MAX_TOKENS - estimate_token_count(base_prompt) - 2000  # Reserve 2000 tokens for overhead and response
    
    # Create deep copies to avoid modifying the originals
    truncated_metadata = metadata.copy()
    truncated_additional_context = {}
    if additional_context:
        truncated_additional_context = {k: v.copy() if isinstance(v, dict) else v for k, v in additional_context.items()}
    
    # Start with most important context - table info from metadata
    metadata_str = json.dumps(truncated_metadata, indent=2)
    metadata_tokens = estimate_token_count(metadata_str)
    
    # If metadata alone is too large, we need to truncate it
    if metadata_tokens > token_budget * 0.7:  # Allocate 70% of budget to metadata
        logger.warning(f"Metadata is too large ({metadata_tokens} tokens). Truncating...")
        
        # Keep database info and table structure, but limit details
        if "tables" in truncated_metadata:
            for table_name, table_data in list(truncated_metadata["tables"].items()):
                # Remove less important sections
                for field in ["technical_details", "business_rules", "metrics", "domain_terms"]:
                    if field in table_data:
                        del table_data[field]
        
        # Reduce column information to essential fields
        if "columns" in truncated_metadata:
            columns_to_keep = {}
            for col_key, col_data in truncated_metadata["columns"].items():
                if "." in col_key:  # Table.column format
                    table_name, col_name = col_key.split(".")
                    # Keep only primary keys and a few important columns
                    if col_data.get("is_primary_key", False):
                        columns_to_keep[col_key] = col_data
                    elif any(important in col_name for important in ["id", "name", "key", "date"]):
                        columns_to_keep[col_key] = col_data
            truncated_metadata["columns"] = columns_to_keep
            
        # Limit number of relationships
        if "relationships" in truncated_metadata and len(truncated_metadata["relationships"]) > 20:
            truncated_metadata["relationships"] = truncated_metadata["relationships"][:20]
            
        metadata_str = json.dumps(truncated_metadata, indent=2)
        metadata_tokens = estimate_token_count(metadata_str)
    
    remaining_budget = token_budget - metadata_tokens
    
    # Check if we have any budget left for additional context
    if remaining_budget <= 0:
        logger.warning("No token budget left for additional context")
        return truncated_metadata, {}
    
    # Prioritize context by importance
    priority_order = ["dvd_table_dict", "dvd_schema", "dvd_embedding_data"]
    
    for context_key in priority_order:
        if context_key in truncated_additional_context:
            context_str = json.dumps(truncated_additional_context[context_key], indent=2)
            context_tokens = estimate_token_count(context_str)
            
            if context_tokens <= remaining_budget:
                remaining_budget -= context_tokens
            else:
                # Need to truncate or remove this context
                if context_key == "dvd_table_dict":
                    # Table dict is high priority - try to keep core info
                    table_dict = truncated_additional_context[context_key]
                    
                    # Keep database_info and a few important tables
                    important_tables = ["film", "customer", "rental", "payment"]
                    truncated_tables = {k: v for k, v in table_dict.items() 
                                      if k == "database_info" or k in important_tables}
                    
                    # For each table, keep only essential columns
                    for table_name, table_data in truncated_tables.items():
                        if isinstance(table_data, dict) and "columns" in table_data:
                            # Keep only a subset of columns
                            col_subset = {}
                            for col_name, col_desc in list(table_data["columns"].items())[:5]:
                                col_subset[col_name] = col_desc
                            table_data["columns"] = col_subset
                            
                            # Remove less critical info
                            for field in ["business_rules", "metrics", "domain_terms"]:
                                if field in table_data:
                                    del table_data[field]
                    
                    truncated_additional_context[context_key] = truncated_tables
                    
                    # Recalculate tokens
                    context_str = json.dumps(truncated_additional_context[context_key], indent=2)
                    context_tokens = estimate_token_count(context_str)
                    
                    if context_tokens <= remaining_budget:
                        remaining_budget -= context_tokens
                    else:
                        # Still too large, remove entirely
                        del truncated_additional_context[context_key]
                else:
                    # Lower priority, just remove
                    del truncated_additional_context[context_key]
    
    logger.info(f"Adjusted context to fit within token limit. Estimated tokens: {MAX_TOKENS - remaining_budget}")
    return truncated_metadata, truncated_additional_context

def anthropic_query_generator(metadata: Dict[str, Any], natural_language: str, additional_context: Dict[str, Any] = None) -> str:
    """
    Generate SQL query using Anthropic API with rate limiting and exponential backoff.
    
    Args:
        metadata: Database metadata
        natural_language: Natural language question
        additional_context: Additional context data
        
    Returns:
        Generated SQL query
    """
    # Load API key from keys.info
    api_key = load_api_key()
    if not api_key:
        raise ValueError("Anthropic API key not found. Please add ANTHROPIC_API_KEY=your_key to keys.info file.")
    
    # Create the Anthropic client with the API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Truncate context if needed to stay within token limits
    truncated_metadata, truncated_additional_context = truncate_context_to_fit_tokens(
        metadata, additional_context or {}, natural_language)
    
    # Prepare the prompt with database structure and question
    prompt = f"""
    You are an expert SQL query generator. Given the database metadata and a natural language question,
    generate the most appropriate SQL query to answer the question. Only return the SQL query without any explanation or markdown formatting.
    
    DATABASE METADATA:
    ```json
    {json.dumps(truncated_metadata, indent=2)}
    ```
    """
    
    # Add additional context if available
    if truncated_additional_context:
        if "dvd_table_dict" in truncated_additional_context:
            prompt += f"""
    TABLE DICTIONARY WITH BUSINESS CONTEXT:
    ```json
    {json.dumps(truncated_additional_context["dvd_table_dict"], indent=2)}
    ```
    """
        
        if "dvd_schema" in truncated_additional_context:
            prompt += f"""
    DATABASE SCHEMA:
    ```json
    {json.dumps(truncated_additional_context["dvd_schema"], indent=2)}
    ```
    """
            
        if "dvd_embedding_data" in truncated_additional_context:
            prompt += f"""
    EMBEDDING DATA:
    ```json
    {json.dumps(truncated_additional_context["dvd_embedding_data"], indent=2)}
    ```
    """
    
    prompt += f"""
    NATURAL LANGUAGE QUESTION:
    {natural_language}
    
    SQL QUERY:
    """
    
    # Log estimated token count
    estimated_tokens = estimate_token_count(prompt)
    logger.info(f"Estimated token count for prompt: {estimated_tokens}")
    
    # Exponential backoff settings
    max_retries = 5
    base_delay = 15  # Starting delay in seconds
    max_delay = 90  # Maximum delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Adjust parameters based on the model used
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1000,
                temperature=0.1,
                system="You are an expert SQL query generator that converts natural language questions to SQL queries. Output ONLY the SQL query without any explanation, comments, or markdown.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            sql_query = response.content[0].text.strip()
            
            # Clean up the response in case it came with markdown formatting
            if sql_query.startswith("```sql"):
                sql_query = sql_query.replace("```sql", "", 1)
                if sql_query.endswith("```"):
                    sql_query = sql_query[:-3]
            elif sql_query.startswith("```"):
                sql_query = sql_query.replace("```", "", 1)
                if sql_query.endswith("```"):
                    sql_query = sql_query[:-3]
                    
            return sql_query.strip()
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for token limit errors
            if "prompt is too long" in error_msg or "invalid_request_error" in error_msg:
                if "prompt is too long" in error_msg:
                    # Extract the actual token count from the error message if available
                    token_match = re.search(r'(\d+) tokens > (\d+) maximum', error_msg)
                    if token_match:
                        actual_tokens = int(token_match.group(1))
                        max_allowed = int(token_match.group(2))
                        logger.warning(f"Prompt too long: {actual_tokens} tokens (max allowed: {max_allowed})")
                    
                # Reduce context size more aggressively for next attempt
                global MAX_TOKENS
                MAX_TOKENS = int(MAX_TOKENS * 0.7)  # Reduce by 30%
                logger.warning(f"Reducing token budget to {MAX_TOKENS} for retry")
                
                if attempt < max_retries - 1:
                    # Try again with reduced context
                    truncated_metadata, truncated_additional_context = truncate_context_to_fit_tokens(
                        metadata, additional_context or {}, natural_language)
                    
                    # Rebuild prompt with reduced context
                    prompt = f"""
                    You are an expert SQL query generator. Given the database metadata and a natural language question,
                    generate the most appropriate SQL query to answer the question. Only return the SQL query without any explanation or markdown formatting.
                    
                    DATABASE METADATA:
                    ```json
                    {json.dumps(truncated_metadata, indent=2)}
                    ```
                    
                    NATURAL LANGUAGE QUESTION:
                    {natural_language}
                    
                    SQL QUERY:
                    """
                    
                    # No delay needed for token errors, just retry immediately with smaller context
                    logger.info(f"Retrying with reduced context (Attempt {attempt+1}/{max_retries})")
                    continue
            
            # Check if it's a rate limit error
            if "rate_limit_error" in error_msg or "Error code: 429" in error_msg:
                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    # Calculate exponential backoff with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 0.1 * delay)  # 10% jitter
                    wait_time = delay + jitter
                    
                    logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            
            # For other errors or if we've exhausted retries
            logger.error(f"Error calling Anthropic API: {e}")
            raise

# Modified benchmark function to include rate limiting between batches
def run_benchmark_with_rate_limiting(benchmarker, generator_func, additional_context=None, batch_size=3, batch_delay=5):
    """Run benchmark with rate limiting between batches of queries."""
    query_patterns = benchmarker.query_patterns
    if not query_patterns:
        logger.warning("No query patterns available for benchmarking")
        return benchmarker.results
    
    # Group by complexity
    queries_by_complexity = {
        "simple": [],
        "medium": [],
        "advanced": []
    }
    
    for pattern in query_patterns:
        if "natural_language_variations" not in pattern or not pattern["natural_language_variations"]:
            if "natural_language" in pattern:
                pattern["natural_language_variations"] = pattern["natural_language"]
            else:
                continue
        
        complexity = pattern.get("complexity", "simple")
        if complexity not in queries_by_complexity:
            complexity = "simple"
            
        queries_by_complexity[complexity].append(pattern)
    
    # Log counts
    for tier in queries_by_complexity:
        logger.info(f"Found {len(queries_by_complexity[tier])} queries in tier: {tier}")
    
    # Process in order of complexity
    all_tiers = ["simple", "medium", "advanced"]
    results = []
    
    for tier in all_tiers:
        if not queries_by_complexity[tier]:
            continue
            
        logger.info(f"Processing {len(queries_by_complexity[tier])} queries in tier: {tier}")
        
        # Process patterns in batches
        patterns = queries_by_complexity[tier]
        for i in range(0, len(patterns), batch_size):
            batch = patterns[i:i+batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1} of {(len(patterns) + batch_size - 1) // batch_size} in tier {tier}")
            
            for pattern in batch:
                # Process pattern with benchmarker
                expected_query = pattern.get("query", pattern.get("sql", ""))
                if not expected_query:
                    continue
                
                nl_variations = pattern.get("natural_language_variations", [])
                if not nl_variations and "natural_language" in pattern:
                    nl_variations = pattern["natural_language"]
                    
                if not nl_variations:
                    continue
                
                for nl_variation in nl_variations:
                    benchmarker.results.total_queries += 1
                    
                    try:
                        actual_query = generator_func(benchmarker.db_metadata, nl_variation, additional_context)
                    except Exception as e:
                        logger.error(f"Error generating query for '{nl_variation}': {e}")
                        actual_query = f"ERROR: {e}"
                        
                    comparison = benchmarker.evaluate_query(
                        natural_language=nl_variation,
                        expected_query=expected_query,
                        actual_query=actual_query,
                        complexity_tier=tier
                    )
                    
                    # Update result counters
                    if comparison.match_type in ["exact", "semantic", "result"]:
                        benchmarker.results.successful_queries += 1
                        
                        if comparison.match_type == "exact":
                            benchmarker.results.exact_matches += 1
                        elif comparison.match_type == "semantic":
                            benchmarker.results.semantic_matches += 1
                        elif comparison.match_type == "result":
                            benchmarker.results.result_matches += 1
                    else:
                        benchmarker.results.failed_queries += 1
                        
                        if comparison.error_category:
                            benchmarker.results.error_categories[comparison.error_category] = \
                                benchmarker.results.error_categories.get(comparison.error_category, 0) + 1
                                
                    benchmarker.results.query_comparisons.append(comparison)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(patterns):
                logger.info(f"Waiting {batch_delay} seconds before processing next batch...")
                time.sleep(batch_delay)
    
    return benchmarker.results

# Rest of the code remains the same
def run_cli():
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="SQLMetadataR Benchmark Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark tests")
    benchmark_parser.add_argument("--db", required=True, help="Path to SQLite database")
    benchmark_parser.add_argument("--semantic", required=True, help="Path to semantic JSON file")
    benchmark_parser.add_argument("--patterns", help="Path to query patterns JSON file")
    benchmark_parser.add_argument("--output", default="benchmark_results.json", help="Output file for benchmark results")
    benchmark_parser.add_argument("--mock", action="store_true", help="Use mock AI instead of actual API")
    benchmark_parser.add_argument("--batch-size", type=int, default=3, help="Number of queries to process in each batch")
    benchmark_parser.add_argument("--batch-delay", type=int, default=5, help="Delay in seconds between batches")
    
    # Evaluate single query command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a single query")
    evaluate_parser.add_argument("--db", required=True, help="Path to SQLite database")
    evaluate_parser.add_argument("--semantic", required=True, help="Path to semantic JSON file")
    evaluate_parser.add_argument("--nl", required=True, help="Natural language query to evaluate")
    evaluate_parser.add_argument("--expected", required=True, help="Expected SQL query")
    evaluate_parser.add_argument("--mock", action="store_true", help="Use mock AI instead of actual API")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Handle benchmark command
    if args.command == "benchmark":
        logger.info(f"Running benchmark tests on {args.db}")
        
        # Initialize benchmarker
        benchmarker = Benchmarker(
            db_path=args.db,
            semantic_json_path=args.semantic,
            query_patterns_path=args.patterns
        )
        
        # Get the base name of the database to load additional context
        db_base_name = Path(args.db).stem
        additional_context = load_additional_context(db_base_name)
        
        # Select AI query generator
        if args.mock:
            logger.info("Using mock AI query generator")
            generator_func = mock_ai_query_generator
            
            # Run regular benchmark for mock generator
            results = benchmarker.run_benchmark(ai_generator_func=lambda metadata, nl: generator_func(metadata, nl))
        else:
            logger.info("Using Anthropic API for query generation with rate limiting")
            
            # Run rate-limited benchmark
            results = run_benchmark_with_rate_limiting(
                benchmarker, 
                anthropic_query_generator,
                additional_context=additional_context,
                batch_size=args.batch_size,
                batch_delay=args.batch_delay
            )
            
        # Generate report
        benchmarker.generate_benchmark_report(args.output)
        logger.info(f"Benchmark complete. Results saved to {args.output}")
        
    # Handle evaluate command
    elif args.command == "evaluate":
        logger.info(f"Evaluating query: {args.nl}")
        
        # Initialize benchmarker
        benchmarker = Benchmarker(
            db_path=args.db,
            semantic_json_path=args.semantic
        )
        
        # Get the base name of the database to load additional context
        db_base_name = Path(args.db).stem
        additional_context = load_additional_context(db_base_name)
        
        # Generate query with AI
        if args.mock:
            actual_query = mock_ai_query_generator(benchmarker.db_metadata, args.nl)
        else:
            actual_query = anthropic_query_generator(benchmarker.db_metadata, args.nl, additional_context)
            
        # Evaluate the query
        comparison = benchmarker.evaluate_query(
            natural_language=args.nl,
            expected_query=args.expected,
            actual_query=actual_query
        )
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"Natural Language: {comparison.natural_language}")
        print(f"\nExpected Query: {comparison.expected_query}")
        print(f"\nGenerated Query: {comparison.actual_query}")
        print(f"\nMatch Type: {comparison.match_type}")
        
        if comparison.match_type == "none":
            print(f"Error Category: {comparison.error_category}")
            print(f"Error Details: {comparison.error_details}")
            
        # Print query execution results
        if comparison.expected_result and not comparison.expected_result.error:
            print(f"\nExpected Result: {len(comparison.expected_result.result_sample)} rows returned")
            
        if comparison.actual_result and not comparison.actual_result.error:
            print(f"Actual Result: {len(comparison.actual_result.result_sample)} rows returned")
        elif comparison.actual_result and comparison.actual_result.error:
            print(f"Actual Result Error: {comparison.actual_result.error}")

if __name__ == "__main__":
    run_cli()