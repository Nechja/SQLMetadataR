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
from typing import Optional, Dict, Any
from pathlib import Path

import anthropic
from .benchmark import Benchmarker, mock_ai_query_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def anthropic_query_generator(metadata: Dict[str, Any], natural_language: str) -> str:
    """
    Generate SQL query using Anthropic API with rate limiting and exponential backoff.
    
    Args:
        metadata: Database metadata
        natural_language: Natural language question
        
    Returns:
        Generated SQL query
    """
    # Load API key from keys.info
    api_key = load_api_key()
    if not api_key:
        raise ValueError("Anthropic API key not found. Please add ANTHROPIC_API_KEY=your_key to keys.info file.")
    
    # Create the Anthropic client with the API key
    client = anthropic.Anthropic(api_key=api_key)
    
    # Prepare the prompt with database structure and question
    prompt = f"""
    You are an expert SQL query generator. Given the database metadata and a natural language question,
    generate the most appropriate SQL query to answer the question. Only return the SQL query without any explanation or markdown formatting.
    
    DATABASE METADATA:
    ```json
    {json.dumps(metadata, indent=2)}
    ```
    
    NATURAL LANGUAGE QUESTION:
    {natural_language}
    
    SQL QUERY:
    """
    
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
def run_benchmark_with_rate_limiting(benchmarker, generator_func, batch_size=5, batch_delay=3):
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
                        actual_query = generator_func(benchmarker.db_metadata, nl_variation)
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
    benchmark_parser.add_argument("--batch-size", type=int, default=5, help="Number of queries to process in each batch")
    benchmark_parser.add_argument("--batch-delay", type=int, default=3, help="Delay in seconds between batches")
    
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
        
        # Select AI query generator
        if args.mock:
            logger.info("Using mock AI query generator")
            generator_func = mock_ai_query_generator
            
            # Run regular benchmark for mock generator
            results = benchmarker.run_benchmark(ai_generator_func=generator_func)
        else:
            logger.info("Using Anthropic API for query generation with rate limiting")
            
            # Run rate-limited benchmark
            results = run_benchmark_with_rate_limiting(
                benchmarker, 
                anthropic_query_generator,
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
        
        # Generate query with AI
        if args.mock:
            actual_query = mock_ai_query_generator(benchmarker.db_metadata, args.nl)
        else:
            actual_query = anthropic_query_generator(benchmarker.db_metadata, args.nl)
            
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