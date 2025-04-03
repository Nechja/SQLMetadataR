"""
Benchmark module for evaluating NLP-to-SQL query translation performance.

This module implements evaluation and benchmarking capabilities for the 
SQLMetadataR project, following the workflow described in what_is_benchmark.md.
"""

import json
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict

from .models import NlpSqlQuery, MetaData, NlpSqlTrainingData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QueryExecutionResult:
    """Results from executing a query against the database."""
    query: str
    execution_time_ms: float
    row_count: int
    column_names: List[str]
    result_sample: List[Dict[str, Any]]  # Sample of rows as dictionaries
    error: Optional[str] = None
    execution_plan: Optional[str] = None
    
@dataclass
class QueryComparison:
    """Comparison between expected and actual query results."""
    natural_language: str
    expected_query: str
    actual_query: str
    match_type: str  # "exact", "semantic", "result", "none"
    expected_result: Optional[QueryExecutionResult] = None
    actual_result: Optional[QueryExecutionResult] = None
    error_category: Optional[str] = None  # join, filter, aggregation, syntax, schema, business_logic
    error_details: Optional[str] = None
    complexity_tier: str = "simple"  # simple, medium, advanced
    
@dataclass
class BenchmarkResults:
    """Overall results from a benchmarking run."""
    timestamp: datetime = field(default_factory=datetime.now)
    database_name: str = ""
    total_queries: int = 0
    successful_queries: int = 0
    exact_matches: int = 0
    semantic_matches: int = 0
    result_matches: int = 0
    failed_queries: int = 0
    error_categories: Dict[str, int] = field(default_factory=dict)
    query_comparisons: List[QueryComparison] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the results to a dictionary for serialization."""
        results_dict = asdict(self)
        results_dict["timestamp"] = self.timestamp.isoformat()
        return results_dict
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the results to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, filepath: str) -> None:
        """Save the results to a JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Benchmark results saved to {filepath}")


class QueryExecutor:
    """Executes SQL queries against a SQLite database."""
    
    def __init__(self, db_path: str):
        """Initialize with path to SQLite database."""
        self.db_path = db_path
        self.conn = None
        
    def connect(self) -> None:
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def execute_query(self, query: str, limit_sample: int = 10) -> QueryExecutionResult:
        """Execute a query and return results."""
        if not self.conn:
            self.connect()
            
        cursor = self.conn.cursor()
        start_time = time.time()
        error = None
        result_sample = []
        column_names = []
        row_count = 0
        execution_plan = None
        
        try:
            # Get query execution plan
            try:
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                execution_plan = "\n".join([str(dict(row)) for row in cursor.fetchall()])
            except sqlite3.Error as e:
                execution_plan = f"Error getting execution plan: {str(e)}"
            
            # Execute the actual query
            cursor.execute(query)
            column_names = [col[0] for col in cursor.description]
            
            # Fetch sample rows
            sample_rows = cursor.fetchmany(limit_sample)
            result_sample = [dict(row) for row in sample_rows]
            
            # Count total rows without fetching all data
            # This might require executing the query again with a COUNT wrapper for complex queries
            try:
                row_count = len(cursor.fetchall()) + len(sample_rows)
            except:
                # If we can't get the exact count, estimate based on what we fetched
                row_count = len(sample_rows)
                if len(sample_rows) == limit_sample:
                    row_count = str(f"{limit_sample}+")
                    
        except sqlite3.Error as e:
            error = str(e)
            
        execution_time_ms = (time.time() - start_time) * 1000
        
        return QueryExecutionResult(
            query=query,
            execution_time_ms=execution_time_ms,
            row_count=row_count,
            column_names=column_names,
            result_sample=result_sample,
            error=error,
            execution_plan=execution_plan
        )


class Benchmarker:
    """Evaluates NLP-to-SQL system performance using a benchmark dataset."""
    
    def __init__(self, 
                db_path: str, 
                semantic_json_path: str,
                query_patterns_path: Optional[str] = None):
        """
        Initialize the benchmarker.
        
        Args:
            db_path: Path to the SQLite database
            semantic_json_path: Path to the semantic JSON file
            query_patterns_path: Optional path to query patterns JSON file
        """
        self.db_path = db_path
        self.executor = QueryExecutor(db_path)
        
        # Load database metadata
        with open(semantic_json_path, 'r') as f:
            self.db_metadata = json.load(f)
            
        # Load query patterns from different potential sources
        self.query_patterns = []
        
        # Try loading from dedicated query patterns file first if provided
        if query_patterns_path:
            try:
                with open(query_patterns_path, 'r') as f:
                    pattern_data = json.load(f)
                    if "query_examples" in pattern_data:
                        self.query_patterns = pattern_data["query_examples"]
                    elif "query_patterns" in pattern_data:
                        self.query_patterns = pattern_data["query_patterns"]
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load query patterns from {query_patterns_path}: {e}")
                
        # If no patterns loaded yet, try to get them from semantic JSON
        if not self.query_patterns and "query_examples" in self.db_metadata:
            logger.info(f"Loading query patterns from semantic JSON file")
            self.query_patterns = self.db_metadata["query_examples"]
            
        # If we have patterns, log the count
        if self.query_patterns:
            logger.info(f"Loaded {len(self.query_patterns)} query patterns")
        else:
            logger.warning("No query patterns found in any source")
                
        self.results = BenchmarkResults(
            database_name=Path(db_path).stem
        )
                
    def categorize_error(self, expected_query: str, actual_query: str, 
                        expected_result: QueryExecutionResult, 
                        actual_result: QueryExecutionResult) -> Tuple[str, str]:
        """
        Categorize the type of error between expected and actual queries.
        
        Returns:
            Tuple of (error_category, error_details)
        """
        # Check for syntax errors first
        if actual_result.error and "syntax" in actual_result.error.lower():
            return "syntax", actual_result.error
            
        # Check for schema errors (referencing non-existent tables/columns)
        if actual_result.error and "no such table" in actual_result.error.lower():
            return "schema", f"Table referenced does not exist: {actual_result.error}"
        if actual_result.error and "no such column" in actual_result.error.lower():
            return "schema", f"Column referenced does not exist: {actual_result.error}"
            
        # Compare the queries (this is simplified - a real implementation would do more sophisticated analysis)
        expected_lower = expected_query.lower()
        actual_lower = actual_query.lower()
        
        # Check for join errors
        if "join" in expected_lower and "join" not in actual_lower:
            return "join", "Missing JOIN clause"
        if "join" not in expected_lower and "join" in actual_lower:
            return "join", "Unnecessary JOIN clause"
            
        # Check for filter errors
        if "where" in expected_lower and "where" not in actual_lower:
            return "filter", "Missing WHERE clause"
        if "where" not in expected_lower and "where" in actual_lower:
            return "filter", "Unnecessary WHERE clause"
            
        # Check for aggregation errors
        agg_functions = ["count", "sum", "avg", "min", "max"]
        expected_has_agg = any(fn in expected_lower for fn in agg_functions)
        actual_has_agg = any(fn in actual_lower for fn in agg_functions)
        
        if expected_has_agg and not actual_has_agg:
            return "aggregation", "Missing aggregation function"
        if not expected_has_agg and actual_has_agg:
            return "aggregation", "Unnecessary aggregation function"
            
        if "group by" in expected_lower and "group by" not in actual_lower:
            return "aggregation", "Missing GROUP BY clause"
        if "group by" not in expected_lower and "group by" in actual_lower:
            return "aggregation", "Unnecessary GROUP BY clause"
            
        # Default to business logic error if nothing specific is found
        return "business_logic", "Query structure differs from expected"
        
    def compare_query_results(self, 
                            expected_result: QueryExecutionResult, 
                            actual_result: QueryExecutionResult) -> bool:
        """
        Compare the results of two queries to determine if they are equivalent.
        
        Returns:
            True if results match, False otherwise
        """
        # If either query has an error, results don't match
        if expected_result.error or actual_result.error:
            return False
            
        # If row counts differ significantly, results don't match
        # We use a rough heuristic here - could be more sophisticated
        if isinstance(expected_result.row_count, int) and isinstance(actual_result.row_count, int):
            if abs(expected_result.row_count - actual_result.row_count) > 3:
                return False
                
        # If column counts differ, results don't match
        if len(expected_result.column_names) != len(actual_result.column_names):
            return False
            
        # Compare sample results (very basic comparison)
        # A real implementation would do more sophisticated comparison
        if not expected_result.result_sample and not actual_result.result_sample:
            return True
        if (not expected_result.result_sample and actual_result.result_sample) or \
           (expected_result.result_sample and not actual_result.result_sample):
            return False
            
        # Compare first row if available
        if (expected_result.result_sample and actual_result.result_sample and 
            len(expected_result.result_sample) > 0 and len(actual_result.result_sample) > 0):
            # Just check if number of columns match for now
            expected_first_row = expected_result.result_sample[0]
            actual_first_row = actual_result.result_sample[0]
            if len(expected_first_row) != len(actual_first_row):
                return False
                
        return True
        
    def evaluate_query(self, 
                      natural_language: str, 
                      expected_query: str, 
                      actual_query: str,
                      complexity_tier: str = "simple") -> QueryComparison:
        """
        Evaluate a single NLP-to-SQL query translation.
        
        Args:
            natural_language: The natural language question
            expected_query: The expected SQL query
            actual_query: The AI-generated SQL query
            complexity_tier: Complexity level (simple, medium, advanced)
            
        Returns:
            QueryComparison object with evaluation results
        """
        # Execute both queries
        expected_result = self.executor.execute_query(expected_query)
        actual_result = self.executor.execute_query(actual_query)
        
        # Determine match type
        if expected_query.lower() == actual_query.lower():
            match_type = "exact"
        elif self.compare_query_results(expected_result, actual_result):
            match_type = "result"
        else:
            # Here we would check for semantic equivalence
            # This is a simplified version - real implementation would be more sophisticated
            expected_normalized = " ".join(expected_query.lower().split())
            actual_normalized = " ".join(actual_query.lower().split())
            
            # Very basic check - in reality, you'd use more sophisticated parsing
            if (expected_normalized.replace("  ", " ") == actual_normalized.replace("  ", " ")):
                match_type = "semantic"
            else:
                match_type = "none"
                
        error_category = None
        error_details = None
        
        # If no match, categorize the error
        if match_type == "none":
            error_category, error_details = self.categorize_error(
                expected_query, actual_query, expected_result, actual_result
            )
            
        return QueryComparison(
            natural_language=natural_language,
            expected_query=expected_query,
            actual_query=actual_query,
            match_type=match_type,
            expected_result=expected_result,
            actual_result=actual_result,
            error_category=error_category,
            error_details=error_details,
            complexity_tier=complexity_tier
        )
        
    def run_benchmark(self, ai_generator_func=None) -> BenchmarkResults:
        """
        Run the benchmark using the query patterns.
        
        Args:
            ai_generator_func: Function that takes (metadata, natural_language) and returns SQL
                              If None, this function must be implemented by the caller
                              
        Returns:
            BenchmarkResults object with evaluation results
        """
        start_time = time.time()
        
        # Initialize results
        self.results = BenchmarkResults(
            database_name=Path(self.db_path).stem,
            timestamp=datetime.now()
        )
        
        if not self.query_patterns:
            logger.warning("No query patterns available for benchmarking")
            return self.results
            
        # Group queries by complexity
        queries_by_complexity = {
            "simple": [],
            "medium": [],
            "advanced": []
        }
        
        for pattern in self.query_patterns:
            # Skip patterns without natural language variations
            if "natural_language_variations" not in pattern or not pattern["natural_language_variations"]:
                if "natural_language" in pattern:
                    # If it has natural_language list but no variations, use that
                    pattern["natural_language_variations"] = pattern["natural_language"]
                else:
                    # Skip pattern with no NL examples
                    continue
            
            # Get complexity (default to simple if not specified)
            complexity = pattern.get("complexity", "simple")
            if complexity not in queries_by_complexity:
                complexity = "simple"
                
            queries_by_complexity[complexity].append(pattern)
            
        # Log number of queries in each complexity tier
        for tier in queries_by_complexity:
            logger.info(f"Found {len(queries_by_complexity[tier])} queries in complexity tier: {tier}")
            
        # Process queries in order of increasing complexity
        all_tiers = ["simple", "medium", "advanced"]
        
        for tier in all_tiers:
            if not queries_by_complexity[tier]:
                continue
                
            logger.info(f"Processing {len(queries_by_complexity[tier])} queries in tier: {tier}")
            
            for pattern in queries_by_complexity[tier]:
                # Get the expected query (use 'query' field name if available, otherwise look for 'sql')
                expected_query = pattern.get("query", pattern.get("sql", ""))
                if not expected_query:
                    logger.warning(f"Skipping pattern with no query/sql: {pattern.get('description', 'Unknown')}")
                    continue
                
                # Get natural language variations
                nl_variations = pattern.get("natural_language_variations", [])
                if not nl_variations and "natural_language" in pattern:
                    nl_variations = pattern["natural_language"]
                    
                if not nl_variations:
                    logger.warning(f"Skipping pattern with no natural language variations: {pattern.get('description', 'Unknown')}")
                    continue
                
                # Process each natural language variation
                for nl_variation in nl_variations:
                    self.results.total_queries += 1
                    
                    # Get actual query from AI (or use expected if no AI function)
                    if ai_generator_func:
                        try:
                            actual_query = ai_generator_func(self.db_metadata, nl_variation)
                        except Exception as e:
                            logger.error(f"Error generating query for '{nl_variation}': {e}")
                            actual_query = "ERROR: " + str(e)
                    else:
                        # For testing without an AI generator
                        actual_query = expected_query
                        
                    # Evaluate the query
                    comparison = self.evaluate_query(
                        natural_language=nl_variation,
                        expected_query=expected_query,
                        actual_query=actual_query,
                        complexity_tier=tier
                    )
                    
                    # Update result counters
                    if comparison.match_type in ["exact", "semantic", "result"]:
                        self.results.successful_queries += 1
                        
                        if comparison.match_type == "exact":
                            self.results.exact_matches += 1
                        elif comparison.match_type == "semantic":
                            self.results.semantic_matches += 1
                        elif comparison.match_type == "result":
                            self.results.result_matches += 1
                    else:
                        self.results.failed_queries += 1
                        
                        # Track error categories
                        if comparison.error_category:
                            self.results.error_categories[comparison.error_category] = \
                                self.results.error_categories.get(comparison.error_category, 0) + 1
                                
                    # Add the comparison to results
                    self.results.query_comparisons.append(comparison)
                    
        # Record total execution time
        self.results.execution_time_ms = (time.time() - start_time) * 1000
        
        return self.results
        
    def generate_benchmark_report(self, output_file: str) -> None:
        """
        Generate a detailed report from benchmark results.
        
        Args:
            output_file: Path to save the benchmark results JSON
        """
        if self.results.total_queries == 0:
            logger.warning("No queries were evaluated. Run benchmark first.")
            return
            
        self.results.save_to_file(output_file)
        
        # Log a summary of the results
        logger.info(f"Benchmark complete: {self.results.successful_queries}/{self.results.total_queries} queries successful ({self.results.success_rate:.2f}%)")
        logger.info(f"Exact matches: {self.results.exact_matches}")
        logger.info(f"Semantic matches: {self.results.semantic_matches}")
        logger.info(f"Result matches: {self.results.result_matches}")
        
        if self.results.error_categories:
            logger.info("Error categories:")
            for category, count in self.results.error_categories.items():
                logger.info(f"  {category}: {count}")
    
    def visualize_results(self):
        """Generate visualizations of benchmark results."""
        # This would be implemented with matplotlib or another visualization library
        # For now, we'll just log that this would create visualizations
        logger.info("Visualization of results is not yet implemented")


# Mock AI function for testing - would be replaced with actual AI implementation
def mock_ai_query_generator(metadata: Dict, natural_language: str) -> str:
    """Mock function to simulate AI-generated SQL queries."""
    # In a real implementation, this would call your NLP-to-SQL model
    return f"SELECT * FROM table WHERE description = '{natural_language}' LIMIT 10;"
