import argparse
import os
import sys
import traceback
from datetime import datetime

from .sql_explorer import SQLExplorer

def main():
    parser = argparse.ArgumentParser(description='Extract SQLite database metadata in AI-friendly format')
    parser.add_argument('database', nargs='?', default=None, 
                        help='Path to the SQLite database file (default: ../Datasets/dvd.db)')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: database_name_metadata.json)')
    parser.add_argument('--sample-rows', '-r', type=int, default=3, help='Number of sample rows to include per table')
    parser.add_argument('--max-values', '-v', type=int, default=10, help='Maximum distinct values to sample per column')
    parser.add_argument('--no-execute', action='store_true', help='Skip query execution')
    parser.add_argument('--query-results', '-q', type=int, default=5, 
                        help='Number of result rows to include for executed queries')
    
    args = parser.parse_args()
    
    try:
        # Set default database path if none provided
        if args.database is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.database = os.path.join(script_dir, "..", "Datasets", "dvd.db")
        
        # Set default output file path if none provided
        if args.output is None:
            base_name = os.path.splitext(os.path.basename(args.database))[0]
            args.output = f"{base_name}_metadata.json"
        
        # Create the explorer and extract metadata
        explorer = SQLExplorer(args.database)
        db = explorer.extract_metadata(
            sample_rows=args.sample_rows,
            max_column_values=args.max_values,
            execute_queries=not args.no_execute,
            query_result_limit=args.query_results
        )
        
        # Save to file
        db.save_to_file(args.output)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
