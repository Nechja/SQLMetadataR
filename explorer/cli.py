import argparse
import os
import sys
import traceback

from .sql_explorer import SQLExplorer

def main():
    parser = argparse.ArgumentParser(description='Extract SQLite database metadata in AI-friendly format')
    parser.add_argument('database', nargs='?', default=None, 
                        help='Path to the SQLite database file (default: ../Datasets/dvd.db)')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: database_name_metadata.json)')
    parser.add_argument('--schema', '-s', help='Schema-only JSON file path (default: database_name_schema.json)')
    parser.add_argument('--sample-rows', '-r', type=int, default=3, help='Number of sample rows to include per table')
    parser.add_argument('--max-values', '-v', type=int, default=10, help='Maximum distinct values to sample per column')
    parser.add_argument('--no-execute', action='store_true', help='Skip query execution')
    parser.add_argument('--query-results', '-q', type=int, default=5, 
                        help='Number of result rows to include for executed queries')
    
    # Modify semantic processing arguments to be enabled by default
    parser.add_argument('--no-semantic', action='store_true', 
                        help='Disable generation of semantic context JSON file')
    parser.add_argument('--semantic-output', '-sem', 
                        help='Path for semantic context JSON file (default: database_name_semantic.json)')
    parser.add_argument('--table-dict', '-td', 
                        help='Path to table dictionary JSON file with business context (default: database_name_table_dict.json)')
    parser.add_argument('--no-embedding-data', action='store_true',
                        help='Disable generation of embedding-ready data JSON file')
    parser.add_argument('--embedding-output', '-e',
                        help='Path for embedding-ready data JSON file (default: database_name_embedding_data.json)')
    parser.add_argument('--nl-descriptions', action='store_true',
                        help='Generate natural language descriptions JSON file')
    parser.add_argument('--nl-output',
                        help='Path for natural language descriptions JSON file (default: database_name_nl_descriptions.json)')
    
    args = parser.parse_args()
    
    try:
        # Set default database path if none provided
        if args.database is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            args.database = os.path.join(script_dir, "..", "Datasets", "project.db")
        
        # Set default output file paths if none provided
        base_name = os.path.splitext(os.path.basename(args.database))[0]
        metadata_output = args.output or f"{base_name}_metadata.json"
        schema_output = args.schema or f"{base_name}_schema.json"
        
        # Set default table dict path if none provided but file exists
        table_dict_path = args.table_dict
        if not table_dict_path:
            default_table_dict = f"{base_name}_table_dict.json"
            if os.path.exists(default_table_dict):
                table_dict_path = default_table_dict
        
        # Enable semantic processing by default
        semantic_output = False if args.no_semantic else (args.semantic_output or True)
        embedding_output = False if args.no_embedding_data else (args.embedding_output or True)
        nl_output = args.nl_output if args.nl_descriptions else False
        
        # Create the explorer and run workflow
        explorer = SQLExplorer(args.database)
        explorer.run_workflow(
            output=metadata_output,
            schema_output=schema_output,
            sample_rows=args.sample_rows,
            max_column_values=args.max_values,
            execute_queries=not args.no_execute,
            query_result_limit=args.query_results,
            semantic_output=semantic_output,
            table_dict_path=table_dict_path,
            embedding_output=embedding_output,
            nl_descriptions_output=nl_output
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
