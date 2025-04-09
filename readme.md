[![Upload Python Package](https://github.com/Nechja/SQLMetadataR/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Nechja/SQLMetadataR/actions/workflows/python-publish.yml)
# SQLMetadataR

SQLMetadataR is a tool for extracting and analyzing metadata from SQLite databases. It provides a comprehensive view of your database’s structure including tables, columns, indexes, foreign keys, and more. It also generates example SQL queries and analyzes relationships between tables.

## Features

- **Metadata Extraction**: Retrieve table structures, row counts, sample data, and column statistics.
- **Relationship Analysis**: Identify and analyze relationships based on foreign keys.
- **Query Generation and Execution**: Generate and run example SQL queries to validate extracted metadata.
- **Semantic Processing**: Augment metadata with semantic information.
- **Command-Line Interface**: Use the provided CLI to run the complete workflow from the terminal.

## Installation

You can install SQLMetadataR directly from PyPI:

```sh
pip install SQLMetadataR
```

## Usage

### As a Python Module

You can use the [`SQLExplorer`](explorer/sql_explorer.py) class from the [`explorer`](explorer) module to extract metadata:

```python
from explorer.sql_explorer import SQLExplorer

# Initialize the explorer with your SQLite database path
explorer = SQLExplorer("path/to/your/database.db")

# Extract metadata:
db_metadata = explorer.extract_metadata(
    sample_rows=5,          # Number of sample rows to retrieve from each table
    max_column_values=10,   # Maximum number of distinct values to sample per column
    execute_queries=True,   # Execute example queries
    query_result_limit=5    # Limit the number of query result rows
)

# Save the metadata to a JSON file
db_metadata.save_to_file("database_metadata.json")
```

### Command-Line Interface

Run the complete workflow directly from the command line:

```bash
python -m explorer.cli path/to/your/database.db --output metadata.json
```

Command-line options include:
- `--output`, `-o`: Specify the output JSON file path
- `--sample-rows`, `-r`: Number of sample rows to include per table
- `--max-values`, `-v`: Maximum number of distinct values to sample per column
- `--no-execute`: Skip executing example queries
- `--query-results`, `-q`: Number of result rows per executed query

## Documentation

For detailed information on domain models, extraction methods, and components, refer to the sql_explorer.md file. In-code documentation and comments also provide guidance on how each module works.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes and new features.

## License

This project is licensed under the Attribution-ShareAlike 4.0 International License.

## Acknowledgements

SQLMetadataR was developed to simplify the process of understanding SQLite database structures and metadata. Thank you for using the project!