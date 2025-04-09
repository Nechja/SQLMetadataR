# SQLMetadataR

SQLMetadataR is a tool for extracting and analyzing metadata from SQLite databases. It provides a comprehensive view of your database’s structure including tables, columns, indexes, foreign keys, and more. It also generates example SQL queries and analyzes relationships between tables.

## Features

- **Metadata Extraction**: Retrieve table structures, row counts, sample data, and column statistics.
- **Relationship Analysis**: Identify and analyze relationships based on foreign keys.
- **Query Generation and Execution**: Generate and run example SQL queries to validate extracted metadata.
- **Semantic Processing**: Augment metadata with semantic information.
- **Command-Line Interface**: Use the provided CLI to run the complete workflow from the terminal.

## Installation

1. Ensure you have Python 3.12 or later installed.
2. Clone the repository:
    ```sh
    git clone <repository_url>
    cd SQLMetadataR
    ```

## Usage

### As a Python Module

You can use the [SQLExplorer](http://_vscodecontentref_/0) class from the [explorer](http://_vscodecontentref_/1) module to extract metadata:

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