# SQLMetadataR Documentation

SQLMetadataR is a tool for extracting and analyzing metadata from SQLite databases. It provides a comprehensive view of database structure, relationships, and example queries that can be run against the database.

## Domain Models

The system uses the following domain models to represent database metadata:

### Core Models

#### `Database`
Represents an entire SQLite database.
- **Properties**:
  - `name`: The database filename
  - `path`: Full path to the database file
  - `size_bytes`: Size of the database file in bytes
  - `tables`: Dictionary of tables (key: table name, value: Table object)
  - `relationships`: List of relationships between tables
  - `query_examples`: List of example queries for the database
- **Methods**:
  - `add_table(table)`: Add a table to the database
  - `add_relationship(relationship)`: Add a relationship to the database
  - `to_dict()`: Convert the database metadata to a dictionary
  - `save_to_file(output_path)`: Save the metadata to a JSON file

#### `Table`
Represents a database table.
- **Properties**:
  - `name`: Table name
  - `columns`: List of Column objects
  - `primary_keys`: List of primary key column names
  - `foreign_keys`: List of ForeignKey objects
  - `indexes`: List of Index objects
  - `row_count`: Number of rows in the table
  - `sample_data`: Sample rows from the table
- **Methods**:
  - `add_column(column)`: Add a column to the table
  - `to_dict()`: Convert the table metadata to a dictionary

#### `Column`
Represents a table column.
- **Properties**:
  - `name`: Column name
  - `data_type`: SQL data type
  - `not_null`: Whether the column disallows NULL values
  - `default_value`: Default value for the column
  - `is_primary_key`: Whether the column is part of the primary key
  - `statistics`: Dictionary of column statistics (null count, distinct count, etc.)

#### `Index`
Represents a database index.
- **Properties**:
  - `name`: Index name
  - `columns`: List of column names included in the index
  - `unique`: Whether the index enforces uniqueness

#### `ForeignKey`
Represents a foreign key relationship.
- **Properties**:
  - `from_column`: Column name in the current table
  - `to_table`: Referenced table name
  - `to_column`: Referenced column name

#### `Relationship`
Represents a relationship between two tables.
- **Properties**:
  - `from_table`: Source table name
  - `from_column`: Source column name
  - `to_table`: Target table name
  - `to_column`: Target column name
  - `relationship_type`: Type of relationship (one_to_one, one_to_many, many_to_one, many_to_many)

### Query-Related Models

#### `QueryExample`
Represents an example SQL query for the database.
- **Properties**:
  - `description`: Description of what the query does
  - `query`: The SQL query text
  - `complexity`: Complexity level (simple, medium, advanced)
  - `execution_result`: QueryResult object with the result of executing the query

#### `QueryResult`
Represents the result of executing a query.
- **Properties**:
  - `success`: Whether the query executed successfully
  - `error`: Error message (if any)
  - `sample_results`: List of result rows (as dictionaries)
  - `row_count`: Total number of rows in the result
  - `columns`: List of column names in the result

## Components

### `SQLExplorer`
Main orchestrator class that coordinates the metadata extraction process.

**Responsibilities**:
- Initialize and manage database connections
- Coordinate the extraction of metadata
- Delegate specific tasks to specialized components
- Provide a simple API for metadata extraction

**Main Methods**:
- `extract_metadata()`: Extract comprehensive metadata from an SQLite database
- `run_workflow()`: Run the complete metadata extraction workflow and save to file

### `TableExtractor`
Responsible for extracting metadata from database tables.

**Responsibilities**:
- Extract column information
- Extract row counts
- Extract indexes and foreign keys
- Extract sample data
- Compute column statistics

**Main Methods**:
- `extract_table()`: Extract all metadata for a specific table
- Various private methods for extracting specific aspects of a table

### `RelationshipAnalyzer`
Analyzes relationships between tables in the database.

**Responsibilities**:
- Extract relationships based on foreign keys
- Determine relationship types (one-to-one, one-to-many, etc.)

**Main Methods**:
- `analyze_relationships()`: Extract and analyze relationships between tables
- `_extract_relationships()`: Extract relationships from foreign keys
- `_determine_relationship_types()`: Determine the cardinality of each relationship

### `QueryGenerator`
Generates example SQL queries for the database.

**Responsibilities**:
- Generate simple queries (SELECT, WHERE, ORDER BY)
- Generate medium complexity queries (JOINs, aggregates)
- Generate advanced queries (multi-joins, subqueries, CTEs)

**Main Methods**:
- `generate_queries()`: Generate all example queries for the database
- Various private methods for generating specific types of queries

### `QueryExecutor`
Executes SQL queries and formats the results.

**Responsibilities**:
- Execute SQL queries against the database
- Format results for JSON serialization
- Handle query errors gracefully

**Main Methods**:
- `execute_queries()`: Execute all generated queries and store results
- `execute_query()`: Execute a single query and return the result