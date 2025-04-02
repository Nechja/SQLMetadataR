import sqlite3
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback

from .models import Database, Table, Column, Index, ForeignKey, Relationship, QueryExample, QueryResult

class SQLExplorer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file '{db_path}' not found.")
        
        self.conn = None
        self.cursor = None
    
    def connect(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def extract_metadata(self, sample_rows: int = 3, max_column_values: int = 10, 
                        execute_queries: bool = True, query_result_limit: int = 5) -> Database:
        """Extract metadata from the SQLite database"""
        try:
            self.connect()
            
            # Create Database object
            db = Database(
                name=os.path.basename(self.db_path),
                path=self.db_path,
                size_bytes=os.path.getsize(self.db_path)
            )
            
            # Get list of tables
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            table_names = [table[0] for table in self.cursor.fetchall()]
            
            # Process each table
            for table_name in table_names:
                print(f"Processing table: {table_name}")
                table = self._extract_table_metadata(table_name, sample_rows, max_column_values)
                db.add_table(table)
            
            # Extract relationships between tables
            self._extract_relationships(db)
            
            # Generate query examples
            self._generate_query_examples(db)
            
            # Execute queries if requested
            if execute_queries:
                self._execute_queries(db, query_result_limit)
            
            return db
        
        finally:
            self.close()
    
    def _extract_table_metadata(self, table_name: str, sample_rows: int, max_column_values: int) -> Table:
        """Extract metadata for a specific table"""
        table = Table(name=table_name)
        
        # Get column information
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        columns_data = self.cursor.fetchall()
        
        for col_data in columns_data:
            col_id, name, data_type, not_null, default_val, is_pk = col_data
            
            column = Column(
                name=name,
                data_type=data_type,
                not_null=bool(not_null),
                default_value=default_val,
                is_primary_key=bool(is_pk)
            )
            
            table.add_column(column)
        
        # Get row count
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            table.row_count = self.cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Error getting row count for {table_name}: {e}")
            table.row_count = -1
        
        # Get index information
        self._extract_indexes(table)
        
        # Get foreign keys
        self._extract_foreign_keys(table)
        
        # Get sample data
        self._extract_sample_data(table, sample_rows)
        
        # Get column statistics
        self._extract_column_statistics(table, max_column_values)
        
        return table
    
    def _extract_indexes(self, table: Table) -> None:
        """Extract index information for a table"""
        self.cursor.execute(f"PRAGMA index_list({table.name})")
        indices = self.cursor.fetchall()
        
        for idx in indices:
            idx_id, idx_name, idx_unique = idx[0], idx[1], bool(idx[2])
            
            self.cursor.execute(f"PRAGMA index_info({idx_name})")
            idx_columns = [row[2] for row in self.cursor.fetchall()]
            
            index = Index(
                name=idx_name,
                columns=idx_columns,
                unique=idx_unique
            )
            
            table.indexes.append(index)
    
    def _extract_foreign_keys(self, table: Table) -> None:
        """Extract foreign key information for a table"""
        self.cursor.execute(f"PRAGMA foreign_key_list({table.name})")
        fks = self.cursor.fetchall()
        
        for fk in fks:
            fk_id, seq, ref_table, from_col, to_col = fk[0], fk[1], fk[2], fk[3], fk[4]
            
            foreign_key = ForeignKey(
                from_column=from_col,
                to_table=ref_table,
                to_column=to_col
            )
            
            table.foreign_keys.append(foreign_key)
    
    def _extract_sample_data(self, table: Table, sample_rows: int) -> None:
        """Extract sample data for a table"""
        try:
            self.cursor.execute(f"SELECT * FROM {table.name} LIMIT {sample_rows}")
            rows = self.cursor.fetchall()
            
            if rows:
                column_names = [description[0] for description in self.cursor.description]
                
                for row in rows:
                    row_dict = {column: row[column] for column in column_names}
                    table.sample_data.append(row_dict)
        except sqlite3.Error as e:
            print(f"Error fetching sample data for {table.name}: {e}")
    
    def _extract_column_statistics(self, table: Table, max_column_values: int) -> None:
        """Extract statistics for each column in a table"""
        for column in table.columns:
            col_name = column.name
            col_stats = {
                "null_count": None,
                "distinct_count": None,
                "sample_values": []
            }
            
            try:
                self.cursor.execute(f"""
                    SELECT COUNT(*) 
                    FROM {table.name} 
                    WHERE {col_name} IS NULL
                """)
                null_count = self.cursor.fetchone()[0]
                col_stats["null_count"] = null_count
            except sqlite3.Error:
                pass
            
            try:
                self.cursor.execute(f"""
                    SELECT COUNT(DISTINCT {col_name}) 
                    FROM {table.name}
                """)
                distinct_count = self.cursor.fetchone()[0]
                col_stats["distinct_count"] = distinct_count
                
                self.cursor.execute(f"""
                    SELECT DISTINCT {col_name}
                    FROM {table.name}
                    WHERE {col_name} IS NOT NULL
                    LIMIT {max_column_values}
                """)
                sample_values = [row[0] for row in self.cursor.fetchall()]
                col_stats["sample_values"] = sample_values
            except sqlite3.Error:
                pass
            
            try:
                if column.data_type.lower() in ('int', 'integer', 'real', 'float', 'numeric', 'decimal'):
                    self.cursor.execute(f"""
                        SELECT 
                            MIN({col_name}),
                            MAX({col_name}),
                            AVG({col_name})
                        FROM {table.name}
                        WHERE {col_name} IS NOT NULL
                    """)
                    min_val, max_val, avg_val = self.cursor.fetchone()
                    
                    col_stats["min"] = min_val
                    col_stats["max"] = max_val
                    col_stats["avg"] = avg_val
            except sqlite3.Error:
                pass
            
            column.statistics = col_stats
    
    def _extract_relationships(self, db: Database) -> None:
        """Extract relationships between tables"""
        for table_name, table in db.tables.items():
            for fk in table.foreign_keys:
                relationship = Relationship(
                    from_table=table_name,
                    from_column=fk.from_column,
                    to_table=fk.to_table,
                    to_column=fk.to_column
                )
                
                db.add_relationship(relationship)
        
        # Determine relationship types
        self._determine_relationship_types(db)
    
    def _determine_relationship_types(self, db: Database) -> None:
        """Determine the type of each relationship"""
        for relationship in db.relationships:
            from_table = relationship.from_table
            from_column = relationship.from_column
            to_table = relationship.to_table
            to_column = relationship.to_column
            
            # Check if columns are unique (primary key or unique index)
            is_to_unique = to_column in db.tables[to_table].primary_keys
            if not is_to_unique:
                for index in db.tables[to_table].indexes:
                    if index.unique and len(index.columns) == 1 and index.columns[0] == to_column:
                        is_to_unique = True
                        break
            
            is_from_unique = from_column in db.tables[from_table].primary_keys
            if not is_from_unique:
                for index in db.tables[from_table].indexes:
                    if index.unique and len(index.columns) == 1 and index.columns[0] == from_column:
                        is_from_unique = True
                        break
            
            if is_to_unique and is_from_unique:
                relationship.relationship_type = "one_to_one"
            elif is_to_unique:
                relationship.relationship_type = "many_to_one"
            elif is_from_unique:
                relationship.relationship_type = "one_to_many"
            else:
                relationship.relationship_type = "many_to_many"
    
    def _generate_query_examples(self, db: Database) -> None:
        """Generate example queries for the database"""
        self._generate_simple_queries(db)
        self._generate_medium_queries(db)
        self._generate_advanced_queries(db)
    
    def _generate_simple_queries(self, db: Database) -> None:
        """Generate simple queries for each table"""
        for table_name, table in db.tables.items():
            # Basic SELECT
            db.add_query_example(QueryExample(
                description=f"Select all data from {table_name}",
                query=f"SELECT * FROM {table_name} LIMIT 10;",
                complexity="simple"
            ))
            
            # SELECT with WHERE clause if table has columns with sample values
            for col in table.columns:
                if col.statistics and col.statistics.get("sample_values") and len(col.statistics["sample_values"]) > 0:
                    sample_value = col.statistics["sample_values"][0]
                    if isinstance(sample_value, str):
                        formatted_value = f"'{sample_value}'"
                    else:
                        formatted_value = str(sample_value) if sample_value is not None else "NULL"
                    
                    db.add_query_example(QueryExample(
                        description=f"Filter {table_name} by {col.name}",
                        query=f"SELECT * FROM {table_name} WHERE {col.name} = {formatted_value} LIMIT 10;",
                        complexity="simple"
                    ))
                    break  # Just one example is enough
            
            # ORDER BY query
            if table.columns:
                db.add_query_example(QueryExample(
                    description=f"Order {table_name} by {table.columns[0].name}",
                    query=f"SELECT * FROM {table_name} ORDER BY {table.columns[0].name} DESC LIMIT 10;",
                    complexity="simple"
                ))
    
    def _generate_medium_queries(self, db: Database) -> None:
        """Generate medium complexity queries like joins and aggregates"""
        # Simple joins based on relationships
        for relationship in db.relationships:
            from_table = relationship.from_table
            from_column = relationship.from_column
            to_table = relationship.to_table
            to_column = relationship.to_column
            
            # Basic join
            db.add_query_example(QueryExample(
                description=f"Join {from_table} with {to_table}",
                query=f"""SELECT t1.*, t2.*
FROM {from_table} t1
JOIN {to_table} t2 ON t1.{from_column} = t2.{to_column}
LIMIT 10;""",
                complexity="medium"
            ))
            
            # Join with WHERE clause
            db.add_query_example(QueryExample(
                description=f"Join {from_table} with {to_table} and filter results",
                query=f"""SELECT t1.*, t2.*
FROM {from_table} t1
JOIN {to_table} t2 ON t1.{from_column} = t2.{to_column}
WHERE t1.{from_column} IS NOT NULL
LIMIT 10;""",
                complexity="medium"
            ))
        
        # Aggregate functions
        for table_name, table in db.tables.items():
            numeric_columns = [col.name for col in table.columns 
                               if col.data_type.lower() in ('int', 'integer', 'real', 'float', 'numeric', 'decimal')]
            categorical_columns = [col.name for col in table.columns 
                                  if col.data_type.lower() in ('text', 'varchar', 'char', 'string')]
            
            if numeric_columns and len(numeric_columns) > 0:
                db.add_query_example(QueryExample(
                    description=f"Aggregate statistics for {table_name}",
                    query=f"""SELECT 
    COUNT(*) as count,
    AVG({numeric_columns[0]}) as average,
    SUM({numeric_columns[0]}) as total,
    MIN({numeric_columns[0]}) as minimum,
    MAX({numeric_columns[0]}) as maximum
FROM {table_name};""",
                    complexity="medium"
                ))
            
            if categorical_columns and len(categorical_columns) > 0:
                db.add_query_example(QueryExample(
                    description=f"Group by query for {table_name}",
                    query=f"""SELECT 
    {categorical_columns[0]},
    COUNT(*) as count
FROM {table_name}
GROUP BY {categorical_columns[0]}
ORDER BY count DESC
LIMIT 10;""",
                    complexity="medium"
                ))
    
    def _generate_advanced_queries(self, db: Database) -> None:
        """Generate advanced queries like complex joins, subqueries and CTEs"""
        table_names = list(db.tables.keys())
        
        # Find connected tables that can be joined in a chain
        table_graph = {}
        for relationship in db.relationships:
            from_table = relationship.from_table
            to_table = relationship.to_table
            
            if from_table not in table_graph:
                table_graph[from_table] = []
            if to_table not in table_graph:
                table_graph[to_table] = []
            
            table_graph[from_table].append((to_table, relationship.from_column, relationship.to_column))
            table_graph[to_table].append((from_table, relationship.to_column, relationship.from_column))
        
        # Generate multi-join queries
        for start_table in table_graph:
            visited = {table: False for table in table_graph}
            path = []
            
            def dfs(current_table, depth=0, max_depth=3):
                nonlocal path
                if depth >= max_depth:
                    return
                
                visited[current_table] = True
                path.append(current_table)
                
                # If we have at least 3 tables in our path, generate a multi-join query
                if len(path) >= 3:
                    tables_in_path = path.copy()
                    self._generate_multi_join_query(db, tables_in_path)
                
                for neighbor, from_col, to_col in table_graph.get(current_table, []):
                    if not visited.get(neighbor, True):  # If we haven't visited this neighbor
                        dfs(neighbor, depth + 1, max_depth)
                
                path.pop()
                visited[current_table] = False
            
            dfs(start_table)
        
        # Subqueries
        for table_name, table in db.tables.items():
            # Find a table that can be used in a subquery
            for rel in db.relationships:
                if rel.to_table == table_name:
                    related_table = rel.from_table
                    related_col = rel.from_column
                    target_col = rel.to_column
                    
                    # Subquery in WHERE clause
                    db.add_query_example(QueryExample(
                        description=f"Subquery filtering records from {table_name}",
                        query=f"""SELECT *
FROM {table_name}
WHERE {target_col} IN (
    SELECT {related_col}
    FROM {related_table}
    LIMIT 100
)
LIMIT 10;""",
                        complexity="advanced"
                    ))
                    
                    # EXISTS subquery
                    db.add_query_example(QueryExample(
                        description=f"EXISTS subquery with {table_name} and {related_table}",
                        query=f"""SELECT *
FROM {table_name} t1
WHERE EXISTS (
    SELECT 1
    FROM {related_table} t2
    WHERE t2.{related_col} = t1.{target_col}
)
LIMIT 10;""",
                        complexity="advanced"
                    ))
                    
                    break
        
        # Common Table Expressions (CTEs)
        if table_names:
            table = db.tables[table_names[0]]
            if len(table.columns) >= 2:
                columns = [col.name for col in table.columns]
                # Simple CTE
                db.add_query_example(QueryExample(
                    description=f"Common Table Expression (CTE) on {table.name}",
                    query=f"""WITH {table.name}_cte AS (
    SELECT {columns[0]}, {columns[1]}
    FROM {table.name}
    WHERE {columns[0]} IS NOT NULL
)
SELECT *
FROM {table.name}_cte
LIMIT 10;""",
                    complexity="advanced"
                ))
        
        # UNION, INTERSECT, EXCEPT
        if len(table_names) >= 2:
            table1_name = table_names[0]
            table2_name = table_names[1]
            
            # Find common column types between tables
            cols1 = {col.name: col.data_type for col in db.tables[table1_name].columns}
            cols2 = {col.name: col.data_type for col in db.tables[table2_name].columns}
            
            common_cols = []
            for col1_name, col1_type in cols1.items():
                for col2_name, col2_type in cols2.items():
                    if col1_type == col2_type:
                        common_cols.append((col1_name, col2_name))
                        break
            
            if common_cols:
                col1_name, col2_name = common_cols[0]
                
                db.add_query_example(QueryExample(
                    description=f"UNION query combining results from {table1_name} and {table2_name}",
                    query=f"""SELECT '{table1_name}' AS source, {col1_name} AS value
FROM {table1_name}
WHERE {col1_name} IS NOT NULL

UNION

SELECT '{table2_name}' AS source, {col2_name} AS value
FROM {table2_name}
WHERE {col2_name} IS NOT NULL
LIMIT 10;""",
                    complexity="advanced"
                ))
    
    def _generate_multi_join_query(self, db: Database, tables_path: List[str]) -> None:
        """Generate a query that joins multiple tables"""
        if len(tables_path) < 3:
            return
        
        join_clauses = []
        for i in range(len(tables_path) - 1):
            t1 = tables_path[i]
            t2 = tables_path[i + 1]
            
            # Find the relationship between t1 and t2
            join_col_t1 = None
            join_col_t2 = None
            
            for rel in db.relationships:
                if (rel.from_table == t1 and rel.to_table == t2):
                    join_col_t1 = rel.from_column
                    join_col_t2 = rel.to_column
                    break
                elif (rel.from_table == t2 and rel.to_table == t1):
                    join_col_t1 = rel.to_column
                    join_col_t2 = rel.from_column
                    break
            
            if join_col_t1 and join_col_t2:
                join_clauses.append(f"JOIN {t2} t{i+2} ON t{i+1}.{join_col_t1} = t{i+2}.{join_col_t2}")
        
        if join_clauses:
            # Select a few columns from each table
            select_clauses = []
            for i, table in enumerate(tables_path):
                cols = [col.name for col in db.tables[table].columns]
                if cols:
                    # Take up to 2 columns from each table
                    selected_cols = cols[:min(2, len(cols))]
                    for col in selected_cols:
                        select_clauses.append(f"t{i+1}.{col}")
            
            if select_clauses:
                query = f"""SELECT {', '.join(select_clauses)}
FROM {tables_path[0]} t1
{' '.join(join_clauses)}
LIMIT 10;"""
                
                db.add_query_example(QueryExample(
                    description=f"Complex join across {len(tables_path)} tables: {', '.join(tables_path)}",
                    query=query,
                    complexity="advanced"
                ))
    
    def _execute_queries(self, db: Database, query_result_limit: int) -> None:
        """Execute the generated queries and store results"""
        print("Executing queries to test and capture results...")
        
        self.connect()
        try:
            for i, query_example in enumerate(db.query_examples):
                print(f"Executing query {i+1}/{len(db.query_examples)}...")
                query_result = self._execute_query(query_example.query, query_result_limit)
                query_example.execution_result = query_result
        finally:
            self.close()
    
    def _execute_query(self, query: str, result_limit: int = 5) -> QueryResult:
        """Execute a query and return the result or error message"""
        result = QueryResult()
        
        try:
            self.cursor.execute(query)
            rows = self.cursor.fetchmany(result_limit + 1)  # Fetch one extra to see if there are more
            
            # Convert results to a list of dictionaries for JSON serialization
            has_more_rows = len(rows) > result_limit
            displayed_rows = rows[:result_limit]
            
            for row in displayed_rows:
                row_dict = {}
                for idx, col in enumerate(self.cursor.description):
                    col_name = col[0]
                    value = row[idx]
                    # Handle non-serializable types
                    if isinstance(value, (datetime,)):
                        value = value.isoformat()
                    row_dict[col_name] = value
                result.sample_results.append(row_dict)
            
            if not has_more_rows:
                # We fetched all rows, report actual count
                result.row_count = len(rows)
            else:
                # There are more rows than we fetched, try to get accurate count
                try:
                    # Use original query to get count
                    count_query = f"SELECT COUNT(*) FROM ({query.rstrip(';')}) AS query_count"
                    self.cursor.execute(count_query)
                    result.row_count = self.cursor.fetchone()[0]
                except:
                    # If count query fails, indicate that we have at least the number we fetched
                    result.row_count = f">{result_limit}"
            
            result.success = True
            result.columns = [col[0] for col in self.cursor.description] if self.cursor.description else []
        
        except Exception as e:
            result.success = False
            result.error = str(e)
        
        return result
