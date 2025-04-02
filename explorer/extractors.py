import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import Table, Column, Index, ForeignKey

class TableMetadataExtractor:
    """
    Responsible for extracting metadata from a single table.
    This includes columns, indexes, foreign keys, sample data, and statistics.
    """
    def __init__(self, connection: sqlite3.Connection):
        self.conn = connection
        self.cursor = connection.cursor()
    
    def extract_table(self, table_name: str, sample_rows: int, max_column_values: int) -> Table:
        """Extract metadata for a specific table"""
        table = Table(name=table_name)
        
        # Get column information
        self._extract_columns(table)
        
        # Get row count
        self._extract_row_count(table)
        
        # Get index information
        self._extract_indexes(table)
        
        # Get foreign keys
        self._extract_foreign_keys(table)
        
        # Get sample data
        self._extract_sample_data(table, sample_rows)
        
        # Get column statistics
        self._extract_column_statistics(table, max_column_values)
        
        return table
    
    def _extract_columns(self, table: Table) -> None:
        """Extract column information for a table"""
        self.cursor.execute(f"PRAGMA table_info({table.name})")
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
    
    def _extract_row_count(self, table: Table) -> None:
        """Extract row count for a table"""
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table.name}")
            table.row_count = self.cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Error getting row count for {table.name}: {e}")
            table.row_count = -1
    
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
            
            self._extract_null_count(table.name, col_name, col_stats)
            self._extract_distinct_values(table.name, col_name, col_stats, max_column_values)
            self._extract_numeric_stats(table.name, column, col_stats)
            
            column.statistics = col_stats
    
    def _extract_null_count(self, table_name: str, col_name: str, col_stats: Dict[str, Any]) -> None:
        """Extract null count for a column"""
        try:
            self.cursor.execute(f"""
                SELECT COUNT(*) 
                FROM {table_name} 
                WHERE {col_name} IS NULL
            """)
            null_count = self.cursor.fetchone()[0]
            col_stats["null_count"] = null_count
        except sqlite3.Error:
            pass
    
    def _extract_distinct_values(self, table_name: str, col_name: str, 
                               col_stats: Dict[str, Any], max_values: int) -> None:
        """Extract distinct value count and sample values for a column"""
        try:
            self.cursor.execute(f"""
                SELECT COUNT(DISTINCT {col_name}) 
                FROM {table_name}
            """)
            distinct_count = self.cursor.fetchone()[0]
            col_stats["distinct_count"] = distinct_count
            
            self.cursor.execute(f"""
                SELECT DISTINCT {col_name}
                FROM {table_name}
                WHERE {col_name} IS NOT NULL
                LIMIT {max_values}
            """)
            sample_values = [row[0] for row in self.cursor.fetchall()]
            col_stats["sample_values"] = sample_values
        except sqlite3.Error:
            pass
    
    def _extract_numeric_stats(self, table_name: str, column: Column, col_stats: Dict[str, Any]) -> None:
        """Extract numeric statistics for a column if applicable"""
        try:
            if column.data_type.lower() in ('int', 'integer', 'real', 'float', 'numeric', 'decimal'):
                self.cursor.execute(f"""
                    SELECT 
                        MIN({column.name}),
                        MAX({column.name}),
                        AVG({column.name})
                    FROM {table_name}
                    WHERE {column.name} IS NOT NULL
                """)
                min_val, max_val, avg_val = self.cursor.fetchone()
                
                col_stats["min"] = min_val
                col_stats["max"] = max_val
                col_stats["avg"] = avg_val
        except sqlite3.Error:
            pass
