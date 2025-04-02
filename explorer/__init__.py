"""
SQLMetadataR - Tools for extracting and analyzing SQLite database metadata
"""

# Import models
from .models import (
    Database, Table, Column, Index, ForeignKey, 
    Relationship, QueryExample, QueryResult
)

# Import explorer
from .sql_explorer import SQLExplorer

__all__ = [
    # Domain models
    'Database', 'Table', 'Column', 'Index', 'ForeignKey', 
    'Relationship', 'QueryExample', 'QueryResult',
    
    # Core functionality
    'SQLExplorer'
]
