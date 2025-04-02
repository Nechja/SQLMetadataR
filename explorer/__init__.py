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

# Import semantic processor
from .semantic_processor import SemanticMetadataProcessor

__all__ = [
    # Domain models
    'Database', 'Table', 'Column', 'Index', 'ForeignKey', 
    'Relationship', 'QueryExample', 'QueryResult',
    
    # Core functionality
    'SQLExplorer',
    
    # Semantic processing
    'SemanticMetadataProcessor'
]
