from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import os
import json

@dataclass
class Column:
    name: str
    data_type: str
    not_null: bool = False
    default_value: Any = None
    is_primary_key: bool = False
    statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Index:
    name: str
    columns: List[str]
    unique: bool = False

@dataclass
class ForeignKey:
    from_column: str
    to_table: str
    to_column: str

@dataclass
class Relationship:
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str = "many_to_one"  # one_to_one, one_to_many, many_to_many

@dataclass
class QueryResult:
    success: bool = False
    error: Optional[str] = None
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    row_count: Optional[Union[int, str]] = None
    columns: List[str] = field(default_factory=list)

@dataclass
class QueryExample:
    description: str
    query: str
    complexity: str  # simple, medium, advanced
    execution_result: Optional[QueryResult] = None

@dataclass
class Table:
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKey] = field(default_factory=list)
    indexes: List[Index] = field(default_factory=list)
    row_count: int = 0
    sample_data: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_column(self, column: Column) -> None:
        self.columns.append(column)
        if column.is_primary_key:
            self.primary_keys.append(column.name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [vars(col) for col in self.columns],
            "primary_keys": self.primary_keys,
            "foreign_keys": [vars(fk) for fk in self.foreign_keys],
            "indexes": [vars(idx) for idx in self.indexes],
            "row_count": self.row_count,
            "sample_data": self.sample_data,
            "column_statistics": {col.name: col.statistics for col in self.columns if col.statistics}
        }

@dataclass
class Database:
    name: str
    path: str
    size_bytes: int
    tables: Dict[str, Table] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    query_examples: List[QueryExample] = field(default_factory=list)
    extraction_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)
    
    def add_table(self, table: Table) -> None:
        self.tables[table.name] = table
    
    def add_relationship(self, relationship: Relationship) -> None:
        self.relationships.append(relationship)
    
    def add_query_example(self, query_example: QueryExample) -> None:
        self.query_examples.append(query_example)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_info": {
                "name": self.name,
                "path": self.path,
                "size_bytes": self.size_bytes,
                "size_mb": self.size_mb,
                "extraction_time": self.extraction_time
            },
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "relationships": [vars(rel) for rel in self.relationships],
            "query_examples": [
                {
                    "description": qe.description,
                    "query": qe.query,
                    "complexity": qe.complexity,
                    "execution_result": vars(qe.execution_result) if qe.execution_result else None
                } 
                for qe in self.query_examples
            ]
        }
    
    def extract_schema_dict(self) -> Dict[str, Any]:
        """Extract only schema information from this Database object."""
        schema = {
            "database_info": {
                "name": self.name,
                "path": self.path,
                "extraction_time": self.extraction_time
            },
            "tables": {}
        }
        
        for table_name, table in self.tables.items():
            schema["tables"][table_name] = {
                "name": table.name,
                "columns": [
                    {
                        "name": col.name,
                        "data_type": col.data_type,
                        "not_null": col.not_null,
                        "default_value": col.default_value,
                        "is_primary_key": col.is_primary_key
                    }
                    for col in table.columns
                ],
                "primary_keys": table.primary_keys,
                "foreign_keys": [
                    {
                        "from_column": fk.from_column,
                        "to_table": fk.to_table,
                        "to_column": fk.to_column
                    }
                    for fk in table.foreign_keys
                ],
                "indexes": [
                    {
                        "name": idx.name,
                        "columns": idx.columns,
                        "unique": idx.unique
                    }
                    for idx in table.indexes
                ]
            }
        
        # Include relationships between tables
        schema["relationships"] = [
            {
                "from_table": rel.from_table,
                "from_column": rel.from_column,
                "to_table": rel.to_table,
                "to_column": rel.to_column,
                "relationship_type": rel.relationship_type
            }
            for rel in self.relationships
        ]
        
        return schema

    def save_schema_to_file(self, output_path: str) -> None:
        """Save only the schema information to a JSON file."""
        schema_dict = self.extract_schema_dict()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, default=str)
        
        print(f"Schema saved to: {output_path}")
    
    def save_to_file(self, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        print(f"Metadata saved to: {output_path}")
