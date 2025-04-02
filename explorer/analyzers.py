from typing import List, Dict

from .models import Database, Relationship

class RelationshipAnalyzer:
    """
    Responsible for analyzing and determining relationships between tables.
    This includes identifying foreign keys and determining relationship types.
    """
    def analyze_relationships(self, db: Database) -> None:
        """
        Extract and analyze relationships between tables in the database.
        """
        self._extract_relationships(db)
        self._determine_relationship_types(db)
    
    def _extract_relationships(self, db: Database) -> None:
        """Extract relationships based on foreign keys"""
        for table_name, table in db.tables.items():
            for fk in table.foreign_keys:
                relationship = Relationship(
                    from_table=table_name,
                    from_column=fk.from_column,
                    to_table=fk.to_table,
                    to_column=fk.to_column
                )
                
                db.add_relationship(relationship)
    
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
