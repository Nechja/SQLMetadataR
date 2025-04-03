import json
import os
from typing import Dict, List, Any, Optional, Union
import re

from .models import Database, Table, Column, Relationship


class SemanticMetadataProcessor:
    """
    Processes database metadata and business context information to create
    rich semantic descriptions for embedding generation in NLP-to-SQL systems.
    
    This component bridges the technical database schema with business meaning
    to enable more natural language interactions with the database.
    """
    
    def __init__(self, database: Database, table_dict_path: Optional[str] = None, table_dict: Optional[Dict] = None):
        """
        Initialize with either a database metadata object and a table dictionary
        
        Args:
            database: The Database object containing technical metadata
            table_dict_path: Path to the table dictionary JSON file (business context)
            table_dict: Dictionary containing table business context (alternative to table_dict_path)
        """
        self.database = database
        self.business_context = {}
        
        # Load business context
        if table_dict:
            self.business_context = table_dict
        elif table_dict_path and os.path.exists(table_dict_path):
            try:
                with open(table_dict_path, 'r', encoding='utf-8') as f:
                    self.business_context = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load table dictionary: {e}")
        
        # Processed data will be stored here
        self.semantic_context = {
            "database_info": {},
            "tables": {},
            "columns": {},
            "relationships": [],
            "query_examples": []
        }
        
        # Process the metadata
        self._process_metadata()
    
    def _process_metadata(self) -> None:
        """Process the database metadata and merge with business context"""
        # Set database info
        self.semantic_context["database_info"] = {
            "name": self.database.name,
            "path": self.database.path,
            "size_bytes": self.database.size_bytes,
            "extraction_time": self.database.extraction_time
        }
        
        # Process tables
        self._process_tables()
        
        # Process relationships
        self._process_relationships()
        
        # Process query examples
        self._process_query_examples()
    
    def _process_tables(self) -> None:
        """Process table metadata and merge with business context"""
        for table_name, table in self.database.tables.items():
            # Initialize table context
            table_context = {
                "name": table.name,
                "description": "",
                "domain": "",
                "business_owner": "",
                "technical_details": {
                    "primary_keys": table.primary_keys,
                    "row_count": table.row_count,
                    "columns": [],
                    "indexes": [idx.__dict__ for idx in table.indexes]
                },
                "business_rules": {},
                "metrics": {},
                "domain_terms": {},
                "columns": {}
            }
            
            # Merge with business context if available - now case-insensitive
            business_table = self._find_business_table(table_name)
            if business_table:
                table_context["description"] = business_table.get("description", f"Table containing {table_name} data")
                table_context["domain"] = business_table.get("domain", "")
                table_context["business_owner"] = business_table.get("business_owner", "")
                table_context["business_rules"] = business_table.get("business_rules", {})
                table_context["metrics"] = business_table.get("metrics", {})
                table_context["domain_terms"] = business_table.get("domain_terms", {})
            else:
                # Generate default description
                table_context["description"] = f"Table containing {table_name} data"
            
            # Process columns
            self._process_columns(table, table_context, table_name)
            
            # Add to semantic context
            self.semantic_context["tables"][table_name] = table_context
    
    def _process_columns(self, table: Table, table_context: Dict, table_name: str) -> None:
        """Process column metadata and merge with business context"""
        # Get business table using case-insensitive lookup
        business_table = self._find_business_table(table_name)
        business_columns = business_table.get("columns", {}) if business_table else {}
        
        for column in table.columns:
            # Initialize column context
            column_context = {
                "name": column.name,
                "description": "",
                "data_type": column.data_type,
                "not_null": column.not_null,
                "default_value": column.default_value,
                "is_primary_key": column.is_primary_key,
                "statistics": column.statistics,
                "business_meaning": "",
                "domain_terms": []
            }
            
            # Add technical details to table
            table_context["technical_details"]["columns"].append({
                "name": column.name,
                "data_type": column.data_type,
                "not_null": column.not_null,
                "is_primary_key": column.is_primary_key
            })
            
            # Case-insensitive column lookup
            column_info = self._find_business_column(business_columns, column.name)
            
            # Merge with business context if available
            if column_info:
                column_context["description"] = column_info
                column_context["business_meaning"] = column_info
            else:
                # Generate default description
                column_context["description"] = f"{column.name} ({column.data_type})"
            
            # Add domain terms for the column - case insensitive
            if business_table:
                domain_terms = business_table.get("domain_terms", {})
                column_domain_terms = self._find_business_column(domain_terms, column.name)
                if column_domain_terms:
                    column_context["domain_terms"] = column_domain_terms
            
            # Add to table context
            table_context["columns"][column.name] = column_context
            
            # Add to global columns context with fully qualified name
            self.semantic_context["columns"][f"{table_name}.{column.name}"] = column_context
    
    def _process_relationships(self) -> None:
        """Process relationship metadata and merge with business context"""
        for rel in self.database.relationships:
            # Initialize relationship context
            rel_context = {
                "from_table": rel.from_table,
                "from_column": rel.from_column,
                "to_table": rel.to_table,
                "to_column": rel.to_column,
                "relationship_type": rel.relationship_type,
                "business_meaning": ""
            }
            
            # Add business meaning from business context if available - case insensitive
            relationships = self.business_context.get("relationships", [])
            for business_rel in relationships:
                parent_table = business_rel.get("parent_table", "")
                child_table = business_rel.get("child_table", "")
                
                if (parent_table.lower() == rel.to_table.lower() and 
                    child_table.lower() == rel.from_table.lower()):
                    rel_context["business_meaning"] = business_rel.get("business_meaning", "")
                    break
            
            # Generate default business meaning if not found
            if not rel_context["business_meaning"]:
                if rel.relationship_type == "many_to_one":
                    rel_context["business_meaning"] = f"Many {rel.from_table} records relate to one {rel.to_table} record"
                elif rel.relationship_type == "one_to_many":
                    rel_context["business_meaning"] = f"One {rel.from_table} record relates to many {rel.to_table} records"
                elif rel.relationship_type == "one_to_one":
                    rel_context["business_meaning"] = f"One {rel.from_table} record relates to one {rel.to_table} record"
                else:
                    rel_context["business_meaning"] = f"Relationship between {rel.from_table} and {rel.to_table}"
            
            # Add to semantic context
            self.semantic_context["relationships"].append(rel_context)
    
    def _process_query_examples(self) -> None:
        """Process query examples and merge with query patterns from business context"""
        # First add query examples from the database
        for qe in self.database.query_examples:
            # Initialize query example
            query_context = {
                "description": qe.description,
                "query": qe.query,
                "complexity": qe.complexity,
                "natural_language_variations": [qe.description],
                "execution_result": None if not qe.execution_result else {
                    "success": qe.execution_result.success,
                    "row_count": qe.execution_result.row_count,
                    "sample_results": qe.execution_result.sample_results[:3] if qe.execution_result.sample_results else []
                }
            }
            
            # Add to semantic context
            self.semantic_context["query_examples"].append(query_context)
        
        # Then add query patterns from business context
        query_patterns = self.business_context.get("query_patterns", {})
        for pattern_name, pattern in query_patterns.items():
            # Initialize query pattern
            query_context = {
                "description": pattern.get("description", ""),
                "query": pattern.get("sample_query", ""),
                "complexity": "medium",  # Default complexity
                "natural_language_variations": pattern.get("patterns", []),
                "execution_result": None
            }
            
            # Add to semantic context
            self.semantic_context["query_examples"].append(query_context)
    
    def generate_nl_descriptions(self) -> Dict[str, str]:
        """Generate natural language descriptions for all database objects"""
        descriptions = {}
        
        # Database description
        db_name = self.semantic_context["database_info"].get("name", "")
        descriptions["database"] = f"Database '{db_name}' contains {len(self.semantic_context['tables'])} tables with information about {', '.join(list(self.semantic_context['tables'].keys())[:5])} and more."
        
        # Table descriptions
        for table_name, table in self.semantic_context["tables"].items():
            description = f"Table '{table_name}': {table['description']} "
            
            # Add primary key info
            if table["technical_details"]["primary_keys"]:
                description += f"Primary key(s): {', '.join(table['technical_details']['primary_keys'])}. "
            
            # Add domain and owner info if available
            if table["domain"]:
                description += f"Domain: {table['domain']}. "
            if table["business_owner"]:
                description += f"Owned by: {table['business_owner']}. "
            
            # Add column count
            description += f"Contains {len(table['columns'])} columns. "
            
            # Add row count if available
            if table["technical_details"]["row_count"] > 0:
                description += f"Has {table['technical_details']['row_count']} rows. "
            
            # Add business rules if available
            if table["business_rules"]:
                rule_descriptions = [f"{rule}: {desc}" for rule, desc in list(table["business_rules"].items())[:3]]
                description += f"Business rules include: {'; '.join(rule_descriptions)}. "
            
            descriptions[table_name] = description
        
        # Column descriptions
        for full_col_name, column in self.semantic_context["columns"].items():
            table_name, col_name = full_col_name.split(".")
            description = f"Column '{col_name}' in table '{table_name}': {column['description']} "
            
            # Add technical details
            description += f"Type: {column['data_type']}. "
            if column["is_primary_key"]:
                description += "Is a primary key. "
            if column["not_null"]:
                description += "Cannot be null. "
            if column["default_value"]:
                description += f"Default value: {column['default_value']}. "
            
            # Add statistics if available
            stats = column.get("statistics", {})
            if stats.get("distinct_count") is not None:
                description += f"Has {stats['distinct_count']} distinct values. "
            if stats.get("null_count") is not None:
                description += f"Contains {stats['null_count']} null values. "
            if stats.get("min") is not None and stats.get("max") is not None:
                description += f"Range: {stats['min']} to {stats['max']}. "
            
            # Add domain terms if available
            if column["domain_terms"]:
                description += f"Also known as: {', '.join(column['domain_terms'])}. "
            
            descriptions[full_col_name] = description
        
        # Relationship descriptions
        for i, rel in enumerate(self.semantic_context["relationships"]):
            rel_id = f"relationship_{i+1}"
            description = f"Relationship: {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']} "
            description += f"Type: {rel['relationship_type']}. "
            if rel["business_meaning"]:
                description += f"Meaning: {rel['business_meaning']}."
            
            descriptions[rel_id] = description
        
        # Query descriptions
        for i, query in enumerate(self.semantic_context["query_examples"]):
            query_id = f"query_{i+1}"
            description = f"Query: {query['description']} "
            description += f"Complexity: {query['complexity']}. "
            if query["natural_language_variations"] and len(query["natural_language_variations"]) > 1:
                description += f"Can be asked as: {'; '.join(query['natural_language_variations'])}. "
            description += f"SQL: {query['query']}"
            
            descriptions[query_id] = description
        
        return descriptions
    
    def generate_embedding_contexts(self) -> List[Dict[str, Any]]:
        """
        Generate contexts suitable for embeddings for each database object.
        
        Each context contains all relevant information about a database object
        in a format that can be used to generate embeddings for semantic search.
        """
        contexts = []
        
        # Table contexts
        for table_name, table in self.semantic_context["tables"].items():
            # Basic table info
            table_context = {
                "object_type": "table",
                "name": table_name,
                "description": table["description"],
                "domain": table["domain"],
                "business_owner": table["business_owner"],
                "primary_keys": table["technical_details"]["primary_keys"],
                "row_count": table["technical_details"]["row_count"],
                "column_count": len(table["columns"]),
                "business_rules": table["business_rules"],
                "column_names": list(table["columns"].keys()),
                "related_tables": self._get_related_tables(table_name),
                "query_examples": self._get_query_examples_for_table(table_name),
                "domain_terms": table["domain_terms"]
            }
            
            # Generate a rich text description
            table_context["text_representation"] = self._generate_table_text(table_name, table)
            
            contexts.append(table_context)
        
        # Column contexts
        for full_col_name, column in self.semantic_context["columns"].items():
            table_name, col_name = full_col_name.split(".")
            
            # Basic column info
            column_context = {
                "object_type": "column",
                "name": col_name,
                "full_name": full_col_name,
                "table_name": table_name,
                "description": column["description"],
                "data_type": column["data_type"],
                "not_null": column["not_null"],
                "is_primary_key": column["is_primary_key"],
                "default_value": column["default_value"],
                "statistics": column["statistics"],
                "domain_terms": column["domain_terms"],
                "related_columns": self._get_related_columns(table_name, col_name),
                "query_examples": self._get_query_examples_for_column(table_name, col_name)
            }
            
            # Generate a rich text description
            column_context["text_representation"] = self._generate_column_text(full_col_name, column, table_name)
            
            contexts.append(column_context)
        
        # Relationship contexts
        for i, rel in enumerate(self.semantic_context["relationships"]):
            # Basic relationship info
            rel_context = {
                "object_type": "relationship",
                "id": f"relationship_{i+1}",
                "from_table": rel["from_table"],
                "from_column": rel["from_column"],
                "to_table": rel["to_table"],
                "to_column": rel["to_column"],
                "relationship_type": rel["relationship_type"],
                "business_meaning": rel["business_meaning"],
                "query_examples": self._get_query_examples_for_relationship(rel["from_table"], rel["to_table"])
            }
            
            # Generate a rich text description
            rel_context["text_representation"] = self._generate_relationship_text(rel)
            
            contexts.append(rel_context)
        
        # Query contexts
        for i, query in enumerate(self.semantic_context["query_examples"]):
            # Basic query info
            query_context = {
                "object_type": "query",
                "id": f"query_{i+1}",
                "description": query["description"],
                "complexity": query["complexity"],
                "sql": query["query"],
                "natural_language_variations": query["natural_language_variations"],
                "tables_used": self._extract_tables_from_query(query["query"]),
                "columns_used": self._extract_columns_from_query(query["query"])
            }
            
            # Generate a rich text description
            query_context["text_representation"] = self._generate_query_text(query)
            
            contexts.append(query_context)
            
        return contexts
    
    def _get_related_tables(self, table_name: str) -> List[Dict[str, str]]:
        """Get tables related to a specific table through relationships"""
        related = []
        
        for rel in self.semantic_context["relationships"]:
            if rel["from_table"] == table_name:
                related.append({
                    "table": rel["to_table"],
                    "relationship": f"{rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}",
                    "type": rel["relationship_type"]
                })
            elif rel["to_table"] == table_name:
                related.append({
                    "table": rel["from_table"],
                    "relationship": f"{rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}",
                    "type": rel["relationship_type"]
                })
        
        return related
    
    def _get_related_columns(self, table_name: str, column_name: str) -> List[Dict[str, str]]:
        """Get columns related to a specific column through relationships"""
        related = []
        
        for rel in self.semantic_context["relationships"]:
            if rel["from_table"] == table_name and rel["from_column"] == column_name:
                related.append({
                    "table": rel["to_table"],
                    "column": rel["to_column"],
                    "relationship": f"{rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}",
                    "type": rel["relationship_type"]
                })
            elif rel["to_table"] == table_name and rel["to_column"] == column_name:
                related.append({
                    "table": rel["from_table"],
                    "column": rel["from_column"],
                    "relationship": f"{rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}",
                    "type": rel["relationship_type"]
                })
        
        return related
    
    def _get_query_examples_for_table(self, table_name: str) -> List[Dict[str, str]]:
        """Get query examples that involve a specific table"""
        examples = []
        
        for query in self.semantic_context["query_examples"]:
            if table_name.lower() in query["query"].lower():
                examples.append({
                    "description": query["description"],
                    "query": query["query"]
                })
        
        return examples
    
    def _get_query_examples_for_column(self, table_name: str, column_name: str) -> List[Dict[str, str]]:
        """Get query examples that involve a specific column"""
        examples = []
        
        for query in self.semantic_context["query_examples"]:
            # Look for table.column or just column if it's unambiguous
            if f"{table_name}.{column_name}".lower() in query["query"].lower() or \
               f"{column_name}".lower() in query["query"].lower():
                examples.append({
                    "description": query["description"],
                    "query": query["query"]
                })
        
        return examples
    
    def _get_query_examples_for_relationship(self, table1: str, table2: str) -> List[Dict[str, str]]:
        """Get query examples that involve a relationship between two tables"""
        examples = []
        
        for query in self.semantic_context["query_examples"]:
            if table1.lower() in query["query"].lower() and table2.lower() in query["query"].lower():
                examples.append({
                    "description": query["description"],
                    "query": query["query"]
                })
        
        return examples
    
    def _extract_tables_from_query(self, query: str) -> List[str]:
        """Extract table names from a SQL query"""
        # Simple approach using regex to find table names after FROM and JOIN
        tables = []
        
        # First, clean up the query by removing quotes
        clean_query = query.replace('"', '').replace('\'', '')
        
        # Look for tables after FROM
        from_matches = re.findall(r'\bFROM\s+([a-zA-Z0-9_]+)', clean_query, re.IGNORECASE)
        if from_matches:
            tables.extend(from_matches)
        
        # Look for tables after JOIN
        join_matches = re.findall(r'\bJOIN\s+([a-zA-Z0-9_]+)', clean_query, re.IGNORECASE)
        if join_matches:
            tables.extend(join_matches)
        
        # Remove duplicates and return
        return list(set(tables))
    
    def _extract_columns_from_query(self, query: str) -> List[str]:
        """Extract column names from a SQL query"""
        # Simple approach using regex to find column names after SELECT, WHERE, GROUP BY, etc.
        columns = []
        
        # First, clean up the query by removing quotes
        clean_query = query.replace('"', '').replace('\'', '')
        
        # Look for columns after SELECT
        select_part = re.search(r'\bSELECT\s+(.*?)\s+FROM\b', clean_query, re.IGNORECASE | re.DOTALL)
        if select_part:
            select_columns = select_part.group(1)
            # Handle * case
            if '*' in select_columns:
                pass  # All columns, can't be specific
            else:
                # Split by commas and extract column names
                for col in select_columns.split(','):
                    col = col.strip()
                    # Handle aliases with AS keyword
                    as_match = re.search(r'(.*?)\s+AS\s+', col, re.IGNORECASE)
                    if as_match:
                        col = as_match.group(1).strip()
                    
                    # Handle table.column notation
                    if '.' in col:
                        columns.append(col)
                    else:
                        # Simple column name
                        if col and not col.startswith('(') and col.lower() not in ('distinct', 'count', 'sum', 'avg', 'min', 'max'):
                            columns.append(col)
        
        # Look for columns after WHERE, GROUP BY, ORDER BY
        clause_columns = re.findall(r'\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING)\s+(.*?)(?:\bLIMIT\b|\bHAVING\b|$|\))', clean_query, re.IGNORECASE | re.DOTALL)
        for clause, clause_content in clause_columns:
            for col in clause_content.split(','):
                col = col.strip()
                # Extract column name from conditions
                col_match = re.search(r'([a-zA-Z0-9_.]+)\s*(=|>|<|>=|<=|!=|LIKE|IN|NOT|IS)', col, re.IGNORECASE)
                if col_match:
                    columns.append(col_match.group(1).strip())
                else:
                    # If no operator, just use the whole thing if it looks like a column
                    if col and not col.startswith('(') and col.lower() not in ('and', 'or', 'not', 'null', 'is'):
                        columns.append(col)
        
        # Remove duplicates and return
        return list(set(columns))
    
    def _generate_table_text(self, table_name: str, table: Dict) -> str:
        """Generate rich text description for a table"""
        text = f"Table: {table_name}\n\n"
        text += f"Description: {table['description']}\n"
        
        if table["domain"]:
            text += f"Domain: {table['domain']}\n"
        
        if table["business_owner"]:
            text += f"Business Owner: {table['business_owner']}\n"
        
        text += f"Primary Keys: {', '.join(table['technical_details']['primary_keys'])}\n"
        text += f"Row Count: {table['technical_details']['row_count']}\n\n"
        
        text += "Columns:\n"
        for col_name, col in table["columns"].items():
            text += f"- {col_name} ({col['data_type']}): {col['description']}\n"
        
        if table["business_rules"]:
            text += "\nBusiness Rules:\n"
            for rule_name, rule_desc in table["business_rules"].items():
                text += f"- {rule_name}: {rule_desc}\n"
        
        related_tables = self._get_related_tables(table_name)
        if related_tables:
            text += "\nRelated Tables:\n"
            for rel in related_tables:
                text += f"- {rel['table']} ({rel['type']}): {rel['relationship']}\n"
        
        if table["domain_terms"]:
            text += "\nAlternative Terms:\n"
            for term, synonyms in table["domain_terms"].items():
                text += f"- {term}: {', '.join(synonyms)}\n"
        
        return text
    
    def _generate_column_text(self, full_col_name: str, column: Dict, table_name: str) -> str:
        """Generate rich text description for a column"""
        table_desc = self.semantic_context["tables"][table_name]["description"]
        _, col_name = full_col_name.split(".")
        
        text = f"Column: {full_col_name}\n\n"
        text += f"Description: {column['description']}\n"
        text += f"Table: {table_name} - {table_desc}\n"
        text += f"Data Type: {column['data_type']}\n"
        
        constraints = []
        if column["is_primary_key"]:
            constraints.append("Primary Key")
        if column["not_null"]:
            constraints.append("Not Null")
        if column["default_value"]:
            constraints.append(f"Default: {column['default_value']}")
        
        if constraints:
            text += f"Constraints: {', '.join(constraints)}\n"
        
        stats = column.get("statistics", {})
        if stats:
            text += "\nStatistics:\n"
            if stats.get("distinct_count") is not None:
                text += f"- Distinct Values: {stats['distinct_count']}\n"
            if stats.get("null_count") is not None:
                text += f"- Null Values: {stats['null_count']}\n"
            if stats.get("min") is not None and stats.get("max") is not None:
                text += f"- Range: {stats['min']} to {stats['max']}\n"
            if stats.get("avg") is not None:
                text += f"- Average: {stats['avg']}\n"
            if stats.get("sample_values") and len(stats["sample_values"]) > 0:
                sample_str = ', '.join([str(v) for v in stats["sample_values"][:5]])
                text += f"- Sample Values: {sample_str}\n"
        
        related_columns = self._get_related_columns(table_name, col_name)
        if related_columns:
            text += "\nRelated Columns:\n"
            for rel in related_columns:
                text += f"- {rel['table']}.{rel['column']} ({rel['type']}): {rel['relationship']}\n"
        
        if column["domain_terms"]:
            text += "\nAlternative Terms: " + ", ".join(column["domain_terms"]) + "\n"
        
        return text
    
    def _generate_relationship_text(self, rel: Dict) -> str:
        """Generate rich text description for a relationship"""
        text = f"Relationship: {rel['from_table']}.{rel['from_column']} → {rel['to_table']}.{rel['to_column']}\n\n"
        text += f"Type: {rel['relationship_type']}\n"
        
        if rel["business_meaning"]:
            text += f"Business Meaning: {rel['business_meaning']}\n"
        
        # Add descriptions of the related tables
        from_table_desc = self.semantic_context["tables"].get(rel["from_table"], {}).get("description", "")
        to_table_desc = self.semantic_context["tables"].get(rel["to_table"], {}).get("description", "")
        
        text += f"\nFrom Table: {rel['from_table']} - {from_table_desc}\n"
        text += f"To Table: {rel['to_table']} - {to_table_desc}\n"
        
        # Add descriptions of the related columns
        from_col_desc = self.semantic_context["columns"].get(f"{rel['from_table']}.{rel['from_column']}", {}).get("description", "")
        to_col_desc = self.semantic_context["columns"].get(f"{rel['to_table']}.{rel['to_column']}", {}).get("description", "")
        
        text += f"\nFrom Column: {rel['from_column']} - {from_col_desc}\n"
        text += f"To Column: {rel['to_column']} - {to_col_desc}\n"
        
        return text
    
    def _generate_query_text(self, query: Dict) -> str:
        """Generate rich text description for a query"""
        text = f"Query: {query['description']}\n\n"
        text += f"Complexity: {query['complexity']}\n"
        
        if query["natural_language_variations"]:
            text += "\nCan be asked as:\n"
            for variation in query["natural_language_variations"]:
                text += f"- {variation}\n"
        
        text += f"\nSQL Query:\n{query['query']}\n"
        
        # Add information about tables and columns used
        tables_used = self._extract_tables_from_query(query["query"])
        if tables_used:
            text += "\nTables used:\n"
            for table in tables_used:
                table_desc = self.semantic_context["tables"].get(table, {}).get("description", "")
                text += f"- {table}: {table_desc}\n"
        
        columns_used = self._extract_columns_from_query(query["query"])
        if columns_used:
            text += "\nColumns used:\n"
            for column in columns_used:
                # Handle both table.column and just column notations
                if "." in column:
                    col_desc = self.semantic_context["columns"].get(column, {}).get("description", "")
                    text += f"- {column}: {col_desc}\n"
                else:
                    # Just a column name, could be ambiguous
                    text += f"- {column}\n"
        
        # Add execution results if available
        if query.get("execution_result"):
            result = query["execution_result"]
            if result["success"]:
                text += f"\nExecution: Successful\n"
                text += f"Row Count: {result['row_count']}\n"
                
                if result["sample_results"]:
                    text += "\nSample Results:\n"
                    for i, row in enumerate(result["sample_results"][:3]):
                        text += f"Row {i+1}: {str(row)}\n"
            else:
                text += "\nExecution: Failed\n"
        
        return text
    
    def save_to_file(self, output_path: str) -> None:
        """Save the semantic context to a JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.semantic_context, f, indent=2, default=str)
        
        print(f"Semantic context saved to: {output_path}")
    
    def save_embeddings_data(self, output_path: str) -> None:
        """Save the embedding context data to a JSON file"""
        contexts = self.generate_embedding_contexts()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, indent=2, default=str)
        
        print(f"Embedding data saved to: {output_path}")
    
    def save_nl_descriptions(self, output_path: str) -> None:
        """Save natural language descriptions to a JSON file"""
        descriptions = self.generate_nl_descriptions()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, indent=2, default=str)
        
        print(f"Natural language descriptions saved to: {output_path}")
    
    def _find_business_table(self, table_name: str) -> Dict:
        """Find a table in business context, case-insensitive"""
        if table_name in self.business_context:
            return self.business_context.get(table_name, {})
        
        # Try case-insensitive lookup
        for key in self.business_context:
            if key.lower() == table_name.lower():
                return self.business_context.get(key, {})
                
        return {}
    
    def _find_business_column(self, columns_dict: Dict, column_name: str) -> Any:
        """Find a column in columns dictionary, case-insensitive"""
        if column_name in columns_dict:
            return columns_dict.get(column_name)
        
        # Try case-insensitive lookup
        for key in columns_dict:
            if key.lower() == column_name.lower():
                return columns_dict.get(key)
                
        return None
