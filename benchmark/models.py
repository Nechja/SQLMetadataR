from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime

@dataclass
class QueryParameter:
    """Represents parameters that might change between similar queries."""
    name: str
    value: Any
    description: Optional[str] = None

@dataclass
class NlpSqlQuery:
    """Represents a single NLP to SQL query pattern."""
    id: str
    natural_language: List[str]
    sql: str
    description: str
    tables: List[str]
    params: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None
    
    def get_params_as_objects(self) -> List[QueryParameter]:
        """Convert the params dictionary to a list of QueryParameter objects."""
        return [QueryParameter(name=k, value=v) for k, v in self.params.items()]

@dataclass
class MetaData:
    """Metadata about the NLP to SQL training dataset."""
    database: str
    version: str
    created: datetime
    description: str

@dataclass
class NlpSqlTrainingData:
    """Root container for NLP to SQL training data."""
    query_patterns: List[NlpSqlQuery]
    meta: MetaData
    
    @classmethod
    def from_json(cls, json_data: Dict) -> "NlpSqlTrainingData":
        """Create NlpSqlTrainingData from a JSON dictionary."""
        meta = MetaData(
            database=json_data["meta"]["database"],
            version=json_data["meta"]["version"],
            created=datetime.fromisoformat(json_data["meta"]["created"]),
            description=json_data["meta"]["description"]
        )
        
        query_patterns = []
        for pattern in json_data["query_patterns"]:
            query = NlpSqlQuery(
                id=pattern["id"],
                natural_language=pattern["natural_language"],
                sql=pattern["sql"],
                description=pattern["description"],
                tables=pattern["tables"],
                params=pattern["params"],
                explanation=pattern.get("explanation")
            )
            query_patterns.append(query)
            
        return cls(query_patterns=query_patterns, meta=meta)
    
    def add_query_pattern(self, query: NlpSqlQuery) -> None:
        """Add a new query pattern to the training data."""
        self.query_patterns.append(query)
    
    def find_query_by_id(self, query_id: str) -> Optional[NlpSqlQuery]:
        """Find a query pattern by its ID."""
        for query in self.query_patterns:
            if query.id == query_id:
                return query
        return None
    
    def find_queries_by_table(self, table_name: str) -> List[NlpSqlQuery]:
        """Find all query patterns that involve a specific table."""
        return [q for q in self.query_patterns if table_name in q.tables]

    def find_similar_queries(self, natural_language_text: str) -> List[NlpSqlQuery]:
        """Simple implementation to find queries with similar natural language text.
        In a real implementation, you would use embeddings or other NLP techniques here.
        """
        matches = []
        for query in self.query_patterns:
            for nl_variant in query.natural_language:
                if natural_language_text.lower() in nl_variant.lower() or nl_variant.lower() in natural_language_text.lower():
                    matches.append(query)
                    break
        return matches