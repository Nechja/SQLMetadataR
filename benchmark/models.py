from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
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

@dataclass
class VisualizationType:
    """Represents a type of data visualization."""
    name: str
    description: str
    suitable_data_structures: List[str]  # e.g., "time series", "categorical", "hierarchical"
    question_intent: List[str]  # e.g., "comparison", "trend", "distribution"
    chart_type: str  # e.g., "bar", "line", "pie", "scatter", "heatmap"
    chart_library: Optional[str] = None  # e.g., "matplotlib", "plotly", "seaborn"
    
@dataclass
class VisualizationRecommendation:
    """Recommendation for a data visualization of query results."""
    query_id: str
    natural_language: str
    sql_query: str
    visualization_types: List[VisualizationType]
    selected_visualization: Optional[str] = None
    user_notes: Optional[str] = None
    
@dataclass
class VisualizationAssessment:
    """Collection of visualization recommendations."""
    database_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[VisualizationRecommendation] = field(default_factory=list)
    
    def add_recommendation(self, recommendation: VisualizationRecommendation) -> None:
        """Add a new visualization recommendation."""
        self.recommendations.append(recommendation)
        
    def to_dict(self) -> Dict:
        """Convert assessment to a dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "timestamp": self.timestamp.isoformat(),
            "recommendations": [
                {
                    "query_id": r.query_id,
                    "natural_language": r.natural_language,
                    "sql_query": r.sql_query,
                    "visualization_types": [
                        {
                            "name": vt.name,
                            "description": vt.description,
                            "suitable_data_structures": vt.suitable_data_structures,
                            "question_intent": vt.question_intent,
                            "chart_type": vt.chart_type,
                            "chart_library": vt.chart_library
                        } for vt in r.visualization_types
                    ],
                    "selected_visualization": r.selected_visualization,
                    "user_notes": r.user_notes
                } for r in self.recommendations
            ]
        }

@dataclass
class BusinessRule:
    """Represents a business rule related to database entities."""
    rule_id: str
    description: str
    tables_affected: List[str]
    columns_affected: List[str]
    rule_type: str  # e.g., "constraint", "validation", "calculation"
    source: str  # e.g., "user_input", "ai_suggestion", "documentation"

@dataclass
class DomainTerm:
    """Represents a domain-specific term and its synonyms."""
    term: str
    definition: str
    synonyms: List[str]
    tables_related: List[str]
    columns_related: List[str]
    source: str  # e.g., "user_input", "ai_suggestion", "domain_expert"

@dataclass
class QueryPattern:
    """Represents a common query pattern in the domain."""
    pattern_id: str
    description: str
    natural_language_templates: List[str]
    sql_template: str
    tables_used: List[str]
    complexity: str  # "simple", "medium", "advanced"
    frequency: str  # "common", "occasional", "rare"
    source: str  # e.g., "user_input", "ai_suggestion", "analytics"

@dataclass
class KnowledgeBase:
    """Container for enhanced domain knowledge."""
    database_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    business_rules: List[BusinessRule] = field(default_factory=list)
    domain_terms: List[DomainTerm] = field(default_factory=list)
    query_patterns: List[QueryPattern] = field(default_factory=list)
    
    def add_business_rule(self, rule: BusinessRule) -> None:
        """Add a new business rule."""
        self.business_rules.append(rule)
        
    def add_domain_term(self, term: DomainTerm) -> None:
        """Add a new domain term."""
        self.domain_terms.append(term)
        
    def add_query_pattern(self, pattern: QueryPattern) -> None:
        """Add a new query pattern."""
        self.query_patterns.append(pattern)
        
    def to_dict(self) -> Dict:
        """Convert knowledge base to a dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "timestamp": self.timestamp.isoformat(),
            "business_rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "tables_affected": r.tables_affected,
                    "columns_affected": r.columns_affected,
                    "rule_type": r.rule_type,
                    "source": r.source
                } for r in self.business_rules
            ],
            "domain_terms": [
                {
                    "term": t.term,
                    "definition": t.definition,
                    "synonyms": t.synonyms,
                    "tables_related": t.tables_related,
                    "columns_related": t.columns_related,
                    "source": t.source
                } for t in self.domain_terms
            ],
            "query_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "description": p.description,
                    "natural_language_templates": p.natural_language_templates,
                    "sql_template": p.sql_template,
                    "tables_used": p.tables_used,
                    "complexity": p.complexity,
                    "frequency": p.frequency,
                    "source": p.source
                } for p in self.query_patterns
            ]
        }

@dataclass
class SemanticEnrichment:
    """Container for semantic enrichment data."""
    database_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    keyword_extractions: Dict[str, List[str]] = field(default_factory=dict)  # query_id -> keywords
    phrase_alternatives: Dict[str, List[str]] = field(default_factory=dict)  # phrase -> alternatives
    term_mappings: Dict[str, str] = field(default_factory=dict)  # domain term -> database entity
    embedding_metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    
    def add_keyword_extraction(self, query_id: str, keywords: List[str]) -> None:
        """Add keywords for a query pattern."""
        self.keyword_extractions[query_id] = keywords
        
    def add_phrase_alternative(self, phrase: str, alternatives: List[str]) -> None:
        """Add alternative phrasings for a term or phrase."""
        self.phrase_alternatives[phrase] = alternatives
        
    def add_term_mapping(self, domain_term: str, database_entity: str) -> None:
        """Map a domain term to a database entity."""
        self.term_mappings[domain_term] = database_entity
        
    def to_dict(self) -> Dict:
        """Convert semantic enrichment to a dictionary for serialization."""
        return {
            "database_name": self.database_name,
            "timestamp": self.timestamp.isoformat(),
            "keyword_extractions": self.keyword_extractions,
            "phrase_alternatives": self.phrase_alternatives,
            "term_mappings": self.term_mappings,
            "embedding_metadata": self.embedding_metadata,
            "embedding_model": self.embedding_model
        }