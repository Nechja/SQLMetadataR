import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser as JsonOutputParser
import json
import logging

logging.basicConfig(level=logging.DEBUG)

# Define paths
# Use absolute paths to ensure files are found correctly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "Datasets", "project.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "jsons", "project_schema.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "output_table_dict.json")
API_KEY_PATH = os.path.join(BASE_DIR, "azureai.info")

# Load your API config
def load_api_key():
    """Load Azure OpenAI API keys and config from azureai.info file"""
    azure_config = {}
    
    # Print the path for debugging
    print(f"Looking for API key file at: {API_KEY_PATH}")
    
    if not os.path.exists(API_KEY_PATH):
        print(f"Error: API key file not found at {API_KEY_PATH}")
        print("Please make sure 'azureai.info' exists in the project root directory.")
        exit(1)
    
    try:
        with open(API_KEY_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse key-value pairs from the file
        for line in lines:
            line = line.strip()
            if line and not line.startswith('//'):  # Skip comments and empty lines
                if '=' in line:
                    key, value = line.split('=', 1)
                    azure_config[key.strip()] = value.strip()
        
        # Fix the endpoint format - extract the base URL without the path and query parameters
        if "endpoint" in azure_config and "/deployments/" in azure_config["endpoint"]:
            # Extract just the base URL (up to .com)
            base_url = azure_config["endpoint"].split("/openai")[0]
            azure_config["endpoint"] = base_url
        
        # Verify required keys exist
        required_keys = ["endpoint", "api_key", "deployment_id", "api_version"]
        missing_keys = [key for key in required_keys if key not in azure_config]
        
        if missing_keys:
            print(f"Error: Missing required Azure OpenAI configuration keys: {', '.join(missing_keys)}")
            print("Please check your azureai.info file format")
            exit(1)
            
        print("API configuration loaded successfully!")
        return azure_config
    except Exception as e:
        print(f"Error loading API key: {e}")
        exit(1)


config = load_api_key()

# Initialize Azure OpenAI LLM
# Use a direct adapter to avoid LangChain's default parameters
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import BaseModel

class CustomAzureChatOpenAI(BaseChatModel, BaseModel):
    deployment_id: str
    api_version: str
    endpoint: str
    api_key: str

    class Config:
        extra = "allow"  # allow extra attributes such as 'client'

    def __init__(self, **data):
        super().__init__(**data)
        from openai import AzureOpenAI
        object.__setattr__(self, 'client', AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        ))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            # Map SystemMessage to role 'user' instead of 'system'
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
        
        # Call API without passing the 'stop' parameter, with error handling for resource not found
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=openai_messages,
                **{k: v for k, v in kwargs.items() if k != 'stop'}
            )
        except Exception as e:
            print(f"API call failed: {e}. Please check your deployment id and azure_endpoint.")
            exit(1)
        
        # Convert response back to LangChain format
        message = AIMessage(content=response.choices[0].message.content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self):
        return "azure-openai"

# Use the custom implementation
llm = CustomAzureChatOpenAI(
    deployment_id=config["deployment_id"],  # pass deployment_id from config
    api_version=config["api_version"],
    endpoint=config["endpoint"],
    api_key=config["api_key"]
)

# Connect to the database
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Create SQL chain to help explore the database
sql_chain = create_sql_query_chain(llm, db)

# Load schema
with open(SCHEMA_PATH, 'r') as f:
    schema = json.load(f)

# Function to explore a table
def explore_table(table_name):
    # Get sample data
    query = f"What kind of data is in the {table_name} table? Show me a few samples."
    response = sql_chain.invoke({"question": query})
    
    # Get interesting patterns
    exploration_query = f"Generate 3 interesting SQL queries that would reveal business patterns in the {table_name} table."
    sql_queries = llm.invoke([
        SystemMessage(content="You are a SQL expert. Generate only SQL queries without explanation."),
        HumanMessage(content=exploration_query)
    ])
    
    exploration_results = []
    for query in sql_queries.content.split("```sql")[1:]:
        query = query.split("```")[0].strip()
        try:
            result = db.run(query)
            exploration_results.append({"query": query, "result": result})
        except Exception as e:
            print(f"Error executing query: {e}")
    
    return {"sample_data": response, "exploration_results": exploration_results}

# Generate business context for each table
table_dict = {}
for table_name in schema['tables']:
    logging.debug(f"Exploring table: {table_name}")
    
    # Explore the table
    exploration = explore_table(table_name)
    
    # Create a prompt for business context
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst who translates technical database structures into concise business context. Return valid JSON."),
        ("human", f"""
        Based on the following database table and exploration:
        Table: {table_name}
        Schema: {schema['tables'][table_name]}
        Sample data: {exploration['sample_data']}
        Data patterns: {exploration['exploration_results']}
        
        Generate a business context JSON with the following structure:
        ```json
        {{{{
            "desc": "An explanation of what this table represents",
            "domain": "The business domain this table belongs to",
            "business_owner": "Responsible party",
            "columns": {{{{
                "column_name": "Brief meaning"
            }}}},
            "business_rules": {{{{
                "rule_name": "Short rule explanation"
            }}}},
            "domain_terms": {{{{
                "column_name": ["related term", "alt term"]
            }}}}
        }}}}
        ```
        """)
    ])
    
    logging.debug(f"Prompt (context_prompt) for {table_name}: {context_prompt}")
    try:
        parser = JsonOutputParser()
        chain = context_prompt | llm | parser
        logging.debug(f"Invoking chain for table: {table_name}")
        table_dict[table_name] = chain.invoke({})
        logging.debug(f"chain.invoke for {table_name} returned: {table_dict[table_name]}")
    except Exception as e:
        logging.exception(f"Error during chain.invoke for table '{table_name}': {e}")

# Generate relationship descriptions
relationships = []
for rel in schema.get('relationships', []):
    from_table = rel['from_table']
    to_table = rel['to_table']
    
    rel_prompt = ChatPromptTemplate.from_messages([
        ("system", "You explain database relationships in business terms."),
        ("human", f"Explain the business meaning of the relationship where {from_table} references {to_table}.")
    ])
    
    relationship_chain = rel_prompt | llm
    business_meaning = relationship_chain.invoke({})
    
    relationships.append({
        "parent_table": to_table,
        "child_table": from_table,
        "business_meaning": business_meaning.content
    })

# Generate query patterns
logging.debug("Generating query patterns...")

query_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an analytics expert who creates helpful database queries. Return valid JSON."),
    ("human", f"""
    Based on the database schema with tables: {list(schema['tables'].keys())},
    generate 15 business query patterns in JSON with the following structure:
    ```json
    {{{{
        "pname": {{{{
            "desc": "Purpose",
            "patterns": ["way1", "way2"],
            "sample_query": "SQL query",
            "explanation": "Short interpretation"
        }}}}
    }}}}
    ```
    Make sure each query pattern name (pname) is business-oriented and addresses common analytics needs. 
    Each sample query should be valid SQL that works with this schema.
    Include a diverse range of query types including aggregations, joins, filters, analytics, and business metrics.
    Cover a variety of business questions across different functional areas.
    """)
])

print("Generating query patterns...")

parser = JsonOutputParser()
query_chain = query_prompt | llm | parser

# Add error handling for query patterns generation
try:
    query_patterns = query_chain.invoke({})
    logging.debug(f"query_patterns returned: {query_patterns}")
    if not isinstance(query_patterns, dict):
        logging.warning("Query patterns not returned as a dictionary. Using empty dict instead.")
        query_patterns = {}
except Exception as e:
    logging.exception(f"Error generating query patterns: {e}")
    query_patterns = {}

print("Generating business context for tables...")

# Combine everything
final_dict = {
    **table_dict,
    "relationships": relationships,
    "query_patterns": query_patterns
}
print("Final dictionary generated.")
# Format the output more nicely - showing table names instead of the full content
table_names = list(table_dict.keys())[:5] if len(table_dict) > 5 else list(table_dict.keys())
print(f"Tables processed: {table_names}")
print(f"Relationships processed: {len(relationships)}")
print(f"Query patterns generated: {len(query_patterns)}")

# Save to output file
with open(OUTPUT_PATH, 'w') as f:
    json.dump(final_dict, f, indent=2)

print(f"Table dictionary saved to {OUTPUT_PATH}")