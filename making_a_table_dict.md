# Creating a Table Dictionary for Semantic Enrichment

The semantic processing features of SQLMetadataR can significantly enhance the meaningfulness of your database metadata by incorporating business context. This guide explains how to create a table dictionary file that provides this business context.

## What is a Table Dictionary?

A table dictionary is a JSON file that provides business context for your database schema. It includes descriptions, domain terminology, and business rules that aren't usually captured in the technical schema.

## Table Dictionary Format

Here's the basic structure of a table dictionary:

```json
{
  "table_name": {
    "description": "A clear business description of what this table represents",
    "domain": "The business domain this table belongs to",
    "business_owner": "The team or person responsible for this data",
    "columns": {
      "column_name": "Description of what this column represents"
    },
    "business_rules": {
      "rule_name": "Description of business rule"
    },
    "domain_terms": {
      "column_name": ["synonym1", "synonym2"]
    }
  },
  "relationships": [
    {
      "parent_table": "parent_table_name",
      "child_table": "child_table_name",
      "business_meaning": "Description of what this relationship means"
    }
  ],
  "query_patterns": {
    "pattern_name": {
      "description": "Description of this query pattern",
      "patterns": [
        "How many {entity} have {attribute}?",
        "Count of {entity} by {attribute}"
      ],
      "sample_query": "SELECT COUNT(*) FROM table WHERE column = value"
    }
  }
}
```

## Example Table Dictionary

Here's a simplified example for a DVD rental database:

```json
{
  "film": {
    "description": "Contains information about films available for rental",
    "domain": "Inventory",
    "business_owner": "Content Management Team",
    "columns": {
      "film_id": "Unique identifier for each film",
      "title": "The title of the film as it appears on the DVD cover",
      "description": "A brief synopsis of the film plot",
      "release_year": "The year the film was released to theaters",
      "rental_rate": "The standard cost to rent this film",
      "length": "The duration of the film in minutes",
      "rating": "The MPAA rating of the film"
    },
    "business_rules": {
      "rating_values": "Rating must be one of: G, PG, PG-13, R, NC-17",
      "rental_rate_range": "Rental rate must be between $0.99 and $4.99"
    },
    "domain_terms": {
      "title": ["movie name", "movie title", "film name"],
      "rental_rate": ["price", "cost", "fee"]
    }
  },
  "relationships": [
    {
      "parent_table": "film",
      "child_table": "inventory",
      "business_meaning": "Each film can have multiple physical copies in inventory"
    }
  ],
  "query_patterns": {
    "films_by_category": {
      "description": "Find films by their category",
      "patterns": [
        "What {films|movies} are in the {category} category?",
        "Show me all {category} {films|movies}",
        "List {films|movies} in {category}"
      ],
      "sample_query": "SELECT f.title FROM film f JOIN film_category fc ON f.film_id = fc.film_id JOIN category c ON fc.category_id = c.category_id WHERE c.name = '{category}'"
    }
  }
}
```

## How to Create Your Table Dictionary

1. Start with the basic structure above
2. Add entries for your most important tables 
3. Focus on providing business descriptions for columns that might be ambiguous
4. Add common domain terminology and synonyms
5. Document important business rules
6. Save the file as `database_name_table_dict.json` in the same directory as your database

## Using Your Table Dictionary

When you run SQLMetadataR, it will automatically look for a table dictionary file with the naming pattern `database_name_table_dict.json`. You can also specify a custom path with the `--table-dict` parameter:

```bash
python -m explorer.cli your_database.db --table-dict your_custom_table_dict.json
```

The semantic processor will use this information to generate rich, context-aware descriptions that better represent the business meaning of your data.