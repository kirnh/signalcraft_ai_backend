"""
Pydantic schemas for structured outputs from agents.
Similar to LangChain's structured output approach.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List

# Load environment variables
load_dotenv()

# Get configurable limits
MIN_SENTIMENT_TOKENS_PER_ARTICLE = int(os.getenv("MIN_SENTIMENT_TOKENS_PER_ARTICLE", "5"))
MAX_SENTIMENT_TOKENS_PER_ARTICLE = int(os.getenv("MAX_SENTIMENT_TOKENS_PER_ARTICLE", "15"))


# Entity Enrichment Agent Output Schema
class RelatedEntity(BaseModel):
    """A single related entity with relationship information."""
    entity_name: str = Field(description="Name of the related entity (company, person, or organization)")
    relationship_strength: float = Field(
        description="Strength of relationship from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )
    relationship_type: str = Field(
        description="Type of relationship: 'competitor', 'supplier', 'executive', 'partner', 'investor', 'customer'"
    )


class EntityEnrichmentOutput(BaseModel):
    """Output schema for Entity Enrichment Agent."""
    company_name: str = Field(description="The company being analyzed")
    entities: List[RelatedEntity] = Field(
        description="List of related entities with relationship information",
        min_length=1
    )


# News Aggregation Agent Output Schema
class NewsArticleBasic(BaseModel):
    """A single news article (without sentiment analysis - used in Step 2)."""
    url: str = Field(description="URL of the news article")
    published_date: str = Field(description="Publication date in ISO format")
    source: str = Field(description="Source of the news (e.g., 'Reuters', 'Bloomberg')")
    title: str = Field(description="Title of the article", default="")


class EntityWithNews(BaseModel):
    """An entity with its associated news articles."""
    entity_name: str = Field(description="Name of the entity")
    relationship_strength: float = Field(ge=0.0, le=1.0)
    relationship_type: str
    news: List[NewsArticleBasic] = Field(description="List of news articles about this entity")


class NewsAggregationOutput(BaseModel):
    """Output schema for News Aggregation Agent."""
    company_name: str = Field(description="The company being analyzed")
    entities: List[EntityWithNews] = Field(
        description="List of entities with their news articles"
    )


# Sentiment Analysis Agent Output Schema
class SentimentToken(BaseModel):
    """A sentiment signal extracted from news."""
    model_config = ConfigDict(populate_by_name=True)  # Allow both token_text and tokenText
    
    token_text: str = Field(
        description="The key phrase or event from the news",
        alias="tokenText"
    )
    impact: str = Field(
        description="Impact on the main company: 'positive', 'negative', or 'neutral'"
    )
    direction: str = Field(
        description="Trading signal direction: 'bullish', 'bearish', or 'neutral'"
    )
    strength: float = Field(
        description="Strength of the signal from 0.0 to 1.0",
        ge=0.0,
        le=1.0
    )


class NewsArticle(BaseModel):
    """A single news article with sentiment analysis (used in Step 3)."""
    url: str = Field(description="URL of the news article")
    published_date: str = Field(description="Publication date in ISO format")
    source: str = Field(description="Source of the news (e.g., 'Reuters', 'Bloomberg')")
    title: str = Field(description="Title of the article", default="")
    sentiment_tokens: List[SentimentToken] = Field(
        description=f"List of sentiment signals extracted from this article. MUST contain at least {MIN_SENTIMENT_TOKENS_PER_ARTICLE} distinct sentiment tokens per article. Each token should represent a different aspect, event, or implication from the article.",
        min_length=MIN_SENTIMENT_TOKENS_PER_ARTICLE,
        max_length=MAX_SENTIMENT_TOKENS_PER_ARTICLE
    )


class EntityWithSentiment(BaseModel):
    """An entity with news articles, each containing sentiment analysis."""
    entity_name: str
    relationship_strength: float = Field(ge=0.0, le=1.0)
    relationship_type: str
    news: List[NewsArticle] = Field(
        description="List of news articles, each with its own sentiment tokens"
    )


class SentimentAnalysisOutput(BaseModel):
    """Output schema for Sentiment Analysis Agent."""
    company_name: str = Field(description="The company being analyzed")
    entities: List[EntityWithSentiment] = Field(
        description="List of entities with news and sentiment analysis"
    )


# Single Entity/Article Processing Schemas (for parallel agent processing)
class SingleEntityNewsOutput(BaseModel):
    """Output schema for a single entity news aggregation (Step 2 - parallel processing)."""
    company_name: str = Field(description="The company being analyzed")
    entity: EntityWithNews = Field(description="Single entity with its news articles")


class SingleArticleSentimentOutput(BaseModel):
    """Output schema for a single article sentiment analysis (Step 3 - parallel processing)."""
    company_name: str = Field(description="The company being analyzed")
    entity_name: str = Field(description="Name of the entity this article belongs to")
    relationship_strength: float = Field(ge=0.0, le=1.0, description="Relationship strength of the entity")
    relationship_type: str = Field(description="Relationship type of the entity")
    article: NewsArticle = Field(description="Single news article with sentiment tokens")


# Example usage and validation
if __name__ == "__main__":
    # Test the schemas
    import json
    
    # Test EntityEnrichmentOutput
    enrichment_data = {
        "company_name": "Apple",
        "entities": [
            {
                "entity_name": "TSMC",
                "relationship_strength": 0.95,
                "relationship_type": "supplier"
            },
            {
                "entity_name": "Samsung",
                "relationship_strength": 0.85,
                "relationship_type": "competitor"
            }
        ]
    }
    
    enrichment_output = EntityEnrichmentOutput(**enrichment_data)
    print("✓ EntityEnrichmentOutput validated")
    print(json.dumps(enrichment_output.model_dump(), indent=2))
    
    # Test NewsAggregationOutput (Step 2 - no sentiment_tokens)
    news_data = {
        "company_name": "Apple",
        "entities": [
            {
                "entity_name": "TSMC",
                "relationship_strength": 0.95,
                "relationship_type": "supplier",
                "news": [
                    {
                        "url": "https://example.com/article",
                        "published_date": "2024-11-07",
                        "source": "Reuters",
                        "title": "TSMC expands capacity"
                    }
                ]
            }
        ]
    }
    
    news_output = NewsAggregationOutput(**news_data)
    print("\n✓ NewsAggregationOutput validated")
    print(json.dumps(news_output.model_dump(), indent=2))
    
    # Test SentimentAnalysisOutput (Step 3 - with MULTIPLE sentiment_tokens per article, minimum 3)
    sentiment_data = {
        "company_name": "Apple",
        "entities": [
            {
                "entity_name": "TSMC",
                "relationship_strength": 0.95,
                "relationship_type": "supplier",
                "news": [
                    {
                        "url": "https://example.com/article",
                        "published_date": "2024-11-07",
                        "source": "Reuters",
                        "title": "TSMC expands",
                        "sentiment_tokens": [
                            {
                                "tokenText": "TSMC expands production capacity by 30%",
                                "impact": "positive",
                                "direction": "bullish",
                                "strength": 0.75
                            },
                            {
                                "tokenText": "Increased capacity reduces Apple supply chain risk",
                                "impact": "positive",
                                "direction": "bullish",
                                "strength": 0.7
                            },
                            {
                                "tokenText": "Expansion costs may lead to 5% higher chip prices",
                                "impact": "negative",
                                "direction": "bearish",
                                "strength": 0.6
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    sentiment_output = SentimentAnalysisOutput(**sentiment_data)
    print("\n✓ SentimentAnalysisOutput validated")
    print(json.dumps(sentiment_output.model_dump(), indent=2))
    
    print("\n✅ All schemas validated successfully!")

