"""
FastAPI application providing granular REST API endpoints for trading signals analysis.
Provides endpoints for entities, news, and sentiment analysis.
"""

import os
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import (
    entity_enrichment_agent,
    single_article_sentiment_agent,
    reset_tool_call_counter,
    Runner
)
from schemas import (
    EntityEnrichmentOutput,
    SingleArticleSentimentOutput,
    RelatedEntity
)
from tools import _get_entity_news_internal as get_entity_news
import time

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SignalCraft AI Trading Signals API",
    description="Granular REST API for AI-powered trading signals",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class HealthResponse(BaseModel):
    status: str
    message: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "SignalCraft AI Trading Signals API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check endpoint."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gnews_key = os.getenv("GNEWS_API_KEY")
    
    return {
        "status": "healthy" if openai_key else "unhealthy",
        "message": f"API Keys: OpenAI={'✓' if openai_key else '✗'}, GNews={'✓' if gnews_key else '✗'}"
    }


@app.get("/api/entities")
async def get_entities(company: str, max: int = 10):
    """
    Get related entities for a company (Step 1 only).
    
    Args:
        company: Company name to analyze
        max: Maximum number of entities to return (default: 10)
        
    Returns:
        Array of entities with relationship information
        
    Example:
        GET /api/entities?company=Apple&max=10
        
    Response:
        [
            {
                "entity_name": "Samsung",
                "relationship_strength": 0.85,
                "relationship_type": "competitor"
            },
            ...
        ]
    """
    if not company:
        raise HTTPException(status_code=400, detail="company parameter is required")
    
    try:
        logger.info(f"Fetching entities for {company} (max={max})")
        
        # Reset tool call counters
        reset_tool_call_counter()
        
        # Run entity enrichment agent
        runner = Runner()
        enrichment_result = await runner.run(
            entity_enrichment_agent,
            input=json.dumps({"company_name": company})
        )
        
        enrichment_data = enrichment_result.final_output_as(EntityEnrichmentOutput)
        
        # Add self company entity
        enrichment_data.entities.append(
            RelatedEntity(
                entity_name=company,
                relationship_strength=1.0,
                relationship_type="self"
            )
        )
        
        # Enforce max limit
        if len(enrichment_data.entities) > max:
            self_entity = [e for e in enrichment_data.entities if e.relationship_type == "self"]
            other_entities = [e for e in enrichment_data.entities if e.relationship_type != "self"]
            other_entities.sort(key=lambda x: x.relationship_strength, reverse=True)
            enrichment_data.entities = other_entities[:max-1] + self_entity
        
        # Convert to response format
        entities = [
            {
                "entity_name": entity.entity_name,
                "relationship_strength": entity.relationship_strength,
                "relationship_type": entity.relationship_type
            }
            for entity in enrichment_data.entities
        ]
        
        logger.info(f"Found {len(entities)} entities for {company}")
        return entities
        
    except Exception as e:
        logger.error(f"Error fetching entities for {company}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news(company: str, entity: str, max: int = 10):
    """
    Get news articles for a specific entity (Step 2 only).
    
    Args:
        company: Company being analyzed
        entity: Entity to fetch news for
        max: Maximum number of news articles to return (default: 10)
        
    Returns:
        Array of news articles
        
    Example:
        GET /api/news?company=Apple&entity=Samsung&max=10
        
    Response:
        [
            {
                "url": "https://...",
                "title": "Samsung announces new chip",
                "source": "Reuters",
                "published_date": "2024-11-08"
            },
            ...
        ]
    """
    if not company:
        raise HTTPException(status_code=400, detail="company parameter is required")
    if not entity:
        raise HTTPException(status_code=400, detail="entity parameter is required")
    
    try:
        logger.info(f"Fetching news for {entity} (company={company}, max={max})")
        
        # Use the tool directly to fetch news
        news_json = get_entity_news(entity, max)
        news_articles = json.loads(news_json)
        
        # Convert to response format (remove description field)
        formatted_articles = [
            {
                "url": article.get("url"),
                "title": article.get("title"),
                "source": article.get("source"),
                "published_date": article.get("published_date")
            }
            for article in news_articles
        ]
        
        logger.info(f"Found {len(formatted_articles)} news articles for {entity}")
        return formatted_articles
        
    except Exception as e:
        logger.error(f"Error fetching news for {entity}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/signals")
async def get_signals(company: str, entity: str, article: str, max: int = 10):
    """
    Get sentiment signals for a specific article (Step 3 only).
    
    Args:
        company: Company being analyzed
        entity: Entity the article is about
        article: URL of the article to analyze
        max: Maximum number of signals to return (default: 10)
        
    Returns:
        Array of sentiment signal tokens
        
    Example:
        GET /api/signals?company=Apple&entity=Samsung&article=https://...&max=10
        
    Response:
        [
            {
                "token_text": "Samsung announces 30% production increase",
                "impact": "positive",
                "direction": "bullish",
                "strength": 0.85
            },
            ...
        ]
    """
    if not company:
        raise HTTPException(status_code=400, detail="company parameter is required")
    if not entity:
        raise HTTPException(status_code=400, detail="entity parameter is required")
    if not article:
        raise HTTPException(status_code=400, detail="article parameter is required")
    
    try:
        logger.info(f"Analyzing sentiment for article from {entity} (company={company})")
        
        # Create input for sentiment analysis
        single_article_input = {
            "company_name": company,
            "entity_name": entity,
            "relationship_strength": 0.8,  # Default value
            "relationship_type": "related",
            "article": {
                "url": article,
                "title": "",
                "source": entity,
                "published_date": time.strftime("%Y-%m-%d")
            }
        }
        
        # Run sentiment analysis agent
        runner = Runner()
        result = await runner.run(
            single_article_sentiment_agent,
            input=json.dumps(single_article_input)
        )
        
        single_result = result.final_output_as(SingleArticleSentimentOutput)
        
        # Limit to max signals
        sentiment_tokens = single_result.article.sentiment_tokens[:max]
        
        # Convert to response format
        signals = [
            {
                "token_text": token.token_text,
                "impact": token.impact,
                "direction": token.direction,
                "strength": token.strength
            }
            for token in sentiment_tokens
        ]
        
        logger.info(f"Generated {len(signals)} sentiment signals for article")
        return signals
        
    except Exception as e:
        logger.error(f"Error analyzing article sentiment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\n{'='*80}")
    print("🚀 SignalCraft AI Trading Signals API")
    print(f"{'='*80}")
    print(f"Server: http://{host}:{port}")
    print(f"\n🔌 Granular REST API Endpoints:")
    print(f"  GET  /api/entities?company=Apple&max=10")
    print(f"  GET  /api/news?company=Apple&entity=Samsung&max=10")
    print(f"  GET  /api/signals?company=Apple&entity=Samsung&article=https://...&max=10")
    print(f"\n🏥 Health:")
    print(f"  GET  /health")
    print(f"\n💡 Ready for frontend integration!")
    print(f"{'='*80}\n")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
