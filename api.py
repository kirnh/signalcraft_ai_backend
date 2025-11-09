"""
FastAPI application providing granular REST API endpoints for trading signals analysis.
Provides endpoints for entities, news, and sentiment analysis.
"""

import os
import json
import logging
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from main import (
    entity_enrichment_agent,
    single_article_sentiment_agent,
    reset_tool_call_counter,
    Runner,
    single_entity_news_agent,
    MAX_CONCURRENT_NEWS_AGENTS,
    MAX_CONCURRENT_SENTIMENT_AGENTS,
    MAX_ENTITIES,
    MAX_NEWS_PER_ENTITY,
    MAX_RETRIES,
    validate_company_input
)
from schemas import (
    EntityEnrichmentOutput,
    SingleArticleSentimentOutput,
    RelatedEntity,
    SingleEntityNewsOutput,
    EntityWithNews,
    NewsArticle
)
from tools import _get_entity_news_internal as get_entity_news
import time
import re

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
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": "company parameter is required"
            }
        )
    
    # Validate that input is about a single company or stock
    is_valid, detected_entity, error_message = await validate_company_input(company)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": error_message
            }
        )
    
    # Use detected entity if available, otherwise use original input
    company_to_use = detected_entity if detected_entity else company
    
    try:
        logger.info(f"Fetching entities for {company_to_use} (max={max})")
        
        # Reset tool call counters
        reset_tool_call_counter()
        
        # Run entity enrichment agent
        runner = Runner()
        enrichment_result = await runner.run(
            entity_enrichment_agent,
            input=json.dumps({"company_name": company_to_use})
        )
        
        enrichment_data = enrichment_result.final_output_as(EntityEnrichmentOutput)
        
        # Add self company entity
        enrichment_data.entities.append(
            RelatedEntity(
                entity_name=company_to_use,
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
        
        logger.info(f"Found {len(entities)} entities for {company_to_use}")
        return entities
        
    except HTTPException:
        # Re-raise HTTP exceptions (like validation errors)
        raise
    except Exception as e:
        logger.error(f"Error fetching entities for {company_to_use}: {e}", exc_info=True)
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


# ============================================================================
# STREAMING ENDPOINT WITH LOADING INDICATORS
# ============================================================================

def format_sse(data: dict, event: str = None) -> str:
    """Format data as Server-Sent Event."""
    msg = f"data: {json.dumps(data)}\n\n"
    if event:
        msg = f"event: {event}\n{msg}"
    return msg


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    error_str = str(error)
    return "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower()


def extract_retry_after(error_message: str) -> float:
    """Extract retry-after time from error message."""
    match = re.search(r'(?:try again in|retry_after[:\s]+)(\d+\.?\d*)\s*s', error_message, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


@app.get("/api/stream")
async def stream_analysis(company: str, max_entities: int = 10, max_news: int = 10):
    """
    Stream complete trading signals analysis with loading indicators.
    Returns Server-Sent Events (SSE) with progressive updates.
    
    Args:
        company: Company name to analyze
        max_entities: Maximum number of entities (default: 10)
        max_news: Maximum news articles per entity (default: 10)
        
    Returns:
        SSE stream with events:
        - entity: New entity discovered (with isLoadingNews: true)
        - entity_update: Entity news loaded (with isLoadingNews: false)
        - article_update: Article signals ready (with isLoading: false)
        - complete: Analysis finished
        - error: Error occurred
        
    Example:
        GET /api/stream?company=Apple&max_entities=10&max_news=10
        
    Response (SSE stream):
        data: {"type": "entity", "data": {"entity_name": "Samsung", "isLoadingNews": true, ...}}
        
        data: {"type": "entity_update", "data": {"entity_name": "Samsung", "isLoadingNews": false, ...}}
        
        data: {"type": "article_update", "data": {"entity_name": "Samsung", "article_url": "...", "isLoading": false, ...}}
        
        data: {"type": "complete"}
    """
    if not company:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "validation_failed",
                "message": "company parameter is required"
            }
        )
    
    async def generate_events():
        try:
            # Validate that input is about a single company or stock
            is_valid, detected_entity, error_message = await validate_company_input(company)
            if not is_valid:
                yield format_sse({
                    "type": "error",
                    "error": "validation_failed",
                    "message": error_message
                })
                return
            
            # Use detected entity if available, otherwise use original input
            company_to_use = detected_entity if detected_entity else company
            
            logger.info(f"Starting streaming analysis for {company_to_use}")
            
            # Reset tool call counters
            reset_tool_call_counter()
            
            # Step 1: Entity Enrichment
            yield format_sse({"type": "status", "message": "Finding related entities..."})
            
            runner = Runner()
            enrichment_result = await runner.run(
                entity_enrichment_agent,
                input=json.dumps({"company_name": company_to_use})
            )
            
            enrichment_data = enrichment_result.final_output_as(EntityEnrichmentOutput)
            
            # Add self company entity
            enrichment_data.entities.append(
                RelatedEntity(
                    entity_name=company_to_use,
                    relationship_strength=1.0,
                    relationship_type="self"
                )
            )
            
            # Enforce max limit
            if len(enrichment_data.entities) > max_entities:
                self_entity = [e for e in enrichment_data.entities if e.relationship_type == "self"]
                other_entities = [e for e in enrichment_data.entities if e.relationship_type != "self"]
                other_entities.sort(key=lambda x: x.relationship_strength, reverse=True)
                enrichment_data.entities = other_entities[:max_entities-1] + self_entity
            
            logger.info(f"Found {len(enrichment_data.entities)} entities")
            
            # Emit entities with loading state
            for entity in enrichment_data.entities:
                yield format_sse({
                    "type": "entity",
                    "data": {
                        "entity_name": entity.entity_name,
                        "relationship_strength": entity.relationship_strength,
                        "relationship_type": entity.relationship_type,
                        "isLoadingNews": True,
                        "newsArticles": []
                    }
                })
            
            # Step 2: News Aggregation (parallel)
            yield format_sse({"type": "status", "message": f"Fetching news for {len(enrichment_data.entities)} entities..."})
            
            news_semaphore = asyncio.Semaphore(MAX_CONCURRENT_NEWS_AGENTS)
            
            async def process_entity_news_stream(entity: RelatedEntity):
                """Process news for a single entity and emit updates."""
                async with news_semaphore:
                    single_entity_input = {
                        "company_name": company_to_use,
                        "entity": {
                            "entity_name": entity.entity_name,
                            "relationship_strength": entity.relationship_strength,
                            "relationship_type": entity.relationship_type
                        }
                    }
                    
                    # Retry logic
                    last_error = None
                    for attempt in range(MAX_RETRIES + 1):
                        try:
                            runner = Runner()
                            result = await runner.run(
                                single_entity_news_agent,
                                input=json.dumps(single_entity_input)
                            )
                            
                            single_result = result.final_output_as(SingleEntityNewsOutput)
                            
                            # Limit news articles
                            if len(single_result.entity.news) > max_news:
                                single_result.entity.news = single_result.entity.news[:max_news]
                            
                            # Emit entity update with news (articles have isLoading: true)
                            news_articles = [
                                {
                                    "url": article.url,
                                    "title": article.title,
                                    "source": article.source,
                                    "published_date": article.published_date,
                                    "isLoading": True,
                                    "signals": []
                                }
                                for article in single_result.entity.news
                            ]
                            
                            return {
                                "entity_name": entity.entity_name,
                                "relationship_strength": entity.relationship_strength,
                                "relationship_type": entity.relationship_type,
                                "isLoadingNews": False,
                                "newsArticles": news_articles
                            }, single_result.entity
                            
                        except Exception as e:
                            last_error = e
                            if is_rate_limit_error(e) and attempt < MAX_RETRIES:
                                retry_after = extract_retry_after(str(e))
                                wait_time = retry_after * 1.1 if retry_after else (2 ** attempt) + (attempt * 0.5)
                                logger.warning(f"Rate limit for {entity.entity_name}, waiting {wait_time:.2f}s")
                                await asyncio.sleep(wait_time)
                            else:
                                break
                    
                    # Failed - return empty
                    logger.error(f"Failed to fetch news for {entity.entity_name}: {last_error}")
                    return {
                        "entity_name": entity.entity_name,
                        "relationship_strength": entity.relationship_strength,
                        "relationship_type": entity.relationship_type,
                        "isLoadingNews": False,
                        "newsArticles": []
                    }, None
            
            # Process all entities in parallel
            entity_tasks = [process_entity_news_stream(entity) for entity in enrichment_data.entities]
            entity_results = await asyncio.gather(*entity_tasks, return_exceptions=True)
            
            # Emit entity updates and collect articles for sentiment analysis
            articles_to_process = []
            
            for result in entity_results:
                if isinstance(result, Exception):
                    logger.error(f"Entity task failed: {result}")
                    continue
                
                entity_update, entity_with_news = result
                
                # Emit entity update
                yield format_sse({
                    "type": "entity_update",
                    "data": entity_update
                })
                
                # Collect articles for sentiment processing
                if entity_with_news and entity_with_news.news:
                    for article in entity_with_news.news:
                        articles_to_process.append({
                            "entity_name": entity_with_news.entity_name,
                            "relationship_strength": entity_with_news.relationship_strength,
                            "relationship_type": entity_with_news.relationship_type,
                            "article": article
                        })
            
            # Step 3: Sentiment Analysis (parallel)
            if articles_to_process:
                yield format_sse({"type": "status", "message": f"Analyzing {len(articles_to_process)} articles..."})
                
                sentiment_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SENTIMENT_AGENTS)
                
                async def process_article_sentiment_stream(article_data: dict):
                    """Process sentiment for a single article and emit update."""
                    async with sentiment_semaphore:
                        entity_name = article_data["entity_name"]
                        article = article_data["article"]
                        
                        single_article_input = {
                            "company_name": company_to_use,
                            "entity_name": entity_name,
                            "relationship_strength": article_data["relationship_strength"],
                            "relationship_type": article_data["relationship_type"],
                            "article": {
                                "url": article.url,
                                "title": article.title,
                                "source": article.source,
                                "published_date": article.published_date
                            }
                        }
                        
                        # Retry logic
                        last_error = None
                        for attempt in range(MAX_RETRIES + 1):
                            try:
                                runner = Runner()
                                result = await runner.run(
                                    single_article_sentiment_agent,
                                    input=json.dumps(single_article_input)
                                )
                                
                                single_result = result.final_output_as(SingleArticleSentimentOutput)
                                
                                # Emit article update with signals
                                return {
                                    "entity_name": entity_name,
                                    "article_url": article.url,
                                    "article_title": article.title,
                                    "isLoading": False,
                                    "signals": [
                                        {
                                            "token_text": token.token_text,
                                            "impact": token.impact,
                                            "direction": token.direction,
                                            "strength": token.strength
                                        }
                                        for token in single_result.article.sentiment_tokens
                                    ]
                                }
                                
                            except Exception as e:
                                last_error = e
                                if is_rate_limit_error(e) and attempt < MAX_RETRIES:
                                    retry_after = extract_retry_after(str(e))
                                    wait_time = retry_after * 1.1 if retry_after else (2 ** attempt) + (attempt * 0.5)
                                    logger.warning(f"Rate limit for article, waiting {wait_time:.2f}s")
                                    await asyncio.sleep(wait_time)
                                else:
                                    break
                        
                        # Failed - return empty signals
                        logger.error(f"Failed to analyze article {article.url}: {last_error}")
                        return {
                            "entity_name": entity_name,
                            "article_url": article.url,
                            "article_title": article.title,
                            "isLoading": False,
                            "signals": []
                        }
                
                # Process all articles in parallel
                article_tasks = [process_article_sentiment_stream(data) for data in articles_to_process]
                article_results = await asyncio.gather(*article_tasks, return_exceptions=True)
                
                # Emit article updates
                for result in article_results:
                    if isinstance(result, Exception):
                        logger.error(f"Article task failed: {result}")
                        continue
                    
                    yield format_sse({
                        "type": "article_update",
                        "data": result
                    })
            
            # Complete
            yield format_sse({"type": "complete", "message": "Analysis complete"})
            logger.info(f"Completed streaming analysis for {company_to_use}")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield format_sse({
                "type": "error",
                "message": str(e)
            })
    
    return StreamingResponse(generate_events(), media_type="text/event-stream")


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
    print(f"\n📡 Streaming Endpoint (with loading indicators):")
    print(f"  GET  /api/stream?company=Apple&max_entities=10&max_news=10")
    print(f"       Returns SSE stream with progressive updates")
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
