import os
import json
import asyncio
import logging
import re
import time
from dotenv import load_dotenv
from agents import Agent, Runner
from tools import get_entity_news, fetch_article_content, reset_tool_call_counter
from prompts import (
    entity_enrichment_agent_config,
    news_aggregation_agent_config,
    sentiment_analysis_agent_config,
    single_entity_news_agent_config,
    single_article_sentiment_agent_config
)
from schemas import (
    EntityEnrichmentOutput, 
    NewsAggregationOutput, 
    SentimentAnalysisOutput, 
    RelatedEntity,
    SingleEntityNewsOutput,
    SingleArticleSentimentOutput,
    EntityWithNews,
    EntityWithSentiment,
    NewsArticle,
    NewsArticleBasic
)

# Load environment variables from .env file
load_dotenv()

# Configure logging with configurable log level
def get_log_level():
    """Get log level from environment variable, defaulting to INFO."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return log_levels.get(log_level_str, logging.INFO)

log_level = get_log_level()
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Log the configured log level
log_level_name = logging.getLevelName(log_level)
logger.info(f"Log level set to: {log_level_name} (from LOG_LEVEL env var: {os.getenv('LOG_LEVEL', 'INFO')})")

# Concurrency limits for parallel processing
MAX_CONCURRENT_NEWS_AGENTS = int(os.getenv("MAX_CONCURRENT_NEWS_AGENTS", "10"))
MAX_CONCURRENT_SENTIMENT_AGENTS = int(os.getenv("MAX_CONCURRENT_SENTIMENT_AGENTS", "5"))  # Reduced default to avoid rate limits
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Maximum retries for rate limit errors

# Configurable limits for entities, news, and sentiment tokens
MAX_ENTITIES = int(os.getenv("MAX_ENTITIES", "10"))
MAX_NEWS_PER_ENTITY = int(os.getenv("MAX_NEWS_PER_ENTITY", "10"))
MIN_SENTIMENT_TOKENS_PER_ARTICLE = int(os.getenv("MIN_SENTIMENT_TOKENS_PER_ARTICLE", "5"))
MAX_SENTIMENT_TOKENS_PER_ARTICLE = int(os.getenv("MAX_SENTIMENT_TOKENS_PER_ARTICLE", "15"))

logger.info(f"Concurrency limits: News agents={MAX_CONCURRENT_NEWS_AGENTS}, Sentiment agents={MAX_CONCURRENT_SENTIMENT_AGENTS}")
logger.info(f"Retry configuration: Max retries={MAX_RETRIES} for rate limit errors")
logger.info(f"Entity/News/Token limits: Max entities={MAX_ENTITIES}, Max news per entity={MAX_NEWS_PER_ENTITY}, Sentiment tokens per article={MIN_SENTIMENT_TOKENS_PER_ARTICLE}-{MAX_SENTIMENT_TOKENS_PER_ARTICLE}")


# Create the three agents with their respective tools and structured outputs
# Note: Entity Enrichment Agent uses web browsing instead of custom tools
entity_enrichment_agent = Agent(
    name="Entity Enrichment Agent",
    instructions=entity_enrichment_agent_config["instructions"],
    tools=[],  # Uses built-in web browsing capability
    model="gpt-4o",
    output_type=entity_enrichment_agent_config["output_type"]
)

news_aggregation_agent = Agent(
    name="News Aggregation Agent", 
    instructions=news_aggregation_agent_config["instructions"],
    tools=[get_entity_news],
    model="gpt-4o",
    output_type=news_aggregation_agent_config["output_type"]
)

sentiment_analysis_agent = Agent(
    name="Sentiment Analysis Agent",
    instructions=sentiment_analysis_agent_config["instructions"],
    tools=[fetch_article_content],
    model="gpt-4o",
    output_type=sentiment_analysis_agent_config["output_type"]
)

# Create agents for parallel processing (one per entity/article)
single_entity_news_agent = Agent(
    name="Single Entity News Agent",
    instructions=single_entity_news_agent_config["instructions"],
    tools=[get_entity_news],
    model="gpt-4o",
    output_type=single_entity_news_agent_config["output_type"]
)

single_article_sentiment_agent = Agent(
    name="Single Article Sentiment Agent",
    instructions=single_article_sentiment_agent_config["instructions"],
    tools=[fetch_article_content],
    model="gpt-4o",
    output_type=single_article_sentiment_agent_config["output_type"]
)


async def run_trading_signal_pipeline(company_name: str):
    """
    Run the complete trading signals pipeline for a company.
    
    Args:
        company_name: Name of the company to analyze
        
    Returns:
        SentimentAnalysisOutput: Validated structured output with all analysis
    """
    # Reset tool call counters at the start
    reset_tool_call_counter()
    
    # Start overall pipeline timer
    pipeline_start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info(f"PIPELINE START: Trading Signal Analysis for {company_name}")
    logger.info("="*60)
    
    print(f"\n{'='*60}")
    print(f"Starting Trading Signal Analysis for: {company_name}")
    print(f"{'='*60}\n")
    
    # Step 1: Entity Enrichment (returns EntityEnrichmentOutput)
    step1_start_time = time.perf_counter()
    logger.info("STEP 1: Entity Enrichment - Starting...")
    logger.debug(f"Input: {json.dumps({'company_name': company_name})}")
    print("Step 1: Entity Enrichment - Finding related entities...")
    
    runner = Runner()
    logger.debug("Runner initialized")
    logger.debug(f"Agent: {entity_enrichment_agent.name}")
    logger.debug(f"Agent tools: {[t.name for t in entity_enrichment_agent.tools]}")
    
    enrichment_result = await runner.run(
        entity_enrichment_agent,
        input=json.dumps({"company_name": company_name})
    )
    logger.debug("Entity enrichment agent completed")
    
    step1_elapsed = time.perf_counter() - step1_start_time
    
    # Get the structured output (automatically parsed and validated!)
    enrichment_data = enrichment_result.final_output_as(EntityEnrichmentOutput)

    # Add self company entity to the output
    try:
        enrichment_data.entities.append(RelatedEntity(entity_name=company_name, relationship_strength=1.0, relationship_type="self"))
    except Exception as e:
        logger.error(f"Failed to add self company entity to output: {e}", exc_info=True)
        print(f"⚠️  Warning: Failed to add self company entity to output: {e}")
        raise
    
    # Enforce MAX_ENTITIES limit
    if len(enrichment_data.entities) > MAX_ENTITIES:
        logger.warning(f"⚠️ Found {len(enrichment_data.entities)} entities, limiting to MAX_ENTITIES={MAX_ENTITIES}")
        print(f"⚠️ Limiting from {len(enrichment_data.entities)} to {MAX_ENTITIES} entities")
        # Keep the highest relationship strength entities plus the self entity
        self_entity = [e for e in enrichment_data.entities if e.relationship_type == "self"]
        other_entities = [e for e in enrichment_data.entities if e.relationship_type != "self"]
        # Sort by relationship strength and take top (MAX_ENTITIES - 1) to make room for self
        other_entities.sort(key=lambda x: x.relationship_strength, reverse=True)
        enrichment_data.entities = other_entities[:MAX_ENTITIES-1] + self_entity
    
    logger.info(f"✓ Found {len(enrichment_data.entities)} related entities")
    
    # Save Step 1 output
    step1_file = f"step1_entity_enrichment_{company_name.lower().replace(' ', '_')}.json"
    with open(step1_file, 'w') as f:
        json.dump(enrichment_data.model_dump(), f, indent=2)
    logger.info(f"✓ Saved Step 1 output to: {step1_file}")
    
    print(f"✓ Found {len(enrichment_data.entities)} related entities")
    for entity in enrichment_data.entities:
        logger.debug(f"Entity: {entity.entity_name} - {entity.relationship_type} (strength: {entity.relationship_strength})")
        print(f"  - {entity.entity_name} ({entity.relationship_type}, strength: {entity.relationship_strength})")
    
    # Log all entities
    logger.info(f"All entities: {[e.entity_name for e in enrichment_data.entities]}")
    
    # Log Step 1 timing
    logger.info(f"⏱️  Step 1 completed in {step1_elapsed:.2f} seconds ({step1_elapsed/60:.2f} minutes)")
    print(f"⏱️  Step 1 completed in {step1_elapsed:.2f} seconds")
    
    # Step 2: News Aggregation (returns NewsAggregationOutput) - PARALLEL PROCESSING
    step2_start_time = time.perf_counter()
    logger.info("-"*60)
    logger.info("STEP 2: News Aggregation - Starting parallel processing...")
    logger.debug(f"Processing {len(enrichment_data.entities)} entities in parallel")
    
    entity_names = [e.entity_name for e in enrichment_data.entities]
    logger.info(f"Entities to fetch news for: {entity_names}")
    logger.info(f"Processing {len(entity_names)} entities with max {MAX_CONCURRENT_NEWS_AGENTS} concurrent agents")
    
    print("\n" + "-"*60)
    print("Step 2: News Aggregation - Fetching news in parallel...")
    print(f"  Entities: {len(entity_names)} total")
    print(f"  Concurrency: {MAX_CONCURRENT_NEWS_AGENTS} agents at a time")
    
    # Create semaphore to limit concurrent agents
    news_semaphore = asyncio.Semaphore(MAX_CONCURRENT_NEWS_AGENTS)
    
    def extract_retry_after_news(error_message: str) -> float:
        """Extract retry-after time from error message."""
        match = re.search(r'(?:try again in|retry_after[:\s]+)(\d+\.?\d*)\s*s', error_message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    
    def is_rate_limit_error_news(error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error)
        return "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower()
    
    async def process_single_entity_news(entity: RelatedEntity, entity_index: int, total_entities: int):
        """Process news for a single entity using a dedicated agent instance with retry logic."""
        async with news_semaphore:
            logger.info(f"Processing entity {entity_index + 1}/{total_entities}: {entity.entity_name}")
            
            # Create input for single entity
            single_entity_input = {
                "company_name": company_name,
                "entity": {
                    "entity_name": entity.entity_name,
                    "relationship_strength": entity.relationship_strength,
                    "relationship_type": entity.relationship_type
                }
            }
            
            # Retry logic with exponential backoff
            last_error = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    runner = Runner()
                    result = await runner.run(
                        single_entity_news_agent,
                        input=json.dumps(single_entity_input)
                    )
                    
                    # Get structured output
                    single_result = result.final_output_as(SingleEntityNewsOutput)
                    logger.info(f"✓ Entity {entity_index + 1}/{total_entities} ({entity.entity_name}): {len(single_result.entity.news)} articles")
                    
                    return single_result.entity, entity_index
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if is_rate_limit_error_news(e) and attempt < MAX_RETRIES:
                        # Extract retry-after time from error message
                        retry_after = extract_retry_after_news(error_str)
                        
                        if retry_after:
                            wait_time = retry_after * 1.1
                            logger.warning(f"Rate limit hit for entity {entity_index + 1} ({entity.entity_name}), waiting {wait_time:.2f}s before retry {attempt + 1}/{MAX_RETRIES}")
                            await asyncio.sleep(wait_time)
                        else:
                            wait_time = (2 ** attempt) + (attempt * 0.5)
                            logger.warning(f"Rate limit hit for entity {entity_index + 1} ({entity.entity_name}), using exponential backoff: {wait_time:.2f}s (retry {attempt + 1}/{MAX_RETRIES})")
                            await asyncio.sleep(wait_time)
                    else:
                        # Not a rate limit error or max retries reached
                        break
            
            # All retries failed or non-rate-limit error
            logger.error(f"Failed to process entity {entity_index + 1} ({entity.entity_name}) after {MAX_RETRIES + 1} attempts: {last_error}", exc_info=True)
            print(f"  ⚠️  ERROR processing {entity.entity_name}: {last_error}")
            
            # Return entity with empty news on error
            error_entity = EntityWithNews(
                entity_name=entity.entity_name,
                relationship_strength=entity.relationship_strength,
                relationship_type=entity.relationship_type,
                news=[]
            )
            return error_entity, entity_index
    
    # Create tasks for all entities
    entity_tasks = [
        process_single_entity_news(entity, idx, len(enrichment_data.entities))
        for idx, entity in enumerate(enrichment_data.entities)
    ]
    
    # Process all entities in parallel with concurrency limit
    logger.info(f"Launching {len(entity_tasks)} parallel news aggregation agents...")
    results = await asyncio.gather(*entity_tasks, return_exceptions=True)
    
    # Aggregate results, maintaining order
    entity_results = []
    successful_count = 0
    failed_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Entity task failed with exception: {result}", exc_info=True)
            failed_count += 1
            continue
        
        entity_data, entity_index = result
        if entity_data:
            entity_results.append((entity_data, entity_index))
            successful_count += 1
    
    # Sort by original index to maintain order
    entity_results.sort(key=lambda x: x[1])
    ordered_entities = [entity for entity, _ in entity_results]
    
    # Enforce MAX_NEWS_PER_ENTITY limit on each entity
    for entity in ordered_entities:
        if len(entity.news) > MAX_NEWS_PER_ENTITY:
            logger.warning(f"⚠️ Entity '{entity.entity_name}' has {len(entity.news)} articles, limiting to MAX_NEWS_PER_ENTITY={MAX_NEWS_PER_ENTITY}")
            entity.news = entity.news[:MAX_NEWS_PER_ENTITY]
    
    # Create aggregated output
    news_data = NewsAggregationOutput(
        company_name=company_name,
        entities=ordered_entities
    )
    
    total_articles = sum(len(entity.news) for entity in news_data.entities)
    logger.info(f"✓ Completed parallel news aggregation: {successful_count} successful, {failed_count} failed")
    logger.info(f"✓ Aggregated {total_articles} news articles across {len(news_data.entities)} entities")
    
    print(f"  ✓ Processed {successful_count}/{len(enrichment_data.entities)} entities")
    if failed_count > 0:
        print(f"  ⚠️  {failed_count} entities failed")
    
    # Save Step 2 output
    step2_file = f"step2_news_aggregation_{company_name.lower().replace(' ', '_')}.json"
    try:
        with open(step2_file, 'w') as f:
            json.dump(news_data.model_dump(), f, indent=2)
        logger.info(f"✓ Saved Step 2 output to: {step2_file}")
        print(f"✓ Saved Step 2 output to: {step2_file}")
    except Exception as e:
        logger.error(f"Failed to save Step 2 output: {e}", exc_info=True)
        print(f"⚠️  Warning: Failed to save Step 2 output: {e}")
    
    for entity in news_data.entities:
        logger.debug(f"Entity '{entity.entity_name}': {len(entity.news)} articles found")
    
    # Check if all entities from step 1 are in step 2
    step1_entities = set(e.entity_name for e in enrichment_data.entities)
    step2_entities = set(e.entity_name for e in news_data.entities)
    missing_entities = step1_entities - step2_entities
    if missing_entities:
        logger.warning(f"⚠️ Missing entities in Step 2: {missing_entities}")
        print(f"  ⚠️  Missing entities: {', '.join(missing_entities)}")
    
    # Check for entities with empty news
    entities_with_no_news = [e.entity_name for e in news_data.entities if len(e.news) == 0]
    if entities_with_no_news:
        logger.warning(f"⚠️ {len(entities_with_no_news)} entities have NO news articles: {entities_with_no_news}")
        print(f"  ⚠️  {len(entities_with_no_news)} entities have no news")
    
    print(f"✓ Aggregated {total_articles} news articles across {len(news_data.entities)} entities")
    
    # Log Step 2 timing
    step2_elapsed = time.perf_counter() - step2_start_time
    logger.info(f"⏱️  Step 2 completed in {step2_elapsed:.2f} seconds ({step2_elapsed/60:.2f} minutes)")
    print(f"⏱️  Step 2 completed in {step2_elapsed:.2f} seconds")
    
    # Step 3: Sentiment Analysis (returns SentimentAnalysisOutput) - PARALLEL PROCESSING
    step3_start_time = time.perf_counter()
    logger.info("-"*60)
    logger.info("STEP 3: Sentiment Analysis - Starting parallel processing...")
    logger.debug(f"Analyzing {total_articles} articles across {len(news_data.entities)} entities")
    
    print("\n" + "-"*60)
    print("Step 3: Sentiment Analysis - Analyzing sentiment signals in parallel...")
    print(f"  Articles: {total_articles} total")
    print(f"  Concurrency: {MAX_CONCURRENT_SENTIMENT_AGENTS} agents at a time")
    
    # Create semaphore to limit concurrent agents
    sentiment_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SENTIMENT_AGENTS)
    
    # Create a flat list of all articles with their entity info
    article_tasks_data = []
    for entity in news_data.entities:
        for article in entity.news:
            article_tasks_data.append({
                "entity_name": entity.entity_name,
                "relationship_strength": entity.relationship_strength,
                "relationship_type": entity.relationship_type,
                "article": article
            })
    
    total_articles_count = len(article_tasks_data)
    logger.info(f"Processing {total_articles_count} articles with max {MAX_CONCURRENT_SENTIMENT_AGENTS} concurrent agents")
    
    def extract_retry_after(error_message: str) -> float:
        """Extract retry-after time from error message."""
        # Look for patterns like "Please try again in 4.03s" or "retry_after: 4.03"
        match = re.search(r'(?:try again in|retry_after[:\s]+)(\d+\.?\d*)\s*s', error_message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    
    def is_rate_limit_error(error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error)
        return "429" in error_str or "rate_limit" in error_str.lower() or "rate limit" in error_str.lower()
    
    async def process_single_article_sentiment(article_data: dict, article_index: int, total_articles: int):
        """Process sentiment for a single article using a dedicated agent instance with retry logic."""
        async with sentiment_semaphore:
            entity_name = article_data["entity_name"]
            article = article_data["article"]
            logger.info(f"Processing article {article_index + 1}/{total_articles}: {entity_name} - {article.title[:50]}...")
            
            # Create input for single article
            single_article_input = {
                "company_name": company_name,
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
            
            # Retry logic with exponential backoff
            last_error = None
            for attempt in range(MAX_RETRIES + 1):
                try:
                    runner = Runner()
                    result = await runner.run(
                        single_article_sentiment_agent,
                        input=json.dumps(single_article_input)
                    )
                    
                    # Get structured output
                    single_result = result.final_output_as(SingleArticleSentimentOutput)
                    token_count = len(single_result.article.sentiment_tokens)
                    logger.info(f"✓ Article {article_index + 1}/{total_articles} ({entity_name}): {token_count} tokens")
                    
                    return single_result, article_index, entity_name
                    
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    
                    # Check if it's a rate limit error
                    if is_rate_limit_error(e) and attempt < MAX_RETRIES:
                        # Extract retry-after time from error message
                        retry_after = extract_retry_after(error_str)
                        
                        if retry_after:
                            # Add a small buffer (10%) to the retry time
                            wait_time = retry_after * 1.1
                            logger.warning(f"Rate limit hit for article {article_index + 1}, waiting {wait_time:.2f}s before retry {attempt + 1}/{MAX_RETRIES}")
                            await asyncio.sleep(wait_time)
                        else:
                            # Exponential backoff: 2^attempt seconds
                            wait_time = (2 ** attempt) + (attempt * 0.5)
                            logger.warning(f"Rate limit hit for article {article_index + 1}, using exponential backoff: {wait_time:.2f}s (retry {attempt + 1}/{MAX_RETRIES})")
                            await asyncio.sleep(wait_time)
                    else:
                        # Not a rate limit error or max retries reached
                        break
            
            # All retries failed or non-rate-limit error
            logger.error(f"Failed to process article {article_index + 1} ({entity_name}) after {MAX_RETRIES + 1} attempts: {last_error}", exc_info=True)
            print(f"  ⚠️  ERROR processing article {article_index + 1}: {last_error}")
            
            # Return article with empty sentiment tokens on error
            error_article = NewsArticle(
                url=article.url,
                title=article.title,
                source=article.source,
                published_date=article.published_date,
                sentiment_tokens=[]
            )
            error_result = SingleArticleSentimentOutput(
                company_name=company_name,
                entity_name=entity_name,
                relationship_strength=article_data["relationship_strength"],
                relationship_type=article_data["relationship_type"],
                article=error_article
            )
            return error_result, article_index, entity_name
    
    # Create tasks for all articles
    article_tasks = [
        process_single_article_sentiment(data, idx, total_articles_count)
        for idx, data in enumerate(article_tasks_data)
    ]
    
    # Process all articles in parallel with concurrency limit
    logger.info(f"Launching {len(article_tasks)} parallel sentiment analysis agents...")
    results = await asyncio.gather(*article_tasks, return_exceptions=True)
    
    # Aggregate results by entity, maintaining order
    entity_articles_dict = {}  # entity_name -> list of articles with sentiment
    successful_count = 0
    failed_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Article task failed with exception: {result}", exc_info=True)
            failed_count += 1
            continue
        
        article_result, article_index, entity_name = result
        if article_result:
            if entity_name not in entity_articles_dict:
                entity_articles_dict[entity_name] = {
                    "entity_name": entity_name,
                    "relationship_strength": article_result.relationship_strength,
                    "relationship_type": article_result.relationship_type,
                    "articles": []
                }
            entity_articles_dict[entity_name]["articles"].append(article_result.article)
            successful_count += 1
    
    # Build ordered entities list matching input order
    ordered_entities = []
    for input_entity in news_data.entities:
        entity_name = input_entity.entity_name
        if entity_name in entity_articles_dict:
            entity_data = entity_articles_dict[entity_name]
            ordered_entities.append({
                "entity_name": entity_name,
                "relationship_strength": entity_data["relationship_strength"],
                "relationship_type": entity_data["relationship_type"],
                "articles": entity_data["articles"]
            })
        else:
            # Entity with no articles processed (shouldn't happen, but handle gracefully)
            logger.warning(f"Entity {entity_name} has no processed articles")
            ordered_entities.append({
                "entity_name": entity_name,
                "relationship_strength": input_entity.relationship_strength,
                "relationship_type": input_entity.relationship_type,
                "articles": []
            })
    
    # Convert to proper schema objects
    sentiment_entities = []
    for entity_dict in ordered_entities:
        sentiment_entities.append(EntityWithSentiment(
            entity_name=entity_dict["entity_name"],
            relationship_strength=entity_dict["relationship_strength"],
            relationship_type=entity_dict["relationship_type"],
            news=entity_dict["articles"]
        ))
    
    sentiment_data = SentimentAnalysisOutput(
        company_name=company_name,
        entities=sentiment_entities
    )
    
    logger.info(f"✓ Completed parallel sentiment analysis: {successful_count} successful, {failed_count} failed")
    print(f"  ✓ Processed {successful_count}/{total_articles_count} articles")
    if failed_count > 0:
        print(f"  ⚠️  {failed_count} articles failed")
    
    # Validate that all entities and articles were processed
    input_entity_count = len(news_data.entities)
    output_entity_count = len(sentiment_data.entities)
    
    print(f"\n📊 Step 3 Validation:")
    print(f"  Input entities: {input_entity_count}")
    print(f"  Output entities: {output_entity_count}")
    
    if input_entity_count != output_entity_count:
        missing_entities = set(e.entity_name for e in news_data.entities) - set(e.entity_name for e in sentiment_data.entities)
        logger.error(f"🚨 CRITICAL: Step 3 processed {output_entity_count} entities, but input had {input_entity_count} entities!")
        logger.error(f"Missing entities: {missing_entities}")
        print(f"  ⚠️  WARNING: Only processed {output_entity_count}/{input_entity_count} entities!")
        print(f"  Missing: {', '.join(missing_entities)}")
    else:
        print(f"  ✓ All {input_entity_count} entities processed")
    
    # Check article counts per entity
    total_input_articles = 0
    total_output_articles = 0
    missing_articles = []
    
    for input_entity in news_data.entities:
        output_entity = next((e for e in sentiment_data.entities if e.entity_name == input_entity.entity_name), None)
        input_article_count = len(input_entity.news)
        total_input_articles += input_article_count
        
        if output_entity:
            output_article_count = len(output_entity.news)
            total_output_articles += output_article_count
            if input_article_count != output_article_count:
                missing_count = input_article_count - output_article_count
                logger.warning(f"⚠️ Entity '{input_entity.entity_name}': Processed {output_article_count}/{input_article_count} articles")
                missing_articles.append(f"{input_entity.entity_name}: {missing_count} missing")
        else:
            logger.error(f"🚨 Entity '{input_entity.entity_name}' is MISSING from Step 3 output!")
            missing_articles.append(f"{input_entity.entity_name}: ALL articles missing")
    
    print(f"  Input articles: {total_input_articles}")
    print(f"  Output articles: {total_output_articles}")
    
    if total_input_articles != total_output_articles:
        print(f"  ⚠️  WARNING: Only processed {total_output_articles}/{total_input_articles} articles!")
        if missing_articles:
            print(f"  Missing articles in: {', '.join(missing_articles[:5])}")
            if len(missing_articles) > 5:
                print(f"  ... and {len(missing_articles) - 5} more entities with missing articles")
    else:
        print(f"  ✓ All {total_input_articles} articles processed")
    
    # Count tokens across all articles in all entities
    total_tokens = sum(
        len(article.sentiment_tokens)
        for entity in sentiment_data.entities
        for article in entity.news
    )
    logger.info(f"✓ Generated {total_tokens} sentiment tokens across all articles")
    
    # Save Step 3 output
    step3_file = f"step3_sentiment_analysis_{company_name.lower().replace(' ', '_')}.json"
    with open(step3_file, 'w') as f:
        json.dump(sentiment_data.model_dump(), f, indent=2)
    logger.info(f"✓ Saved Step 3 output to: {step3_file}")
    
    # Check if all entities from step 2 are in step 3
    step2_entities = set(e.entity_name for e in news_data.entities)
    step3_entities = set(e.entity_name for e in sentiment_data.entities)
    missing_entities_step3 = step2_entities - step3_entities
    if missing_entities_step3:
        logger.warning(f"⚠️ Missing entities in Step 3: {missing_entities_step3}")
    
    # Log token distribution per entity and article
    for entity in sentiment_data.entities:
        entity_token_count = sum(len(article.sentiment_tokens) for article in entity.news)
        logger.debug(f"Entity '{entity.entity_name}': {entity_token_count} sentiment tokens across {len(entity.news)} articles")
        for article in entity.news:
            if article.sentiment_tokens:
                logger.debug(f"  Article '{article.title[:50]}...': {len(article.sentiment_tokens)} tokens")
    
    print(f"✓ Generated {total_tokens} sentiment tokens across all articles")
    
    # Log Step 3 timing
    step3_elapsed = time.perf_counter() - step3_start_time
    logger.info(f"⏱️  Step 3 completed in {step3_elapsed:.2f} seconds ({step3_elapsed/60:.2f} minutes)")
    print(f"⏱️  Step 3 completed in {step3_elapsed:.2f} seconds")
    
    # Show sample sentiment tokens
    logger.debug("Sample sentiment tokens:")
    shown_count = 0
    for entity in sentiment_data.entities:
        if shown_count >= 2:
            break
        for article in entity.news:
            if article.sentiment_tokens and shown_count < 2:
                print(f"\n  {entity.entity_name} - {article.title[:60]}...")
                logger.debug(f"Entity: {entity.entity_name}, Article: {article.title[:50]}... - {len(article.sentiment_tokens)} tokens")
                for token in article.sentiment_tokens[:2]:
                    print(f"    • {token.token_text}")
                    print(f"      Impact: {token.impact}, Direction: {token.direction}, Strength: {token.strength}")
                    logger.debug(f"  Token: {token.token_text} | {token.impact}/{token.direction} ({token.strength})")
                shown_count += 1
                if shown_count >= 2:
                    break
    
    # Calculate total pipeline time
    total_pipeline_time = time.perf_counter() - pipeline_start_time
    
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}\n")
    
    # Pipeline Summary
    logger.info("="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Step 1 - Entity Enrichment: {len(enrichment_data.entities)} entities found")
    logger.info(f"Step 2 - News Aggregation: {len(news_data.entities)} entities with news")
    logger.info(f"Step 3 - Sentiment Analysis: {len(sentiment_data.entities)} entities with sentiment")
    
    if len(enrichment_data.entities) != len(news_data.entities):
        logger.warning(f"⚠️ Entity count mismatch: Step 1 ({len(enrichment_data.entities)}) vs Step 2 ({len(news_data.entities)})")
    if len(news_data.entities) != len(sentiment_data.entities):
        logger.warning(f"⚠️ Entity count mismatch: Step 2 ({len(news_data.entities)}) vs Step 3 ({len(sentiment_data.entities)})")
    
    logger.info(f"Total news articles: {total_articles}")
    logger.info(f"Total sentiment tokens: {total_tokens} (across all articles)")
    
    # Timing Summary
    logger.info("-"*60)
    logger.info("⏱️  TIMING SUMMARY")
    logger.info("-"*60)
    logger.info(f"Step 1 - Entity Enrichment: {step1_elapsed:.2f}s ({step1_elapsed/60:.2f} min)")
    logger.info(f"Step 2 - News Aggregation: {step2_elapsed:.2f}s ({step2_elapsed/60:.2f} min)")
    logger.info(f"Step 3 - Sentiment Analysis: {step3_elapsed:.2f}s ({step3_elapsed/60:.2f} min)")
    logger.info(f"Total Pipeline Time: {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} min)")
    
    # Print timing summary to console
    print(f"\n⏱️  Timing Summary:")
    print(f"  Step 1: {step1_elapsed:.2f}s ({step1_elapsed/total_pipeline_time*100:.1f}%)")
    print(f"  Step 2: {step2_elapsed:.2f}s ({step2_elapsed/total_pipeline_time*100:.1f}%)")
    print(f"  Step 3: {step3_elapsed:.2f}s ({step3_elapsed/total_pipeline_time*100:.1f}%)")
    print(f"  Total:  {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} min)")
    
    # Print intermediate files saved
    logger.info("-"*60)
    logger.info("Intermediate outputs saved:")
    logger.info(f"  - {step1_file}")
    logger.info(f"  - {step2_file}")
    logger.info(f"  - {step3_file}")
    
    logger.info("="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    
    return sentiment_data


async def main():
    """Main entry point for the trading signals application."""
    
    logger.info("="*80)
    logger.info("APPLICATION START: AI Trading Signals - Multi-Agent Pipeline")
    logger.info("="*80)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    logger.info("✓ OPENAI_API_KEY found")
    
    # Check for news API configuration
    news_api_provider = os.getenv("NEWS_API_PROVIDER", "gnews").lower()
    logger.info(f"News API Provider: {news_api_provider}")
    
    if news_api_provider == "gnews":
        if not os.getenv("GNEWS_API_KEY"):
            logger.warning("GNEWS_API_KEY not set, using default key")
        else:
            logger.info("✓ GNEWS_API_KEY found")
    elif news_api_provider == "newsapi":
        if not os.getenv("NEWSAPI_KEY"):
            logger.warning("NEWSAPI_KEY not set, using default key")
        else:
            logger.info("✓ NEWSAPI_KEY found")
    else:
        logger.warning(f"Unknown NEWS_API_PROVIDER '{news_api_provider}', defaulting to 'gnews'")
    
    # Example usage
    companies = ["Apple", "Microsoft", "Tesla"]
    logger.info(f"Available companies: {companies}")
    
    print("AI Trading Signals - Multi-Agent Pipeline")
    print("Agents:")
    print("  1. Entity Enrichment Agent - Finds related entities (uses web browsing)")
    print("  2. News Aggregation Agent - Fetches news articles")
    print("  3. Sentiment Analysis Agent - Analyzes sentiment signals")
    print()
    
    # Run for first company as example
    company = companies[0]
    logger.info(f"Selected company: {company}")
    
    result = await run_trading_signal_pipeline(company)
    
    logger.info(f"Total entities analyzed: {len(result.entities)}")
    
    total_tokens = sum(
        len(article.sentiment_tokens)
        for entity in result.entities
        for article in entity.news
    )
    logger.info(f"Total sentiment signals: {total_tokens} (across all articles)")
    
    print(f"  Total entities analyzed: {len(result.entities)}")
    print(f"  Total sentiment signals: {total_tokens} (across all articles)")
    
    logger.info("="*80)
    logger.info("APPLICATION END")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
