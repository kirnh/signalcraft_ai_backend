import requests
from bs4 import BeautifulSoup
from agents import function_tool
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logger for tools
logger = logging.getLogger(__name__)

# Tool call counter for debugging
_tool_call_counter = {"get_entity_news": 0, "fetch_article_content": 0}

# News API configuration
NEWS_API_PROVIDER = os.getenv("NEWS_API_PROVIDER", "gnews").lower()  # 'gnews' or 'newsapi'
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "db2651be6ce35c4956fbe1fc2a5a8cdb")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "48c51ab301094753bb46f899b6b5a103")

# Configurable limit for news articles per entity
MAX_NEWS_PER_ENTITY = int(os.getenv("MAX_NEWS_PER_ENTITY", "10"))



def _fetch_news_gnews(entity_name: str, num_results: int = 10) -> list:
    """
    Fetch news articles using GNews API.
    
    Args:
        entity_name: Name of the entity to fetch news for
        num_results: Number of articles to fetch
        
    Returns:
        List of formatted article dictionaries
    """
    url = "https://gnews.io/api/v4/search"
    
    params = {
        'q': entity_name,
        'lang': 'en',
        'max': num_results,
        'apikey': GNEWS_API_KEY
    }
    
    logger.debug(f"Calling GNews API: {url}")
    logger.debug(f"Query params: q={entity_name}, lang=en, max={num_results}")
    
    response = requests.get(url, params=params, timeout=10)
    logger.debug(f"API response status: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    
    articles = data.get('articles', [])
    logger.info(f"✓ GNews: Fetched {len(articles)} articles for '{entity_name}'")
    
    # Format to match expected output
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            'url': article.get('url'),
            'published_date': article.get('publishedAt'),
            'source': article.get('source', {}).get('name'),
            'title': article.get('title'),
            'description': article.get('description')
        })
        logger.debug(f"  Article: {article.get('title')} from {article.get('source', {}).get('name')}")
    
    return formatted_articles


def _fetch_news_newsapi(entity_name: str, num_results: int = 10) -> list:
    """
    Fetch news articles using NewsAPI.
    
    Args:
        entity_name: Name of the entity to fetch news for
        num_results: Number of articles to fetch
        
    Returns:
        List of formatted article dictionaries
    """
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': entity_name,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_results,
        'apiKey': NEWSAPI_KEY
    }
    
    logger.debug(f"Calling NewsAPI: {url}")
    logger.debug(f"Query params: q={entity_name}, sortBy=publishedAt, pageSize={num_results}")
    
    response = requests.get(url, params=params, timeout=10)
    logger.debug(f"API response status: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    
    if data.get('status') != 'ok':
        error_msg = data.get('message', 'Unknown error')
        logger.error(f"NewsAPI Error: {error_msg}")
        raise Exception(f"NewsAPI Error: {error_msg}")
    
    articles = data.get('articles', [])
    logger.info(f"✓ NewsAPI: Fetched {len(articles)} articles for '{entity_name}'")
    
    # Format to match expected output (convert NewsAPI format to standard format)
    formatted_articles = []
    for article in articles:
        formatted_articles.append({
            'url': article.get('url'),
            'published_date': article.get('publishedAt'),
            'source': article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source'),
            'title': article.get('title'),
            'description': article.get('description')
        })
        logger.debug(f"  Article: {article.get('title')} from {article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else article.get('source')}")
    
    return formatted_articles


def _get_entity_news_internal(entity_name: str, num_results: int = None) -> str:
    """
    Internal function to fetch news articles for an entity.
    This is the actual implementation that can be called directly.
    
    Args:
        entity_name: Name of the entity to fetch news for
        num_results: Number of articles to fetch (default: uses MAX_NEWS_PER_ENTITY env var, defaults to 10)
        
    Returns:
        JSON string containing news articles with url, published_date, source, title, description
    """
    # Use configured default if not specified
    if num_results is None:
        num_results = MAX_NEWS_PER_ENTITY
    
    _tool_call_counter["get_entity_news"] += 1
    call_count = _tool_call_counter["get_entity_news"]
    logger.info(f"🔧 TOOL CALL #{call_count}: get_entity_news(entity_name='{entity_name}', num_results={num_results})")
    logger.info(f"Using news API provider: {NEWS_API_PROVIDER} (MAX_NEWS_PER_ENTITY={MAX_NEWS_PER_ENTITY})")
    print(f"  → Tool call #{call_count}: Fetching news for '{entity_name}' using {NEWS_API_PROVIDER}")
    
    try:
        # Route to appropriate API provider
        if NEWS_API_PROVIDER == "newsapi":
            formatted_articles = _fetch_news_newsapi(entity_name, num_results)
        elif NEWS_API_PROVIDER == "gnews":
            formatted_articles = _fetch_news_gnews(entity_name, num_results)
        else:
            logger.warning(f"Unknown news API provider '{NEWS_API_PROVIDER}', defaulting to 'gnews'")
            formatted_articles = _fetch_news_gnews(entity_name, num_results)
        
        logger.debug(f"Returning {len(formatted_articles)} formatted articles")
        return json.dumps(formatted_articles, indent=2)
            
    except Exception as e:
        logger.error(f"Error fetching news for '{entity_name}' using {NEWS_API_PROVIDER}: {e}", exc_info=True)
        print(f"Error fetching news: {e}")
        return json.dumps([])


@function_tool
def get_entity_news(entity_name: str, num_results: int = None) -> str:
    """
    Fetch news articles for an entity using a configurable news API (GNews or NewsAPI).
    Configure via NEWS_API_PROVIDER environment variable ('gnews' or 'newsapi').
    
    Args:
        entity_name: Name of the entity to fetch news for
        num_results: Number of articles to fetch (default: uses MAX_NEWS_PER_ENTITY env var, defaults to 10)
        
    Returns:
        JSON string containing news articles with url, published_date, source, title, description
    """
    return _get_entity_news_internal(entity_name, num_results)


@function_tool
def fetch_article_content(url: str) -> str:
    """
    Fetch a URL and return parsed article text content for sentiment analysis.
    
    Args:
        url: The URL of the article to fetch
        
    Returns:
        String containing the article title and main text content
    """
    _tool_call_counter["fetch_article_content"] += 1
    call_count = _tool_call_counter["fetch_article_content"]
    logger.info(f"🔧 TOOL CALL #{call_count}: fetch_article_content(url='{url[:50]}...')")
    print(f"  → Tool call #{call_count}: Fetching article content")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        logger.debug(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        logger.debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        logger.debug("Parsing HTML content with BeautifulSoup")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract title
        title = soup.find('title')
        title_text = title.string if title else "No title found"
        logger.debug(f"Article title: {title_text}")
        
        # Extract main content (article text)
        # Try common article containers
        article = soup.find('article') or soup.find('div', class_='article-content') or soup.find('div', class_='content')
        
        if article:
            text = article.get_text(separator='\n', strip=True)
            logger.debug("Extracted text from article container")
        else:
            text = soup.get_text(separator='\n', strip=True)
            logger.debug("Extracted text from full page")
        
        # Clean up whitespace
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        
        original_length = len(text)
        # Limit text length to avoid token limits (first 2000 chars)
        if len(text) > 2000:
            text = text[:2000] + "..."
            logger.debug(f"Truncated text from {original_length} to 2000 characters")
        
        logger.info(f"✓ Successfully fetched and parsed article ({len(text)} chars)")
        return f"Title: {title_text}\n\nContent:\n{text}"
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching article from {url}: {e}", exc_info=True)
        return f"Error fetching article: {str(e)}"


def get_tool_call_count(tool_name: str = None) -> dict:
    """Get the tool call counter for debugging."""
    if tool_name:
        return _tool_call_counter.get(tool_name, 0)
    return _tool_call_counter.copy()


def reset_tool_call_counter():
    """Reset all tool call counters."""
    global _tool_call_counter
    _tool_call_counter = {"get_entity_news": 0, "fetch_article_content": 0}
    logger.info("Tool call counters reset")
