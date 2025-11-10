# SignalCraft AI - Multi-Agent trading signals generator

A multi-agent system using OpenAI's Agents SDK to analyze trading signals by enriching company data, aggregating news, and performing sentiment analysis on related entities.

## Architecture

The system uses three specialized agents that work in a pipeline, with **structured outputs** using Pydantic schemas (similar to LangChain's approach) for guaranteed, validated responses:

### 1. Entity Enrichment Agent
- **Purpose**: Discovers entities related to a target company
- **Tools**: Uses built-in web browsing (no custom tools)
- **Output**: List of related entities (competitors, suppliers, executives, partners) with relationship strength and type

### 2. News Aggregation Agent
- **Purpose**: Fetches recent news for each related entity
- **Tools**: `get_entity_news`
- **Output**: News articles with URLs, publication dates, and sources

### 3. Sentiment Analysis Agent
- **Purpose**: Analyzes news sentiment and generates trading signals
- **Tools**: `fetch_article_content`
- **Output**: Sentiment tokens indicating impact, direction, and strength

## Structured Outputs with Pydantic

All agents use **Pydantic schemas** for structured outputs, ensuring type-safe, validated responses. You define output schemas using Pydantic BaseModel classes, and agents return validated data matching these schemas. The system automatically validates all outputs, providing type safety and IDE support.

**Benefits:**
- ✅ Guaranteed format from LLM responses
- ✅ Automatic validation
- ✅ Type safety and IDE support
- ✅ Self-documenting schemas
- ✅ Fewer runtime errors

> 📖 **Important Notes**:
> - The parameter is `output_type`, not `response_format`
> - `Runner.run()` is **async** - always use `await` and `asyncio.run()`
> - Access output with `result.final_output_as(YourSchema)` - automatically validated!
> - Tools must use `@function_tool` decorator from `agents` package


## How Tools Connect to Agents

Agents can use custom tools or built-in capabilities. Custom tools are passed to the Agent constructor, while built-in capabilities like web browsing are available without explicit tool configuration. Each agent specifies its output schema using Pydantic models for structured, validated responses.

### Tool Requirements

Each tool must be a **Python function decorated with `@function_tool`** with:
1. **`@function_tool` decorator** from `agents` package
2. **Type hints** for parameters and return value
3. **Docstring** explaining what it does
4. **String return type** (for OpenAI Agents SDK)

The `@function_tool` decorator:
- Wraps the function as a `FunctionTool` object
- Parses the function signature automatically
- Reads the docstring for tool description
- Makes it recognizable by the SDK
- Enables the LLM to call it properly

## Installation

1. Install dependencies using `uv sync` or `pip install -e .`

2. Create a `.env` file with your configuration. See the Configuration section below for all available options.

**Note:** The `.env` file is automatically loaded when the application starts. You don't need to export environment variables manually.

## Configuration

All configuration is done via environment variables in a `.env` file. Here are the available options:

### Required Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### News API Configuration
- `NEWS_API_PROVIDER`: Set to `gnews` (default) or `newsapi` to choose which provider to use
- `GNEWS_API_KEY`: Your GNews API key (get free key at https://gnews.io)
- `NEWSAPI_KEY`: Your NewsAPI key (get free key at https://newsapi.org)

### Logging Configuration
- `LOG_LEVEL`: Set to `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` (default: `INFO`)

### Concurrency Configuration
- `MAX_CONCURRENT_NEWS_AGENTS`: Maximum number of concurrent agents processing news for entities (default: `10`)
- `MAX_CONCURRENT_SENTIMENT_AGENTS`: Maximum number of concurrent agents processing sentiment analysis (default: `5`)
- `MAX_RETRIES`: Maximum retries for rate limit errors (default: `3`)

### Entity, News, and Token Limits
- `MAX_ENTITIES`: Maximum number of related entities to find per company (default: `10`)
- `MAX_NEWS_PER_ENTITY`: Maximum number of news articles to fetch per entity (default: `10`)
- `MIN_SENTIMENT_TOKENS_PER_ARTICLE`: Minimum number of sentiment tokens to extract per article (default: `5`)
- `MAX_SENTIMENT_TOKENS_PER_ARTICLE`: Maximum number of sentiment tokens to extract per article (default: `15`)

### Example `.env` file

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# News API Configuration
NEWS_API_PROVIDER=gnews
GNEWS_API_KEY=your-gnews-api-key-here

# Logging Configuration
LOG_LEVEL=INFO

# Concurrency Configuration
MAX_CONCURRENT_NEWS_AGENTS=10
MAX_CONCURRENT_SENTIMENT_AGENTS=5
MAX_RETRIES=3

# Entity, News, and Token Limits
MAX_ENTITIES=10
MAX_NEWS_PER_ENTITY=10
MIN_SENTIMENT_TOKENS_PER_ARTICLE=5
MAX_SENTIMENT_TOKENS_PER_ARTICLE=15
```

## Verbose Logging

The system includes **comprehensive verbose logging** to help you understand what's happening at every step. Logging is automatically configured with DEBUG level and shows agent initialization and execution, tool calls and results, API requests and responses, data processing steps, and errors with full stack traces.

**Log levels:**
- `INFO` - Major steps and results
- `DEBUG` - Detailed execution flow, API calls, data processing
- `ERROR` - Errors with stack traces

All logs include **timestamps**, **module names**, and **log levels** for easy debugging. Example log entries show pipeline start, agent execution steps, tool calls, API responses, and results.

## Usage

### 🚀 REST API (For Frontend Integration)

The system includes a **FastAPI-based REST API** with granular endpoints for fetching entities, news, and sentiment analysis. This approach gives your frontend full control over progressive loading and caching.

#### Starting the API Server

```bash
python api.py
```

The API server will start on `http://localhost:8000` (configurable via `HOST` and `PORT` environment variables).

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and API status |
| `/api/entities` | GET | Get related entities for a company |
| `/api/news` | GET | Get news articles for a specific entity |
| `/api/signals` | GET | Get sentiment signals for a specific article |

#### Endpoint Details

**GET /api/entities**

Fetches related entities for a company.

```bash
curl "http://localhost:8000/api/entities?company=Apple&max=10"
```

Response:
```json
[
  {
    "entity_name": "Samsung",
    "relationship_strength": 0.85,
    "relationship_type": "competitor"
  },
  {
    "entity_name": "TSMC",
    "relationship_strength": 0.92,
    "relationship_type": "supplier"
  }
]
```

**GET /api/news**

Fetches news articles for a specific entity.

```bash
curl "http://localhost:8000/api/news?company=Apple&entity=Samsung&max=10"
```

Response:
```json
[
  {
    "url": "https://reuters.com/article/samsung-chip",
    "title": "Samsung announces new 3nm chip production",
    "source": "Reuters",
    "published_date": "2024-11-08"
  }
]
```

**GET /api/signals**

Fetches sentiment signals for a specific article.

```bash
curl "http://localhost:8000/api/signals?company=Apple&entity=Samsung&article=https://...&max=10"
```

Response:
```json
[
  {
    "token_text": "Samsung announces 30% production increase",
    "impact": "positive",
    "direction": "bullish",
    "strength": 0.85
  }
]
```

#### Frontend Integration Example

**TypeScript/JavaScript:**

```typescript
const API_BASE_URL = 'http://localhost:8000';

// Fetch entities
const response = await fetch(
  `${API_BASE_URL}/api/entities?company=${encodeURIComponent('Apple')}&max=10`
);
const entities = await response.json();

// Fetch news for an entity
const newsResponse = await fetch(
  `${API_BASE_URL}/api/news?company=Apple&entity=${encodeURIComponent('Samsung')}&max=10`
);
const news = await newsResponse.json();

// Fetch signals for an article
const signalsResponse = await fetch(
  `${API_BASE_URL}/api/signals?company=Apple&entity=Samsung&article=${encodeURIComponent(articleUrl)}&max=10`
);
const signals = await signalsResponse.json();
```

**Progressive Loading Pattern:**

```typescript
// Load entities first (fast)
const entities = await api.getEntities(company, maxEntities);
setEntities(entities); // User sees entities immediately!

// Then load news progressively
for (const entity of entities) {
  await new Promise(resolve => setTimeout(resolve, 100)); // Smooth animation
  const news = await api.getNews(company, entity.entity_name, maxNews);
  setEntityNews(entity, news); // News appears one entity at a time
}

// Load signals on-demand when user clicks an article
const signals = await api.getSignals(company, entity, article.url, maxSignals);
```

#### CORS Configuration

The API includes CORS middleware configured to accept requests from any origin (development mode). For production, update the `allow_origins` in `api.py` to specify your frontend URL:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 📦 Batch Processing (Python Script)

For batch processing or testing, you can use the Python script directly.

#### Run the Complete Pipeline

Run `python main.py` to execute the complete pipeline. This will:
1. Analyze the target company (default: Apple)
2. Find related entities
3. Fetch news for each entity
4. Perform sentiment analysis
5. Save results to JSON files

#### Basic Usage

Import `run_trading_signal_pipeline` from `main` and call it asynchronously with a company name:

```python
import asyncio
from main import run_trading_signal_pipeline

async def main():
    result = await run_trading_signal_pipeline("Apple")
    print(f"Found {len(result.entities)} entities")
    print(f"Total sentiment signals: {sum(len(a.sentiment_tokens) for e in result.entities for a in e.news)}")

asyncio.run(main())
```

#### Using Individual Agents

You can use individual agents by importing them from `main` and using the `Runner` class:

```python
import asyncio
from agents import Runner
from main import entity_enrichment_agent
from schemas import EntityEnrichmentOutput

async def main():
    runner = Runner()
    result = await runner.run(
        entity_enrichment_agent,
        input='{"company_name": "Tesla"}'
    )
    enrichment_data = result.final_output_as(EntityEnrichmentOutput)
    print(f"Found entities: {[e.entity_name for e in enrichment_data.entities]}")

asyncio.run(main())
```

## Project Structure

The project contains:
- `main.py`: Main pipeline and agent initialization (batch processing)
- `api.py`: **FastAPI REST API server** (for frontend integration)
- `prompts.py`: Agent instructions and configurations
- `tools.py`: Tool functions for agents
- `schemas.py`: Pydantic schemas for structured outputs
- `pyproject.toml`: Dependencies
- `README.md`: This file (full documentation)
- `ARCHITECTURE.md`: System architecture and data flow

## Tool-to-Agent Mapping

| Agent | Tools Used | Purpose |
|-------|-----------|---------|
| Entity Enrichment Agent | Web browsing (built-in) | Find competitors, suppliers, executives |
| News Aggregation Agent | `get_entity_news` | Fetch recent news articles |
| Sentiment Analysis Agent | `fetch_article_content` | Parse articles and extract sentiment |

## Data Flow

The pipeline processes data through three stages:
1. **Input**: Company Name
2. **Entity Enrichment Agent**: Uses web browsing to find real-time data and discover related entities with relationships
3. **News Aggregation Agent**: Uses `get_entity_news` to fetch news articles for each entity
4. **Sentiment Analysis Agent**: Uses `fetch_article_content` to parse articles and extract sentiment tokens
5. **Output**: JSON with actionable trading signals and insights

## Extending the System

### Adding a New Tool

1. **Create the tool function in `tools.py`**: Define a function with the `@function_tool` decorator, include type hints and a docstring, and return a JSON string.

2. **Add it to an agent in `main.py`**: Include the tool in the `tools` parameter when creating the agent.

3. **Update the agent instructions in `prompts.py`** to mention the new tool.

### Creating a New Agent

1. **Add configuration to `prompts.py`**: Define an agent configuration dictionary with instructions and output format.

2. **Create the agent in `main.py`**: Instantiate an Agent with the configuration, tools, and model.

3. **Integrate into the pipeline**: Add the agent to the pipeline workflow.

## API Keys

See the Configuration section above for details on setting up API keys.

## Notes

- The Entity Enrichment Agent uses the model's built-in web browsing to find real-time data about competitors, suppliers, executives, and partners
- Rate limiting and error handling should be enhanced for production use
- Consider adding caching for news and entity lookups
- Sentiment analysis can be improved with fine-tuned models
- For more structured entity data, consider integrating with:
  - Crunchbase API
  - LinkedIn API
  - Financial data providers (Bloomberg, FactSet)
  - SEC EDGAR for filings

## Example Output

The system outputs a JSON structure containing the company name, a list of entities with their relationship information, associated news articles, and sentiment tokens. Each sentiment token includes the token text, impact (positive/negative/neutral), direction (bullish/bearish/neutral), and strength (0.0 to 1.0).

## License

MIT

