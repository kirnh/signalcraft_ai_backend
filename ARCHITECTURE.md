# System Architecture

## Tool-to-Agent Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TOOLS (tools.py)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. get_entity_news(entity_name: str, num_results: int) -> str      │
│     └─ Fetches recent news articles from GNews API                  │
│                                                                     │
│  2. fetch_article_content(url: str) -> str                          │
│     └─ Scrapes and returns article text for sentiment analysis      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                   BUILT-IN CAPABILITIES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  • Web Browsing (gpt-4o built-in)                                   │
│     └─ Used by Entity Enrichment Agent to find real-time data       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

                              ↓ ↓ ↓

┌─────────────────────────────────────────────────────────────────────┐
│                      AGENTS (main.py)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Agent 1: Entity Enrichment Agent                                   │
│  ┌────────────────────────────────────┐                             │
│  │ Instructions: prompts.py           │                             │
│  │ Tools: [] (uses web browsing)      │                             │
│  │ Model: gpt-4o                      │                             │
│  └────────────────────────────────────┘                             │
│                                                                     │
│  Agent 2: News Aggregation Agent                                    │
│  ┌────────────────────────────────────┐                             │
│  │ Instructions: prompts.py           │                             │
│  │ Tools: [get_entity_news]           │                             │
│  │ Model: gpt-4o                      │                             │
│  └────────────────────────────────────┘                             │
│                                                                     │
│  Agent 3: Sentiment Analysis Agent                                  │
│  ┌────────────────────────────────────┐                             │
│  │ Instructions: prompts.py           │                             │
│  │ Tools: [fetch_article_content]     │                             │
│  │ Model: gpt-4o                      │                             │
│  └────────────────────────────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Pipeline Flow

```
┌──────────────────────┐
│   User Input:        │
│   Company Name       │
│   (e.g., "Apple")    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 1: Entity Enrichment Agent                                 │
│  ────────────────────────────────────────────────────────────    │
│  Input:  {"company_name": "Apple"}                               │
│                                                                  │
│  Agent uses web browsing to search for:                          │
│  - Apple's main competitors                                      │
│  - Apple's key suppliers                                         │
│  - Apple's executives                                            │
│  - Apple's partners                                              │
│  ↓                                                               │
│  Web browsing returns real-time data:                            │
│  [                                                               │
│    {"entity_name": "TSMC", "strength": 0.95, "type": "supplier"},│
│    {"entity_name": "Samsung", "strength": 0.85, "type": "comp"},│
│    {"entity_name": "Tim Cook", "strength": 1.0, "type": "exec"},│
│  ]                                                               │
│                                                                  │
│  Output: {"company_name": "Apple", "entities": [...]}            │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 2: News Aggregation Agent                                  │
│  ────────────────────────────────────────────────────────────    │
│  Input:  {"company_name": "Apple", "entities": [...]}            │
│                                                                  │
│  Agent iterates over entities and calls:                         │
│    get_entity_news("TSMC")                                       │
│    get_entity_news("Samsung")                                    │
│    get_entity_news("Tim Cook")                                   │
│  ↓                                                               │
│  Tool returns for each:                                          │
│  [                                                               │
│    {"url": "...", "published_date": "...", "source": "..."},    │
│    ...                                                           │
│  ]                                                               │
│                                                                  │
│  Output: {                                                       │
│    "company_name": "Apple",                                      │
│    "entities": [                                                 │
│      {                                                           │
│        "entity_name": "TSMC",                                    │
│        "strength": 0.95,                                         │
│        "type": "supplier",                                       │
│        "news": [{"url": "...", ...}, ...]                        │
│      },                                                          │
│      ...                                                         │
│    ]                                                             │
│  }                                                               │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│  STEP 3: Sentiment Analysis Agent                                │
│  ────────────────────────────────────────────────────────────    │
│  Input:  {"company_name": "Apple", "entities": [{...news...}]}   │
│                                                                  │
│  Agent iterates over news articles and calls:                    │
│    fetch_article_content("https://example.com/article1")         │
│    fetch_article_content("https://example.com/article2")         │
│  ↓                                                               │
│  Tool returns article text content                               │
│  Agent analyzes sentiment and generates tokens                   │
│                                                                  │
│  Output: {                                                       │
│    "company_name": "Apple",                                      │
│    "entities": [                                                 │
│      {                                                           │
│        "entity_name": "TSMC",                                    │
│        "strength": 0.95,                                         │
│        "type": "supplier",                                       │
│        "news": [...],                                            │
│        "sentiment_tokens": [                                     │
│          {                                                       │
│            "tokenText": "TSMC expands capacity",                 │
│            "impact": "positive",                                 │
│            "direction": "bullish",                               │
│            "strength": 0.75                                      │
│          }                                                       │
│        ]                                                         │
│      },                                                          │
│      ...                                                         │
│    ]                                                             │
│  }                                                               │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│   Final Output:      │
│   Trading Signals    │
│   with Sentiment     │
└──────────────────────┘
```
