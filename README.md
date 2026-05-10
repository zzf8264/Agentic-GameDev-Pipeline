# Agentic GameDev Pipeline

**Multi-agent game development workflow** demonstrating high-frequency LLM usage,
long-context ingestion, and autonomous self-correction. Designed as a reference
architecture for **API Token quota applications**.

## Architecture

```
                        ┌─────────────────┐
                        │    Scout Agent   │  Playwright headless browser
                        │  (trend intel)   │  Douyin / TikTok scraping
                        └────────┬────────┘
                                 │ XML-tagged report (1k–4k tokens)
                                 ▼
                        ┌─────────────────┐
                        │    Dev Agent     │  LLM-generated TypeScript
                        │  (code gen)      │  Cocos Creator 3.x match-3
                        └────────┬────────┘
                                 │ generated .ts source + LLM self-review
                                 ▼
                        ┌─────────────────┐
                        │    QA Agent      │  Static analysis + EasyClick
                        │  (automated QA)  │  crash-log synthesis
                        └────────┬────────┘
                           ┌─────┴─────┐
                           ▼           ▼
                       ┌──────┐   ┌──────────┐
                       │ PASS │   │  FAIL    │── crash log (~400 tokens) ──► Dev (refactor)
                       └──────┘   └──────────┘
                                      │ (up to MAX_SELF_HEAL_ITERATIONS)
                                      ▼
                              ┌────────────────┐
                              │  Token Report   │  Real token counts from
                              │  (for quota)    │  LLM API responses
                              └────────────────┘
```

## Why This Matters

| Criterion | Demonstration |
|-----------|--------------|
| **Real LLM calls** | Every agent uses `openai.AsyncOpenAI` against any OpenAI-compatible endpoint (Ollama, OpenAI, vLLM, etc.). Token counts come from the actual API response. |
| **Long-context ingestion** | Scout report (1k–4k tokens) + crash logs (~400 tokens each) + review feedback concatenated into single prompts. Self-heal loop accumulates multi-turn context. |
| **Multi-agent orchestration** | Three specialised agents with shared `LLMClient`, checkpoint persistence, and structured data contracts between phases. |
| **Self-correction loop** | QA defect detection is **deterministic** (static analysis of actual source code). Crash logs are synthesised from real defect signatures and fed back to Dev for refactoring. |
| **Production patterns** | asyncio concurrency, tenacity retry, token tracking, `DRY_RUN` mode for offline testing, checkpointing, structured logging. |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run with default fallback (no LLM needed)
python orchestrator.py

# Run with local LLM (Ollama)
LLM_ENDPOINT=http://localhost:11434/v1 python orchestrator.py

# Dry-run mode (skip all real LLM calls, use fallbacks)
DRY_RUN=true python orchestrator.py
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENDPOINT` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen2.5:7b` | Model name |
| `LLM_MAX_TOKENS` | `8192` | Max output tokens per LLM call |
| `DRY_RUN` | `false` | Skip LLM calls (use fallback templates) |
| `MAX_SELF_HEAL_ITERATIONS` | `3` | Max QA → Dev refactor cycles |
| `SCOUT_PLATFORM` | `douyin` | Target platform for trend scraping |
| `SCOUT_HASHTAGS` | (comma-separated) | Hashtags to scrape |
| `GAME_MECHANIC` | `match-3` | Game mechanic for code generation |
| `LOG_LEVEL` | `INFO` | Python log level |

## Project Structure

```
├── agents/
│   ├── __init__.py        # Agent & data-contract exports
│   ├── scout_agent.py      # Playwright-based trend scraper + cache
│   ├── dev_agent.py        # LLM code generator + fallback templates
│   └── qa_agent.py         # Static analysis + deterministic crash-log synthesis
├── core/
│   └── llm.py              # Shared OpenAI-compatible wrapper with token tracking
├── prompts/
│   ├── system_prompts.yaml # Agent identity & constraint definitions
│   └── review_templates.md # Code-review format templates
├── docker/
│   └── Dockerfile          # Multi-stage Python + Playwright image
├── docker-compose.yml      # Pipeline orchestration
├── orchestrator.py         # Main self-healing loop
├── requirements.txt        # Dependencies
└── README.md
```

## Token Consumption Report

Every pipeline run ends with a structured log:

```
  Total LLM calls:          7
  Prompt tokens:            12483
  Completion tokens:        3821
  Total tokens consumed:    16304
  Largest context window:   6240 tokens
  Self-heal iterations:     2
  Crash log tokens:         4210 tokens
```

## Output

Checkpoints are written to `data/checkpoints/`:
- `01_scout_report.json` — Raw trend data
- `02_gen_Match3Board.txt` — Generated TypeScript source
- `03_review.txt` — LLM review output
- `04_qa_iter*.json` — Per-iteration QA results
- `05_refactored_iter*` — Refactored code after each heal cycle
