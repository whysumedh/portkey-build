# Agent Model Optimization Platform

A production-grade platform for AI agent developers to analyze performance and receive explainable model recommendations based on real production logs.

## Overview

This platform helps agent developers who:

- Own multiple AI agents
- Iterate on prompts, tools, and models
- Want to optimize cost, quality, refusal rate, and stability
- Need explainable, confidence-backed recommendations

### Core Concept

- A **Project** represents exactly ONE AI agent
- Each project ingests real production logs from that agent
- The platform analyzes logs, replays prompts on selected models, evaluates outputs, and recommends the best models
- All recommendations are **advisory only** - no auto-switching

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│              Dashboard │ Projects │ Analytics │ Recs            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                         API Layer (FastAPI)                     │
│     /projects │ /logs │ /analytics │ /evaluations │ /recs      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                         Core Platform                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │   Log    │ │ Analytics│ │  Model   │ │      Replay      │   │
│  │Ingestion │ │  Engine  │ │ Selector │ │      Engine      │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │Evaluation│ │Aggregation│ │  Drift   │ │  Recommendation  │   │
│  │  Engine  │ │  & State │ │ Detector │ │      Engine      │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────┴───────────────────────────────┐
│                      External Services                          │
│              Portkey API │ LLM Providers │ PostgreSQL           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Log Ingestion

- Sync production logs from Portkey observability
- Immutable, versioned log storage
- Privacy mode (option to hash prompts)

### 2. Analytics Engine

- Safe, structured analysis requests (no arbitrary code execution)
- Supports: distributions, percentiles, correlations, aggregations, clustering, sampling
- Deterministic, cached, budget-limited

### 3. Model Selector

- Deterministic pruning rules first
- AI-assisted ranking
- Outputs structured JSON with explanations
- Never sees raw logs

### 4. Replay Engine

- Replays historical prompts deterministically
- Executes candidate models via Portkey
- Tracks tokens, latency, cost, errors

### 5. Evaluation Engine

- Multiple specialized AI judges:
  - Correctness Judge
  - Safety Judge
  - Quality Judge
  - Helpfulness Judge
- Tracks judge disagreement and variance
- Versioned prompts for reproducibility

### 6. Recommendation Engine

- Confidence-gated recommendations
- "NO RECOMMENDATION" when uncertain
- Full trade-off analysis
- Explainable reasoning

### 7. Drift Detection

- Monitors performance over time windows
- Alerts on degradation
- Actionable recommendations

## Tech Stack

| Component       | Technology                                  |
| --------------- | ------------------------------------------- |
| Backend         | Python 3.11+, FastAPI, SQLAlchemy, Alembic  |
| Analytics       | Pandas, NumPy, SciPy                        |
| Database        | PostgreSQL 15+                              |
| Task Queue      | APScheduler                                 |
| Frontend        | React 18, TypeScript, TailwindCSS, Recharts |
| LLM Integration | Portkey Python SDK                          |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Portkey API key

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings

# Run database migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Access

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/v1/docs
- Frontend: http://localhost:3000

## API Endpoints

### Projects

- `GET /api/v1/projects` - List projects
- `POST /api/v1/projects` - Create project
- `GET /api/v1/projects/{id}` - Get project
- `PATCH /api/v1/projects/{id}` - Update project

### Logs

- `POST /api/v1/logs/{project_id}/sync` - Sync logs from Portkey
- `GET /api/v1/logs/{project_id}/stats` - Get log statistics
- `GET /api/v1/logs/{project_id}` - List logs

### Analytics

- `POST /api/v1/analytics` - Run analysis
- `GET /api/v1/analytics/{project_id}/summary` - Get summary statistics

### Evaluations

- `POST /api/v1/evaluations` - Create evaluation run
- `GET /api/v1/evaluations/{project_id}` - List evaluations
- `GET /api/v1/evaluations/{project_id}/{id}` - Get evaluation details

### Recommendations

- `POST /api/v1/recommendations/{project_id}/generate` - Generate recommendation
- `GET /api/v1/recommendations/{project_id}/latest` - Get latest recommendation
- `POST /api/v1/recommendations/{project_id}/{id}/acknowledge` - Acknowledge recommendation

### Scheduler

- `GET /api/v1/scheduler/status` - Get scheduler status
- `POST /api/v1/scheduler/trigger` - Trigger ad-hoc evaluation

## Safety Guarantees

1. **No Arbitrary Code Execution**: Analytics engine accepts only structured requests
2. **Immutable Logs**: All log entries are append-only with version tracking
3. **Versioned Everything**: Models, prompts, judges, and evaluators are versioned
4. **Advisory Only**: Recommendations never auto-switch models
5. **Confidence Gating**: System outputs "NO RECOMMENDATION" when confidence < threshold
6. **Deterministic Replay**: Fixed seeds, no state mutation by LLMs

## Configuration

Key environment variables:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/agent_optimizer

# Portkey
PORTKEY_API_KEY=your-api-key
PORTKEY_BASE_URL=https://api.portkey.ai

# Evaluation
DEFAULT_REPLAY_SAMPLE_SIZE=100
CONFIDENCE_THRESHOLD=0.7
JUDGE_DISAGREEMENT_THRESHOLD=0.3

# Scheduler
ENABLE_SCHEDULER=true
EVALUATION_SCHEDULE_CRON="0 2 * * 0"
```

## Project Structure

```
portkey-build/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API routes
│   │   ├── core/             # Config, database, logging
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   │   ├── ingestion/    # Log ingestion
│   │   │   ├── analytics/    # Analytics engine
│   │   │   ├── selector/     # Model selection
│   │   │   ├── replay/       # Replay engine
│   │   │   ├── evaluation/   # AI judges
│   │   │   ├── aggregation/  # State & drift
│   │   │   ├── recommendation/ # Recommendations
│   │   │   └── scheduler/    # Task scheduling
│   │   └── main.py           # FastAPI app
│   ├── alembic/              # DB migrations
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/              # API client
│   │   ├── components/       # React components
│   │   ├── pages/            # Page components
│   │   └── types/            # TypeScript types
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## License

MIT

push test3
