# AML Investigation Platform

Master's thesis prototype for detecting fraudulent transactions in blockchain using Graph Neural Networks.

## Architecture

- **Frontend**: React + Vite + Sigma.js (WebGL graph rendering)
- **Backend**: FastAPI + igraph + PostgreSQL
- **Database**: PostgreSQL 16 with JSONB for investigation snapshots

## Features

- Interactive graph exploration with k-hop subgraph extraction
- Real-time threshold adjustment with visual feedback
- Node inspector with risk scores and SHAP feature contributions
- Save/load investigation snapshots with exact visual reproducibility
- WebGL-powered rendering for smooth 60 FPS on large graphs

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB RAM recommended

### Run the application

```bash
# Start all services
docker-compose up --build

# Frontend: http://localhost:5173
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Development (without Docker)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_mock_data.py
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI app
│   ├── graph_service.py     # igraph-based k-hop extraction
│   ├── database.py          # SQLAlchemy models
│   ├── schemas.py           # Pydantic schemas
│   ├── config.py            # Settings
│   ├── generate_mock_data.py # Mock Elliptic dataset generator
│   └── data/                # Generated parquet files
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app component
│   │   ├── components/      # React components
│   │   ├── api/             # API client
│   │   └── styles/          # Tailwind CSS
│   └── package.json
└── docker-compose.yml
```

## Usage

1. **Load a subgraph**: Enter a node ID (e.g., `tx_000000`) and click "Load Graph"
2. **Adjust threshold**: Use the slider to filter nodes by risk score
3. **Inspect nodes**: Click any node to see risk score, GNN score, and SHAP features
4. **Save investigation**: Click "Save Investigation" to preserve the current graph state
5. **Load saved work**: Click any item in the sidebar to restore a saved investigation

## Thesis Context

This prototype demonstrates:
- Hybrid-OOF [x || s] cross-fitting to eliminate stacking leakage
- Interactive threshold calibration for temporal drift (post-step 43 analysis)
- SHAP explainability for AML compliance analysts
- Production-grade visual reproducibility for audit trails

## Database Schema

```sql
CREATE TABLE investigations (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    analyst_id VARCHAR(255),
    name VARCHAR(255),
    description TEXT,
    snapshot JSONB,  -- Frozen graph state with coordinates
    metadata JSONB   -- Node count, analyst notes, etc.
);
```

## Performance Targets

- Subgraph extraction: <300ms (5k nodes)
- Graph rendering: 60 FPS (10k edges)
- Investigation load: <100ms (cached)
- Threshold re-coloring: <50ms

## Tech Stack

**Backend:**
- FastAPI 0.115.5
- python-igraph 0.11.8 (C-accelerated layouts)
- SQLAlchemy 2.0.36
- PostgreSQL 16

**Frontend:**
- React 18
- Vite 5
- Sigma.js 3 + Graphology (WebGL)
- Tailwind CSS

## License

MIT - Academic thesis prototype
