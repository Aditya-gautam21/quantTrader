# QuantTrader Real-Time Trading Platform

This plan outlines the architecture for adding a real-time trading dashboard using the existing Machine Learning models and data collectors, strictly without altering any existing codebase files.

## User Review Required

> [!IMPORTANT]
> The prompt specified to "make a [main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py) in the backend" but also "you are not allowed to change anything in the existing codebase". The current repository already has a [backend/main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py) which runs the RL agent offline pipeline. 
> **Question:** Can I create the new real-time entrypoint as `backend/realtime_main.py` (or `api.py`) instead of overwriting the original [main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py) to strictly respect the "do not change existing files" rule? Please clarify if you strictly want it named [main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py) (which requires overwriting the old one).

## Proposed Changes

We will introduce a FastAPI backend to serve as the integration layer for the NN models and data collectors, and a modern Vite+React frontend application.

### Backend

We will create a new real-time orchestrator script that exposes the data, models, and signals via REST/Websocket APIs.

#### [NEW] [realtime_main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/realtime_main.py)
This script will:
- Spin up a FastAPI web server (Uvicorn).
- Initialize [CryptoDataCollector](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/data_collector/crypto_collector.py#6-74) (from `data_collector.crypto_collector`) to pull real-time symbol data.
- Initialize [NewsCollector](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/data_collector/crypto_news.py#8-109) (from `data_collector.crypto_news`) to fetch real-time crypto news.
- Load [LSTMTradingModel](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/training/LSTM.py#6-47) and [TransformerTradingModel](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/training/LSTM.py#48-83) (from `training.LSTM`) utilizing the saved weights in [models/best_lstm.pth](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/models/best_lstm.pth) and [models/best_transformer.pth](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/models/best_transformer.pth).
- Periodically compute technical indicators (`features.indicators.TechnicalIndicators`) on live data to sequence the inputs and pass them to the loaded neural networks.
- Serve JSON responses for the frontend:
  1. `/api/market` -> Latest prices and historical frame.
  2. `/api/news` -> Aggregated news feeds.
  3. `/api/predictions` -> Output from LSTM and Transformer networks (Buy/Sell/Hold signals).

*(Note: If you confirm I should overwrite [main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py), this file will be named [main.py](file:///home/adityagautam/Desktop/Projects/quantTrader/backend/main.py) instead).*

---

### Frontend

We will build a high-quality, modern React dashboard that consumes the APIs from the backend. The UI will feature dark mode styling, glassmorphism, and live charts.

#### [NEW] [frontend setup](file:///home/adityagautam/Desktop/Projects/quantTrader/frontend/)
- Initialize a React application using `npx -y create-vite@latest . --template react-ts`.
- Setup dynamic design assets using standard CSS (no Tailwind per rules unless requested):
  - **Dashboard.tsx**: Main layout organizing the cards.
  - **MarketChart.tsx**: Component rendering real-time candlestick data.
  - **SignalPanel.tsx**: Premium panel displaying real-time LSTM & Transformer confidence levels and the unified trading action.
  - **NewsFeed.tsx**: Scrolling news list highlighting recent crypto sentiments.
- Use `index.css` for rich aesthetics (glow effects, dynamic layout, smooth transitions).

## Verification Plan

### Automated Tests
- Run `python backend/realtime_main.py` and curl `/api/market`, `/api/predictions` to ensure dependencies load correctly, existing models process data without crashing, and paths are correct.
- Verify frontend builds correctly via `npm run build`.

### Manual Verification
- Run the backend FastAPI server on `localhost:8000`.
- Run the frontend via `npm run dev` on `localhost:5173`.
- Open the modern UI in a browser to confirm:
  1. Market statistics and charts populate in real-time.
  2. News collector items appear dynamically.
  3. The LSTM and Transformer models' decisions populate without altering any of their original code.
