// Types for our APIs and components
export interface MarketDataPoint {
  Timestamp: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

export interface NewsItem {
  source: string;
  title: string;
  link: string;
  published: string;
  summary: string;
}

export interface PredictionData {
  LSTM_Signal: "BUY" | "SELL" | "HOLD";
  Transformer_Signal: "BUY" | "SELL" | "HOLD";
  Confidence: number;
}
