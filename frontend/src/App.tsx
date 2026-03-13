import { useEffect, useState } from 'react';
import axios from 'axios';
import { Activity, Brain, Newspaper, TrendingUp, Zap } from 'lucide-react';
import { MarketDataPoint, NewsItem, PredictionData } from './types';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

const API_URL = 'http://localhost:8000/api';

function App() {
  const [marketData, setMarketData] = useState<MarketDataPoint[]>([]);
  const [news, setNews] = useState<NewsItem[]>([]);
  const [predictions, setPredictions] = useState<PredictionData>({
    LSTM_Signal: "HOLD",
    Transformer_Signal: "HOLD",
    Confidence: 0
  });
  
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [symbol] = useState<string>("BTC/USDT");

  const fetchData = async () => {
    try {
      const [marketRes, newsRes, predRes] = await Promise.all([
        axios.get(`${API_URL}/market/${symbol.replace('/', '%2F')}`),
        axios.get(`${API_URL}/news`),
        axios.get(`${API_URL}/predictions/${symbol.replace('/', '%2F')}`)
      ]);

      if (marketRes.data?.data) {
        // Format timestamp for chart
        const formatted = marketRes.data.data.map((d: any) => ({
          ...d,
          timeStr: new Date(d.Timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
        }));
        setMarketData(formatted);
      }
      
      if (newsRes.data?.news) setNews(newsRes.data.news);
      if (predRes.data?.prediction) setPredictions(predRes.data.prediction);
      
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Failed fetching data", error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Poll every minute
    return () => clearInterval(interval);
  }, [symbol]);

  const currentPrice = marketData.length > 0 ? marketData[marketData.length - 1].Close : 0;
  const prevPrice = marketData.length > 1 ? marketData[marketData.length - 2].Close : 0;
  const priceChange = currentPrice - prevPrice;
  const priceChangePct = prevPrice > 0 ? (priceChange / prevPrice) * 100 : 0;
  
  const isPositive = priceChange >= 0;

  return (
    <div className="app-container">
      <header>
        <div className="header-title">
          <Zap className="text-accent-blue" size={28} />
          <span>QuantTrader AI Dashboard</span>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-sm text-muted hidden md:block">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
          <div className="status-badge">
            <div className="status-dot"></div>
            SYSTEM LIVE
          </div>
        </div>
      </header>

      <div className="dashboard-grid">
        {/* CHART SECTION */}
        <section className="chart-section glass-panel">
          <div className="flex justify-between items-start mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xl font-bold">{symbol}</span>
                <span className="text-xs px-2 py-1 rounded bg-[rgba(255,255,255,0.1)] text-white">4H</span>
              </div>
              <div className="flex items-end gap-3">
                <span className="text-4xl font-mono font-bold">
                  ${currentPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                </span>
                <span className={`text-lg font-medium pb-1 ${isPositive ? 'text-accent-green' : 'text-accent-red'}`}>
                  {isPositive ? '+' : ''}{priceChangePct.toFixed(2)}%
                </span>
              </div>
            </div>
            
            <div className="flex bg-[rgba(255,255,255,0.05)] border border-[rgba(255,255,255,0.1)] rounded-lg overflow-hidden">
              <button className="px-4 py-2 text-sm font-medium bg-[rgba(62,139,250,0.2)] text-white">Price</button>
              <button className="px-4 py-2 text-sm font-medium text-muted hover:text-white transition-colors">Volume</button>
            </div>
          </div>
          
          <div className="flex-1 w-full mt-4 min-h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={marketData} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={isPositive ? "var(--accent-green)" : "var(--accent-red)"} stopOpacity={0.3}/>
                    <stop offset="95%" stopColor={isPositive ? "var(--accent-green)" : "var(--accent-red)"} stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis 
                  dataKey="timeStr" 
                  stroke="var(--text-muted)" 
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={30}
                />
                <YAxis 
                  domain={['auto', 'auto']} 
                  stroke="var(--text-muted)" 
                  fontSize={12}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(val) => `$${val.toLocaleString()}`}
                  width={80}
                  orientation="right"
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'var(--bg-panel)', 
                    borderColor: 'var(--border-color)',
                    borderRadius: '12px',
                    boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                    backdropFilter: 'blur(12px)'
                  }}
                  itemStyle={{ color: 'var(--text-main)', fontFamily: 'Space Grotesk' }}
                  labelStyle={{ color: 'var(--text-muted)', marginBottom: '4px' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="Close" 
                  stroke={isPositive ? "var(--accent-green)" : "var(--accent-red)"} 
                  strokeWidth={2}
                  fillOpacity={1} 
                  fill="url(#colorClose)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* AI SIGNALS SECTION */}
        <section className="signals-section glass-panel">
          <div className="panel-header">
            <Brain size={20} />
            <h2>AI Intelligence Signals</h2>
          </div>
          
          <div className="signal-card">
            <div>
              <div className="signal-model-name flex items-center gap-1">
                <Activity size={12} />
                LSTM Network
              </div>
              <div className={`signal-value signal-${predictions.LSTM_Signal.toLowerCase()}`}>
                {predictions.LSTM_Signal}
              </div>
            </div>
          </div>
          
          <div className="signal-card">
            <div>
              <div className="signal-model-name flex items-center gap-1">
                <TrendingUp size={12} />
                Transformer Model
              </div>
              <div className={`signal-value signal-${predictions.Transformer_Signal.toLowerCase()}`}>
                {predictions.Transformer_Signal}
              </div>
            </div>
          </div>
          
          <div className="mt-6 mb-2 flex justify-between text-sm font-medium">
            <span className="text-muted">Aggregate Confidence</span>
            <span className="text-white">{predictions.Confidence.toFixed(1)}%</span>
          </div>
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar" 
              style={{ width: `${predictions.Confidence}%` }}
            ></div>
          </div>
        </section>

        {/* NEWS FEED SECTION */}
        <section className="news-section glass-panel">
          <div className="panel-header">
            <Newspaper size={20} />
            <h2>Market Sentiment Feed</h2>
          </div>
          
          <div className="news-list">
            {news.length > 0 ? news.map((item, idx) => (
              <a href={item.link} target="_blank" rel="noreferrer" key={idx} className="news-item block" style={{ textDecoration: 'none' }}>
                <div className="news-source">{item.source}</div>
                <div className="news-title">{item.title}</div>
                <div className="news-meta">
                  <span>{new Date(item.published).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                </div>
              </a>
            )) : (
               <div className="text-center text-muted p-4 border border-dashed border-[rgba(255,255,255,0.1)] rounded-lg mt-4">
                  Awaiting real-time news data...
               </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
