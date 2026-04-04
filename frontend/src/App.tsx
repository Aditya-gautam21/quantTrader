import { useState, useEffect, useRef } from 'react';
import { 
  ChevronDown, 
  Settings, 
  RotateCw, 
  ScanLine, 
  ArrowDownUp, 
  Wallet,
  Activity,
  LogOut
} from 'lucide-react';
import { createChart, ColorType, Time, AreaSeries } from 'lightweight-charts';

import axios from 'axios';
const API_URL = 'http://localhost:8000/api';

function App() {
  const [timeframe, setTimeframe] = useState<number>(5);
  const [marketData, setMarketData] = useState<any[]>([]);
  const [prediction, setPrediction] = useState<any>(null);
  
  const [tradeMode, setTradeMode] = useState<'buy' | 'sell'>('buy');
  const [cryptAmount, setCryptAmount] = useState<string>('0.05');
  const [fiatAmount, setFiatAmount] = useState<string>('3000.00');
  
  const chartContainerRef = useRef<HTMLDivElement>(null);
  
  const currentPrice = marketData.length > 0 ? marketData[marketData.length - 1].value : 0;
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [marketRes, predRes] = await Promise.all([
          axios.get(`${API_URL}/market/BTC%2FUSDT`),
          axios.get(`${API_URL}/predictions/BTC%2FUSDT`)
        ]);

        if (marketRes.data?.data) {
          const formatted = marketRes.data.data.map((d: any) => ({
            time: Math.floor(new Date(d.Timestamp).getTime() / 1000) as Time,
            value: Number(d.Close)
          }));
          const uniqueData = Array.from(new Map(formatted.map((item: any) => [item.time, item])).values());
          
          uniqueData.sort((a: any, b: any) => a.time - b.time);
          
          setMarketData(uniqueData as any);
        }

        if (predRes.data?.prediction) {
          setPrediction(predRes.data.prediction);
        }
      } catch (error) {
        console.error("API error", error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#848A96',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
      },
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
        timeVisible: true,
      },
      crosshair: {
        vertLine: {
          color: 'rgba(255, 255, 255, 0.1)',
          width: 1,
          style: 3,
        },
        horzLine: {
          color: 'rgba(255, 255, 255, 0.1)',
          width: 1,
          style: 3,
        },
      },
    });

    const newSeries = chart.addSeries(AreaSeries, {
      lineColor: '#DDF77D',
      topColor: 'rgba(221, 247, 125, 0.3)',
      bottomColor: 'rgba(221, 247, 125, 0.0)',
      lineWidth: 2,
    });

    newSeries.setData(marketData);

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [marketData]);

  return (
    <div className="app-window">
      {/* HEADER */}
      <header className="top-nav">
        <div className="logo-section">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L2 22h20L12 2z" fill="url(#luma-gradient)" />
            <defs>
              <linearGradient id="luma-gradient" x1="12" y1="2" x2="12" y2="22" gradientUnits="userSpaceOnUse">
                <stop stopColor="#DDF77D" />
                <stop offset="1" stopColor="#344C3D" />
              </linearGradient>
            </defs>
          </svg>
          quantTrader
        </div>
        
        <div className="nav-links">
          <div className="nav-link active">Dashboard</div>
          <div className="nav-link">Trade</div>
          <div className="nav-link">Market <ChevronDown size={14} /></div>
        </div>
        
        <div className="user-section">
          <div className="address-pill">
            <img src="https://api.dicebear.com/7.x/pixel-art/svg?seed=adityagautam" alt="user avatar" />
            0xA74F...23B0F
          </div>
          <button className="btn-primary">
            Log in <LogOut size={16} />
          </button>
        </div>
      </header>

      {/* MAIN LAYOUT */}
      <div className="dashboard-content">
        
        {/* LEFT PANEL */}
        <div className="left-panel">
          
          <div className="chart-header">
            <div className="pair-info">
              <div className="pair-title">
                <div className="crypto-icon-group">
                  <div className="icon-1"><Activity size={18} color="white" /></div>
                  <div className="icon-2"><div style={{fontSize: '12px', fontWeight: 'bold', color: 'white'}}>₿</div></div>
                </div>
                BTC/USDT <ChevronDown size={20} color="var(--text-muted)" />
              </div>
              <div className="price-display">
                <div className="price-value">${currentPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                <div className="price-change">{prediction ? `${prediction.LSTM_Signal} Signal (${prediction.Confidence}%)` : 'Analyzing'}</div>
              </div>
            </div>
            
            <div className="chart-tools-container" style={{display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'flex-end'}}>
              <div className="chart-tools">
                <button className="tool-btn"><Activity size={18} /></button>
                <button className="tool-btn"><Settings size={18} /></button>
              </div>
              <div className="chart-filters">
                {[
                  { label: '1m', value: 1 },
                  { label: '5m', value: 5 },
                  { label: '15m', value: 15 },
                  { label: '1h', value: 60 },
                  { label: '4h', value: 240 }
                ].map((tf) => (
                  <button 
                    key={tf.label}
                    className={`chart-filter-btn ${timeframe === tf.value ? 'active' : ''}`}
                    onClick={() => setTimeframe(tf.value)}
                  >
                    {tf.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="chart-container" ref={chartContainerRef} style={{height: '300px', width: '100%', position: 'relative'}}>
          </div>

          <table className="exchanges-table">
            <thead>
              <tr>
                <th>Exchange</th>
                <th>BTC/USDT Model</th>
                <th>Confidence</th>
                <th>Signal</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <div className="exchange-cell">
                    <img src="https://cryptologos.cc/logos/bitcoin-btc-logo.svg?v=026" alt="QuantTrader" style={{width:24, height:24}} />
                    LSTM Model
                  </div>
                </td>
                <td>{prediction ? prediction.LSTM_Signal : 'Loading...'}</td>
                <td>{prediction ? prediction.Confidence + '%' : '...'}</td>
                <td>
                  <span className={`tag-${prediction?.LSTM_Signal === 'BUY' ? 'rising' : prediction?.LSTM_Signal === 'SELL' ? 'limited' : 'trending'}`}>
                    {prediction ? prediction.LSTM_Signal : 'HOLD'}
                  </span>
                </td>
              </tr>
              <tr>
                <td>
                  <div className="exchange-cell">
                    <img src="https://cryptologos.cc/logos/bitcoin-btc-logo.svg?v=026" alt="QuantTrader" style={{width:24, height:24}} />
                    Transformer Model
                  </div>
                </td>
                <td>{prediction ? prediction.Transformer_Signal : 'Loading...'}</td>
                <td>{prediction ? prediction.Confidence + '%' : '...'}</td>
                <td>
                  <span className={`tag-${prediction?.Transformer_Signal === 'BUY' ? 'rising' : prediction?.Transformer_Signal === 'SELL' ? 'limited' : 'trending'}`}>
                    {prediction ? prediction.Transformer_Signal : 'HOLD'}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>

        </div>

        {/* RIGHT PANEL - TRADE WIDGET */}
        <div className="trade-widget">
          
          <div className="trade-header">
            <div className="trade-tabs">
              <button 
                className={`trade-tab ${tradeMode === 'buy' ? 'active' : ''}`}
                onClick={() => setTradeMode('buy')}
              >
                BUY
              </button>
              <button 
                className={`trade-tab ${tradeMode === 'sell' ? 'active' : ''}`}
                onClick={() => setTradeMode('sell')}
              >
                SELL
              </button>
            </div>
            <div className="chart-tools" style={{marginLeft: 0}}>
              <button className="tool-btn"><RotateCw size={16} /></button>
              <button className="tool-btn"><ScanLine size={16} /></button>
              <button className="tool-btn"><Settings size={16} /></button>
            </div>
          </div>

          <div className="trade-top-panel">
            
            <div className="input-card">
              <div className="input-left">
                <div className="crypto-select">
                  <div style={{width: 28, height: 28, borderRadius: '50%', background: '#3E8BFA', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                    <Activity size={14} color="white" />
                  </div>
                  BTC
                </div>
                <input 
                  type="text" 
                  className="input-field" 
                  value={cryptAmount}
                  onChange={(e) => setCryptAmount(e.target.value)}
                  placeholder="0.00"
                />
              </div>
              <div className="input-right">
                <span className="input-label">You {tradeMode === 'buy' ? 'Buy' : 'Sell'}</span>
                <div style={{marginTop: 'auto'}}>
                  <div className="balance-label">Balance</div>
                  <div className="balance-value">293.0187</div>
                </div>
              </div>
            </div>

            <div className="switch-wrapper">
              <button className="switch-btn">
                <ArrowDownUp size={18} />
              </button>
            </div>

            <div className="input-card">
              <div className="input-left">
                <div className="crypto-select">
                  <div style={{width: 28, height: 28, borderRadius: '50%', background: '#22C55E', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                    <div style={{fontSize:'14px', fontWeight:'bold', color:'white'}}>$</div>
                  </div>
                  USD
                </div>
                <input 
                  type="text" 
                  className="input-field" 
                  value={fiatAmount}
                  onChange={(e) => setFiatAmount(e.target.value)}
                  placeholder="0.00"
                />
              </div>
              <div className="input-right">
                <span className="input-label">You {tradeMode === 'buy' ? 'Spend' : 'Receive'}</span>
                <div style={{marginTop: 'auto'}}>
                  <div className="balance-label">Balance</div>
                  <div className="balance-value">12,987.21</div>
                </div>
              </div>
            </div>

          </div>

          <button className="btn-action-primary">
            {tradeMode === 'buy' ? 'Buy BTC' : 'Sell BTC'}
          </button>
          
          <button className="btn-action-secondary">
            Connect Wallet <Wallet size={16} />
          </button>

          <div className="summary-card">
            <div className="summary-header">Available Balance</div>
            <div className="summary-balance">
              1.0182 BTC <span className="summary-balance-pct">+7.45%</span>
            </div>
            
            <div className="summary-details">
              <div className="detail-item">
                <span className="detail-label">Estimate fee</span>
                <span className="detail-value">4.28 USD</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">You will receive</span>
                <span className="detail-value">108.35 USD</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Spread</span>
                <span className="detail-value">0%</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}

export default App;
