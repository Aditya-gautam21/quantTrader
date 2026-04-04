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

const generateInitialData = (minutesPerCandle: number) => {
  const data = [];
  const now = Math.floor(Date.now() / 1000);
  let lastVal = 3500;
  const intervalSeconds = minutesPerCandle * 60;
  // 60 points for 60 intervals
  for (let i = 60; i >= 0; i--) {
    const time = (now - i * intervalSeconds) as Time;
    lastVal = lastVal + (Math.random() - 0.48) * (10 * Math.sqrt(minutesPerCandle));
    data.push({ time, value: lastVal });
  }
  return data;
};

function App() {
  const [timeframe, setTimeframe] = useState<number>(5);
  const [tradeMode, setTradeMode] = useState<'buy' | 'sell'>('buy');
  const [cryptAmount, setCryptAmount] = useState<string>('12.695');
  const [fiatAmount, setFiatAmount] = useState<string>('9,853.00');
  
  const [currentPrice, setCurrentPrice] = useState<number>(3615.86);
  const [priceChange, setPriceChange] = useState<number>(3.27);
  
  const chartContainerRef = useRef<HTMLDivElement>(null);

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

    const initData = generateInitialData(timeframe);
    newSeries.setData(initData);

    let lastTime = initData[initData.length - 1].time as number;
    let lastValue = initData[initData.length - 1].value;
    setCurrentPrice(lastValue);

    const intervalSeconds = timeframe * 60;

    const interval = setInterval(() => {
      const now = Math.floor(Date.now() / 1000);
      lastValue = lastValue + (Math.random() - 0.45) * 5;
      
      // If we crossed into a new timeframe boundary, update the lastTime explicitly
      if (now - lastTime >= intervalSeconds) {
        lastTime = now - (now % intervalSeconds);
      }
      
      newSeries.update({
        time: lastTime as Time,
        value: lastValue
      });
      
      setCurrentPrice(lastValue);
    }, 1000);

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [timeframe]);

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
                  <div className="icon-2"><div style={{fontSize: '12px', fontWeight: 'bold', color: 'white'}}>$</div></div>
                </div>
                ETH/USD <ChevronDown size={20} color="var(--text-muted)" />
              </div>
              <div className="price-display">
                <div className="price-value">${currentPrice.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
                <div className="price-change">{priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}% today</div>
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
                <th>BNB/USD</th>
                <th>Amount</th>
                <th>Diff</th>
                <th style={{textAlign: 'right'}}>Volume</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <div className="exchange-cell">
                    <img src="https://cryptologos.cc/logos/uniswap-uni-logo.svg?v=026" alt="UniSwap" style={{width:24, height:24}} />
                    UniSwap
                  </div>
                </td>
                <td>3,615.32</td>
                <td>1.6254 ETH</td>
                <td><span className="tag-limited">Limited</span></td>
                <td style={{textAlign: 'right'}}>$5,875.00</td>
              </tr>
              <tr>
                <td>
                  <div className="exchange-cell">
                    <img src="https://cryptologos.cc/logos/sushiswap-sushi-logo.svg?v=026" alt="SushiSwap" style={{width:24, height:24}} />
                    SushiSwap
                  </div>
                </td>
                <td>3,617.12</td>
                <td>1.6203 ETH</td>
                <td><span className="tag-trending">Trending</span></td>
                <td style={{textAlign: 'right'}}>$5,860.12</td>
              </tr>
              <tr>
                <td>
                  <div className="exchange-cell">
                    <img src="https://cryptologos.cc/logos/pancakeswap-cake-logo.svg?v=026" alt="PancakeSwap" style={{width:24, height:24}} />
                    PancakeSwap
                  </div>
                </td>
                <td>3,620.00</td>
                <td>1.5000 ETH</td>
                <td><span className="tag-rising">Rising</span></td>
                <td style={{textAlign: 'right'}}>$5,430.00</td>
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
                  ETH
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
            {tradeMode === 'buy' ? 'Buy ETH' : 'Sell ETH'}
          </button>
          
          <button className="btn-action-secondary">
            Connect Wallet <Wallet size={16} />
          </button>

          <div className="summary-card">
            <div className="summary-header">Available Balance</div>
            <div className="summary-balance">
              293.0187 ETH <span className="summary-balance-pct">+7.45%</span>
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
