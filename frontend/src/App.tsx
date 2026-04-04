import React, { useState, useEffect } from 'react';
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
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

// Mock Chart Data to match the visual curve
const chartData = [
  { time: '2:00 AM', value: 3620 },
  { time: '3:00 AM', value: 3560 },
  { time: '4:00 AM', value: 3555 },
  { time: '5:00 AM', value: 3610 },
  { time: '6:00 AM', value: 3580 },
  { time: '7:00 AM', value: 3530 },
  { time: '8:00 AM', value: 3540 },
  { time: '9:00 AM', value: 3560 },
  { time: '10:00 AM', value: 3550 },
  { time: '11:00 AM', value: 3570 },
  { time: '12:00 PM', value: 3580 },
  { time: '1:00 PM', value: 3590 },
  { time: '2:00 PM', value: 3615.86 },
  { time: '3:00 PM', value: 3590 },
  { time: '4:00 PM', value: 3630 },
];

function App() {
  const [tradeMode, setTradeMode] = useState<'buy' | 'sell'>('buy');
  const [cryptAmount, setCryptAmount] = useState<string>('12.695');
  const [fiatAmount, setFiatAmount] = useState<string>('9,853.00');
  
  // Custom tooltip for chart
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          padding: '8px',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          color: 'white',
          fontWeight: 600,
          fontSize: '14px'
        }}>
          {payload[0].value.toFixed(2)}
        </div>
      );
    }
    return null;
  };

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
                <div className="price-value">$3,615.86</div>
                <div className="price-change">+3.27% today</div>
              </div>
            </div>
            
            <div className="chart-tools-container" style={{display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'flex-end'}}>
              <div className="chart-tools">
                <button className="tool-btn"><Activity size={18} /></button>
                <button className="tool-btn"><Settings size={18} /></button>
              </div>
              <div className="chart-filters">
                <button className="chart-filter-btn">1h</button>
                <button className="chart-filter-btn">24h</button>
                <button className="chart-filter-btn">1w</button>
                <button className="chart-filter-btn active">1m</button>
              </div>
            </div>
          </div>

          <div className="chart-container">
             <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={chartData} margin={{ top: 20, right: 0, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#DDF77D" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#DDF77D" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
                  <XAxis 
                    dataKey="time" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{fill: 'var(--text-muted)', fontSize: 12}}
                    minTickGap={30}
                    dy={10}
                  />
                  <YAxis 
                    domain={['dataMin - 10', 'dataMax + 10']} 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{fill: 'var(--text-muted)', fontSize: 12}}
                    tickFormatter={(val) => val.toFixed(2)}
                    width={80}
                    hide={false}
                    orientation="left"
                    dx={-10}
                  />
                  <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.1)', strokeWidth: 1, strokeDasharray: '3 3' }} />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="#DDF77D" 
                    strokeWidth={2.5}
                    fill="url(#colorValue)" 
                    activeDot={{ r: 6, fill: '#DDF77D', stroke: '#1A1D1F', strokeWidth: 2 }}
                  />
                </AreaChart>
             </ResponsiveContainer>
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
