#!/bin/bash

echo "🚀 Starting QuantTrader Real-Time Platform..."
echo ""

# Activate conda environment
source /home/adityagautam/mambaforge/bin/activate main

# Check if we're in the right directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Run this script from the quantTrader root directory"
    exit 1
fi

# Start backend
echo "📊 Starting Backend API..."
cd backend
python realtime_api.py > /tmp/quanttrader_backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start frontend
echo "🎨 Starting Frontend Dashboard..."
cd frontend
npm run dev > /tmp/quanttrader_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"
cd ..

sleep 2

echo ""
echo "✅ QuantTrader is running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo ""
echo "   Backend logs:  tail -f /tmp/quanttrader_backend.log"
echo "   Frontend logs: tail -f /tmp/quanttrader_frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
