echo "Downloading BTC/USDT 5m..."
python -m zhisa.scripts.ingest_real_data --exchange binance --symbol BTC/USDT --timeframe 5m --since 2024-06-01 --max-bars 250000 --db-root data/tsdb

echo "Downloading ETH/USDT 5m..."
python -m zhisa.scripts.ingest_real_data --exchange binance --symbol ETH/USDT --timeframe 5m --since 2024-06-01 --max-bars 250000 --db-root data/tsdb

echo "Downloading SOL/USDT 5m..."
python -m zhisa.scripts.ingest_real_data --exchange binance --symbol SOL/USDT --timeframe 5m --since 2024-06-01 --max-bars 250000 --db-root data/tsdb

echo "Downloading Futures Context..."
python -m zhisa.scripts.ingest_binance_futures_context --symbols BTCUSDT,ETHUSDT,SOLUSDT --timeframe 5m --start 2024-06-01T00:00:00Z --end 2026-06-01T00:00:00Z --out-root data/futures_context/binance_usdm

echo "Done!"
