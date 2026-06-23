$coins = @("ADA/USDT", "BNB/USDT", "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
foreach ($coin in $coins) {
    echo "Downloading $coin 15m..."
    python -m zhisa.scripts.ingest_real_data --exchange binance --symbol $coin --timeframe 15m --since "2019-01-01" --until "2026-05-17T23:59:59Z" --max-bars 400000 --db-root data/tsdb
}
echo "Done downloading all 15m coins!"
