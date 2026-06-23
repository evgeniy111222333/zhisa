$coins = @(
    @("BTC/USDT", "2019-09-08"),
    @("ETH/USDT", "2019-11-27"),
    @("XRP/USDT", "2020-01-06"),
    @("ADA/USDT", "2020-01-31"),
    @("BNB/USDT", "2020-02-10"),
    @("SOL/USDT", "2020-09-14")
)

foreach ($c in $coins) {
    $sym = $c[0]
    $start = $c[1]
    echo "Downloading full context for $sym starting from $start..."
    python src/zhisa/scripts/download_full_context.py --symbol $sym --start $start --end "2026-05-17"
}
echo "Done downloading context for all 6 coins!"
