from zhisa.scripts.download_massive import last_closed_bar_open_ms, parse_date


def test_last_closed_bar_open_uses_requested_interval():
    interval = 15 * 60_000
    assert last_closed_bar_open_ms(31 * 60_000, interval) == 15 * 60_000
    assert last_closed_bar_open_ms(30 * 60_000, interval) == 15 * 60_000


def test_parse_date_accepts_explicit_utc_timestamp():
    assert parse_date("2024-01-01T01:00:00Z") - parse_date("2024-01-01") == 3_600_000
