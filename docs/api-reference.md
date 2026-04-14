# API Reference

## Download Module

The `egxpy.download` module provides functions to fetch market data from the Egyptian Exchange (EGX).

---

### `egxpy.download.get_OHLCV_data`

Fetches close prices for a single ticker.

```python
egxpy.download.get_OHLCV_data(symbol, exchange, interval, n_bars)
```

**Parameters:**

| Parameter  | Type  | Description                              |
|------------|-------|------------------------------------------|
| `symbol`   | `str` | Ticker symbol                            |
| `exchange` | `str` | Exchange name                            |
| `interval` | `str` | Time interval: `'Daily'`, `'Weekly'`, `'Monthly'` |
| `n_bars`   | `int` | Number of last n bars to retrieve        |

**Returns:** `pd.DataFrame` - OHLCV data for the requested ticker.

---

### `egxpy.download.get_EGXdata`

Fetches historical close prices data for EGX stocks.

```python
egxpy.download.get_EGXdata(stock_list: list, interval: str, start: date, end: date)
```

**Parameters:**

| Parameter    | Type   | Description                                        |
|--------------|--------|----------------------------------------------------|
| `stock_list` | `list` | Desired stock list                                 |
| `interval`   | `str`  | Time interval: `'Daily'`, `'Weekly'`, `'Monthly'`  |
| `start`      | `date` | Starting date                                      |
| `end`        | `date` | End date                                            |

**Returns:** `pd.DataFrame` - Historical close prices for the requested stocks.

---

### `egxpy.download.get_EGX_intraday_data`

Fetches intraday data for EGX stocks.

```python
egxpy.download.get_EGX_intraday_data(stock_list: list, interval: str, start: date, end: date)
```

**Parameters:**

| Parameter    | Type   | Description                                            |
|--------------|--------|--------------------------------------------------------|
| `stock_list` | `list` | Desired stocks                                         |
| `interval`   | `str`  | Time interval: `'1 Minute'`, `'5 Minute'`, `'30 Minute'` |
| `start`      | `date` | Starting date                                          |
| `end`        | `date` | End date                                                |

**Returns:** `pd.DataFrame` - Intraday close prices for the requested stocks.
