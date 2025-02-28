import datetime
import os
import time

import requests
import numpy as np
import plotly.graph_objs as go
from flask import Flask, render_template_string
import yfinance as yf

################################################################################
# 1) DATA FETCHING: Polygon vs. Yahoo
################################################################################
def fetch_stock_data_polygon(api_key, symbol, start_date, end_date, next_url=None, limit=5000):
    """
    Fetch stock data from Polygon.io in 1-minute intervals.
    Returns a list of dictionaries with 't','o','h','l','c','v' keys.
    """
    if next_url:
        url = f"{next_url}&adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
    else:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start_date}/{end_date}"
            f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
        )
    response = requests.get(url)
    response_json = response.json()

    if response.status_code != 200:
        print(f"Polygon error: {response.status_code}, {response.text}")
        return [], None

    if "results" not in response_json:
        # Possibly no data found
        return [], None

    # Transform the results to a common structure
    polygon_data = response_json["results"]
    out_data = []
    for row in polygon_data:
        epoch_millis = row['t']  # e.g. 1675084800000
        out_data.append({
            't': epoch_millis,
            'o': row['o'],
            'h': row['h'],
            'l': row['l'],
            'c': row['c'],
            'v': row['v']
        })

    next_url = response_json.get("next_url")
    return out_data, next_url


def fetch_stock_data_yahoo(symbol, start_date, end_date):
    """
    Fetch stock data from Yahoo Finance in 1-minute intervals via yfinance.
    Returns a list of dictionaries with 't','o','h','l','c','v' keys.
    """
    # yfinance formatting: 'start_date', 'end_date', and 'interval'
    # Example: interval = '1m' (1 minute). The free version might not always
    # provide 1m intervals for certain symbols or historical date ranges.
    print(symbol)
    df = yf.download(symbol, start=start_date, end=end_date, interval="1m", progress=False)
    # df = yf.download("SPY", "2025-01-27", interval="1m", progress=False)
    if df.empty:
        return []

    # 'df' has columns: ['Open','High','Low','Close','Adj Close','Volume']
    # The index is a DatetimeIndex in UTC.
    # We'll transform each row to our common structure.
    out_data = []
    for ts, row in df.iterrows():
        # ts is a UTC pandas Timestamp
        epoch_millis = int(ts.timestamp() * 1000)
        out_data.append({
            't': epoch_millis,
            'o': float(row['Open']),
            'h': float(row['High']),
            'l': float(row['Low']),
            'c': float(row['Close']),
            'v': float(row['Volume'])
        })
    return out_data


def get_stock_quotes(data_source, symbol, start_date, end_date, api_key=None):
    """
    Unified function to retrieve stock data from either 'polygon' or 'yahoo'.
    Returns a list of dicts: [{'t':..., 'o':..., 'h':..., 'l':..., 'c':..., 'v':...}, ...]
    """
    if data_source.lower() == "polygon":
        # We'll keep calling until we exhaust 'next_url'
        all_data = []
        next_url = None
        while True:
            data, next_url = fetch_stock_data_polygon(api_key, symbol, start_date, end_date, next_url)
            all_data.extend(data)
            if not next_url or not data:
                break
            time.sleep(1)
        return all_data

    elif data_source.lower() == "yahoo":
        # Single fetch from Yahoo
        return fetch_stock_data_yahoo(symbol, start_date, end_date)

    else:
        raise ValueError("Unknown data source. Must be 'polygon' or 'yahoo'.")
    
# ################################################################################
# # 1) POLYGON FETCH LOGIC
# ################################################################################
# def fetch_stock_data(api_key, symbol, start_date, end_date, next_url=None, limit=5000):
#     if next_url:
#         url = f"{next_url}&adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
#     else:
#         url = (
#             f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
#             f"{start_date}/{end_date}?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
#         )

#     response = requests.get(url)
#     response_json = response.json()

#     if response.status_code == 200 and "results" in response_json:
#         return response_json["results"], response_json.get("next_url")
#     else:
#         print(f"Error fetching data: {response.status_code} - {response.text}")
#         return [], None

# def get_stock_quotes(api_key, symbol, start_date, end_date):
#     all_data = []
#     next_url = None

#     while True:
#         data, next_url = fetch_stock_data(api_key, symbol, start_date, end_date, next_url)
#         if not data:
#             break
#         all_data.extend(data)
#         if not next_url:
#             break
#         time.sleep(1)

#     return all_data

################################################################################
# 2) DATA PROCESSING
################################################################################
def filter_market_hours(data_points):
    """
    Returns only data points whose time is between 9:30 and 16:00 (assumes local exchange time).
    """
    filtered = []
    market_open = datetime.time(7, 30)
    market_close = datetime.time(16, 0)

    for dp in data_points:
        dt = datetime.datetime.fromtimestamp(dp['t'] / 1000)
        if market_open <= dt.time() < market_close:
            filtered.append(dp)
    return filtered

def compute_rolling_slope(data_points, window_size=10):
    """
    For each data point, compute the slope of the best-fit line on the
    last `window_size` close prices. Returns a list of dicts with:
      { 't': <epoch-ms>, 'slope': <float or None> }
    """
    slopes = []
    closes = [dp['c'] for dp in data_points]
    times = [dp['t'] for dp in data_points]  # epoch ms

    for i in range(len(data_points)):
        # Only compute once we have >= window_size points
        if i+1 < window_size:
            slopes.append(None)
            continue
        
        # last `window_size` points
        window_closes = closes[i+1-window_size : i+1]
        # We only need the relative index for polyfit, so x = 0..window_size-1
        x_vals = np.arange(window_size)
        y_vals = np.array(window_closes)

        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        slopes.append(slope)

    slope_points = []
    for dp, slope_val in zip(data_points, slopes):
        slope_points.append({'t': dp['t'], 'slope': slope_val})

    return slope_points

def find_local_minima_below_zero(slope_points):
    """
    Returns a list of indices i at which slope[i] is a local min below 0:
      slope[i] < slope[i-1] AND slope[i] < slope[i+1] AND slope[i] < 0
    Excludes start/end points (which can't be local minima by this definition).
    """
    # Convert slope_points to parallel arrays for easier indexing
    slopes = [sp['slope'] for sp in slope_points]

    local_min_indices = []
    for i in range(1, len(slopes) - 1):
        s_prev = slopes[i-1]
        s_curr = slopes[i]
        s_next = slopes[i+1]

        # We only check local minima if all three slope values are non-None
        if s_prev is not None and s_curr is not None and s_next is not None:
            if s_curr < s_prev and s_curr < s_next and s_curr < 0:
                local_min_indices.append(i)
    return local_min_indices

################################################################################
# 3) PLOT GENERATION (PLOTLY)
################################################################################
def build_plot(data_points, slope_points):
    """
    Build a Plotly figure with:
      - Top subplot: time-series of close prices + local-min markers
      - Bottom subplot: slope (rolling 10-pt best fit)
    """

    # Convert epoch -> datetime for main (top) chart
    x_times = [datetime.datetime.fromtimestamp(dp['t'] / 1000) for dp in data_points]
    y_closes = [dp['c'] for dp in data_points]

    # For the slope (bottom chart), ignoring None values
    x_times_slope = [datetime.datetime.fromtimestamp(sp['t'] / 1000)
                     for sp in slope_points if sp['slope'] is not None]
    y_slopes = [sp['slope'] for sp in slope_points if sp['slope'] is not None]

    # 3A) Find local minima below zero
    local_min_indices = find_local_minima_below_zero(slope_points)

    # For each local minimum index, we retrieve the exact time and matching close price
    # We'll create a dictionary for quick timestamp -> close lookups
    close_by_time = {dp['t']: dp['c'] for dp in data_points}
    min_mark_times = []
    min_mark_closes = []

    for idx in local_min_indices:
        t = slope_points[idx]['t']       # epoch ms
        slope_val = slope_points[idx]['slope']
        # Convert epoch to datetime
        dt_obj = datetime.datetime.fromtimestamp(t / 1000)
        if t in close_by_time:
            min_mark_times.append(dt_obj)
            min_mark_closes.append(close_by_time[t])

    # 3B) Build main chart (top subplot)
    import plotly.subplots as sp
    fig = sp.make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Close Prices (Market Hours)", "Rolling 10-Point Slope")
    )

    # Price trace
    trace_price = go.Scatter(
        x=x_times,
        y=y_closes,
        mode='lines+markers',
        name='Close Price'
    )

    # Marker for local minima
    trace_minima = go.Scatter(
        x=min_mark_times,
        y=min_mark_closes,
        mode='markers',
        name='Local Min Slope < 0',
        marker=dict(color='red', size=10, symbol='triangle-down')
    )

    # Slope trace (bottom subplot)
    trace_slope = go.Scatter(
        x=x_times_slope,
        y=y_slopes,
        mode='lines+markers',
        name='Slope (Last 10 Obs)',
        marker=dict(color='blue')
    )

    # Add them to the figure
    fig.add_trace(trace_price, row=1, col=1)
    fig.add_trace(trace_minima, row=1, col=1)
    fig.add_trace(trace_slope, row=2, col=1)

    fig.update_layout(
        height=700,
        title="Live Stock Data with Rolling Slope & Local-Min Markers",
        showlegend=True
    )

    return fig.to_html(full_html=False)

################################################################################
# 4) FLASK APP FOR EMBEDDING
################################################################################
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Data</title>
</head>
<body>
    <h1>Interactive Stock Chart</h1>
    <div>
        {{ plot_div | safe }}
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    # 1) Get data (including pre-/post-market). For demonstration, pick a date range
    # api_key = "YOUR_API_KEY"
    api_key = os.environ.get('API_KEY')
    symbol = "SPY"
    today = datetime.date.today()
    # start_date_str = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start_date_str = today.strftime('%Y-%m-%d')
    end_date_str = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    # end_date_str   = today.strftime('%Y-%m-%d')

    raw_data = get_stock_quotes("yahoo", symbol, start_date_str, end_date_str)

    # 2) Compute slope using all data
    slope_points = compute_rolling_slope(raw_data)

    # 3) Filter the data so the chart only *displays* 9:30-16:00
    display_data = filter_market_hours(raw_data)

    # We also want to filter the slope points to keep consistent times displayed
    # (the slope calculation used all data, but for the chart, we only show points within market hours).
    display_slope_points = []
    market_open = datetime.time(7, 30)
    market_close = datetime.time(16, 0)
    for sp in slope_points:
        dt = datetime.datetime.fromtimestamp(sp['t'] / 1000)
        if market_open <= dt.time() < market_close:
            display_slope_points.append(sp)

    # 4) Build the Plotly figure
    plot_html = build_plot(display_data, display_slope_points)

    return render_template_string(HTML_TEMPLATE, plot_div=plot_html)

if __name__ == "__main__":
    app.run(debug=True)
