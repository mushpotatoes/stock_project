import os
import requests
import logging
import matplotlib.pyplot as plt
from datetime import datetime, time
from matplotlib.dates import DateFormatter # Import DateFormatter

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_stock_data(api_key, symbol, date, limit=5000, next_url=None):
    """Fetches aggregated stock data for a given day using Polygon.io API."""
    if next_url:
        url = next_url + f"&apiKey={api_key}"
    else:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{date}/{date}?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
        )

    logging.info(f"Fetching stock data from: {url.split('apiKey')[0]}...")
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("results", []), response_json.get("next_url")
    else:
        logging.error(f"Error fetching stock data: HTTP {response.status_code} - {response.text}")
        return [], None


def fetch_option_data(api_key, option_ticker, date, limit=5000, next_url=None):
    """Fetches aggregated option data for a given day using Polygon.io API."""
    if next_url:
        url = next_url + f"&apiKey={api_key}"
    else:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/minute/"
            f"{date}/{date}?limit={limit}&apiKey={api_key}"
        )

    logging.info(f"Fetching option data from: {url.split('apiKey')[0]}...")
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        return response_json.get("results", []), response_json.get("next_url")
    else:
        logging.error(f"Error fetching option data: HTTP {response.status_code} - {response.text}")
        return [], None


def construct_option_ticker(underlying, expiration, strike, call=True):
    """Constructs a Polygon.io option ticker string."""
    try:
        year, month, day = expiration.split("-")
        exp_formatted = f"{year[2:]}{month}{day}"
        option_type = "C" if call else "P"
        strike_int = int(round(strike * 1000))
        strike_formatted = f"{strike_int:08d}"
        return f"O:{underlying}{exp_formatted}{option_type}{strike_formatted}"
    except ValueError:
        raise ValueError("Expiration date must be in YYYY-MM-DD format")


def plot_put_credit_spread(stock_data, short_put_data, long_put_data, symbol, short_strike, long_strike, start_time_str='08:30', end_time_str='15:00'):
    """
    Plots the underlying price, two put premiums, and the calculated credit spread
    within a specified time window.
    """
    # 1. Process underlying stock data
    if not stock_data:
        logging.error("No stock data to plot.")
        return
    stock_times = [datetime.fromtimestamp(item["t"] / 1000) for item in stock_data]
    stock_closes = [item["c"] for item in stock_data]
    trade_date = stock_times[0].date()

    # 2. Synchronize the two option datasets by timestamp
    short_put_prices = {item["t"]: item["c"] for item in short_put_data}
    long_put_prices = {item["t"]: item["c"] for item in long_put_data}
    common_timestamps = sorted(list(set(short_put_prices.keys()) & set(long_put_prices.keys())))

    if not common_timestamps:
        logging.error("No common timestamps found between options. Cannot plot spread.")
        return

    # 3. Create aligned data lists and calculate the spread value
    spread_times = [datetime.fromtimestamp(ts / 1000) for ts in common_timestamps]
    aligned_short_prices = [short_put_prices[ts] for ts in common_timestamps]
    aligned_long_prices = [long_put_prices[ts] for ts in common_timestamps]
    spread_values = [short - long for short, long in zip(aligned_short_prices, aligned_long_prices)]

    # 4. Create the 3-part plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{symbol} Put Credit Spread Analysis for {trade_date}", fontsize=16)

    # --- Subplot 1: Underlying Stock Price ---
    axes[0].plot(stock_times, stock_closes, label=f'{symbol} Close', color='dodgerblue')
    axes[0].set_title(f'{symbol} Price')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True)

    # --- Subplot 2: Individual Put Option Premiums ---
    axes[1].plot(spread_times, aligned_short_prices, label=f'Short Put ({short_strike})', color='red')
    axes[1].plot(spread_times, aligned_long_prices, label=f'Long Put ({long_strike})', color='green')
    axes[1].set_title('Put Option Premiums')
    axes[1].set_ylabel('Premium ($)')
    axes[1].legend()
    axes[1].grid(True)

    # --- Subplot 3: Put Credit Spread Value ---
    axes[2].plot(spread_times, spread_values, label=f'Spread Value ({short_strike}/{long_strike})', color='purple')
    axes[2].set_title('Credit Spread Value')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Spread Credit ($)')
    axes[2].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[2].legend()
    axes[2].grid(True)
    
    # 5. Set plot time limits
    try:
        start_t = datetime.strptime(start_time_str, '%H:%M').time()
        end_t = datetime.strptime(end_time_str, '%H:%M').time()
        start_limit = datetime.combine(trade_date, start_t)
        end_limit = datetime.combine(trade_date, end_t)
        axes[2].set_xlim(start_limit, end_limit)
        logging.info(f"Setting plot time range from {start_time_str} to {end_time_str}.")
    except ValueError:
        logging.warning("Invalid time format for plot limits. Please use HH:MM. Using default data range.")

    # 6. Format the x-axis to show time as HH:MM
    date_format = DateFormatter('%H:%M')
    axes[2].xaxis.set_major_formatter(date_format)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    """Main function to define parameters, fetch data, and generate the plot."""
    api_key = os.environ.get('API_KEY')
    if not api_key:
        raise EnvironmentError("API_KEY environment variable not found.")

    # --- User-defined parameters for the Put Credit Spread ---
    underlying_symbol = "SPY"
    eval_date = "2025-07-22"
    expiration_date = "2025-07-22"
    short_put_strike = 628.0  
    long_put_strike = 627.0
    
    # Optional: Set the time range for the plot in local time (HH:MM format).
    plot_start_time = '08:30'
    plot_end_time = '15:00'
    # -----------------------------------------------------------

    # 1. Fetch data
    stock_data, _ = fetch_stock_data(api_key, underlying_symbol, eval_date)
    short_put_ticker = construct_option_ticker(underlying_symbol, expiration_date, short_put_strike, call=False)
    short_put_data, _ = fetch_option_data(api_key, short_put_ticker, eval_date)
    long_put_ticker = construct_option_ticker(underlying_symbol, expiration_date, long_put_strike, call=False)
    long_put_data, _ = fetch_option_data(api_key, long_put_ticker, eval_date)

    # 2. Plot the data if all three datasets are available
    if stock_data and short_put_data and long_put_data:
        plot_put_credit_spread(
            stock_data, short_put_data, long_put_data,
            underlying_symbol, short_put_strike, long_put_strike,
            start_time_str=plot_start_time,
            end_time_str=plot_end_time
        )
    else:
        logging.error("Insufficient data to plot. One or more required datasets are missing.")


if __name__ == "__main__":
    main()