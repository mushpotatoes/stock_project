import os
import requests
import logging
import matplotlib.pyplot as plt
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def fetch_stock_data(api_key, symbol, date, limit=5000, next_url=None):
    """
    Fetches aggregated stock data for a given day using Polygon.io API.

    Args:
        api_key (str): Polygon.io API key.
        symbol (str): Stock ticker symbol.
        date (str): Date in YYYY-MM-DD format.
        limit (int, optional): Maximum number of results per request.
        next_url (str, optional): URL for pagination if available.
    
    Returns:
        tuple: (list of aggregated data points, next_url if available)
    """
    if next_url:
        url = next_url + f"&apiKey={api_key}"
    else:
        # Using minute-level aggregates for the specified day.
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{date}/{date}?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
        )

    logging.info(f"Fetching stock data from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        if "results" in response_json:
            return response_json["results"], response_json.get("next_url")
        else:
            logging.error("Response JSON did not contain 'results'.")
            return [], None
    else:
        logging.error(f"Error fetching stock data: HTTP {response.status_code}")
        return [], None


def fetch_option_data(api_key, option_ticker, date, multiplier=1, timespan='minute', limit=5000, next_url=None):
    """
    Fetches aggregated option data for a given day using Polygon.io API.

    Args:
        api_key (str): Your Polygon.io API key.
        option_ticker (str): Option contract ticker (e.g., "O:SPY250228C00450000").
        date (str): Date for which to fetch data in YYYY-MM-DD format.
        multiplier (int, optional): Aggregation multiplier (default: 1).
        timespan (str, optional): Aggregation timespan (default: 'minute').
        limit (int, optional): Maximum number of results per request (default: 5000).
        next_url (str, optional): URL for pagination if available.

    Returns:
        tuple: (list of aggregated data points, next_url if available)
    """
    if next_url:
        url = next_url + f"&apiKey={api_key}"
    else:
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/"
            f"{multiplier}/{timespan}/{date}/{date}?limit={limit}&apiKey={api_key}"
        )

    logging.info(f"Fetching option data from URL: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        response_json = response.json()
        if "results" in response_json:
            return response_json["results"], response_json.get("next_url")
        else:
            logging.error("Response JSON did not contain 'results'.")
            return [], None
    else:
        logging.error(f"Error fetching option data: HTTP {response.status_code}")
        return [], None


def construct_option_ticker(underlying, expiration, strike, call=True):
    """
    Constructs a Polygon option ticker string.

    Polygon option tickers follow the format:
        O:{UNDERLYING}{EXPIRATION}{C/P}{STRIKE}
    Where:
      - {UNDERLYING} is the underlying symbol (e.g., SPY)
      - {EXPIRATION} is the expiration date in YYMMDD format.
      - {C/P} is 'C' for call options or 'P' for put options.
      - {STRIKE} is the strike price multiplied by 1000 and zero-padded to 8 digits.

    Args:
        underlying (str): Underlying stock symbol.
        expiration (str): Expiration date in YYYY-MM-DD format.
        strike (float): Strike price.
        call (bool, optional): True for a call option, False for a put option.

    Returns:
        str: Constructed option ticker (e.g., "O:SPY250228C00450000").
    """
    try:
        year, month, day = expiration.split("-")
    except ValueError:
        raise ValueError("Expiration date must be in YYYY-MM-DD format")
    exp_formatted = f"{year[2:]}{month}{day}"
    option_type = "C" if call else "P"
    # Multiply strike by 1000 and pad to 8 digits.
    strike_int = int(round(strike * 1000))
    strike_formatted = f"{strike_int:08d}"

    option_ticker = f"O:{underlying}{exp_formatted}{option_type}{strike_formatted}"
    return option_ticker


def plot_data(stock_data, option_data):
    """
    Plots the close price data for the stock and option in separate subplots.

    Args:
        stock_data (list): List of aggregated data points for the stock.
        option_data (list): List of aggregated data points for the option.
    """
    # Convert timestamps (in milliseconds) to datetime objects and extract close prices.
    stock_times = [datetime.fromtimestamp(item["t"] / 1000) for item in stock_data]
    stock_closes = [item["c"] for item in stock_data]

    option_times = [datetime.fromtimestamp(item["t"] / 1000) for item in option_data]
    option_closes = [item["c"] for item in option_data]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot for SPY close data.
    axes[0].plot(stock_times, stock_closes, label='SPY Close', color='blue')
    axes[0].set_title('SPY Close Prices')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True)

    # Subplot for Option close data.
    axes[1].plot(option_times, option_closes, label='Option Close', color='green')
    axes[1].set_title('Option Close Prices')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Price')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    # Retrieve your Polygon.io API key from environment variables.
    api_key = os.environ.get('API_KEY')
    if not api_key:
        raise EnvironmentError("API_KEY environment variable not found.")

    # --- User-defined parameters ---
    underlying_symbol = "SPY"         # Underlying asset symbol.
    eval_date = "2025-07-22"            # Date for which to retrieve the data.
    strike_price = 628.0              # Desired strike price.
    expiration_date = eval_date      # Expiration date of the option contract.
    # --------------------------------

    # Fetch SPY stock data for the specified day.
    stock_results, _ = fetch_stock_data(api_key, underlying_symbol, eval_date)
    if not stock_results:
        logging.warning("No stock data returned for SPY.")
    else:
        logging.info("Fetched SPY stock data successfully.")

    # Construct the option ticker for the call option.
    option_ticker = construct_option_ticker(underlying_symbol, expiration_date, strike_price, call=True)
    logging.info(f"Constructed option ticker: {option_ticker}")

    # Fetch option data for the specified day.
    option_results, _ = fetch_option_data(api_key, option_ticker, eval_date)
    if not option_results:
        logging.warning("No option data returned for the specified option contract.")
    else:
        logging.info("Fetched option data successfully.")

    # Plot the data if both datasets are available.
    if stock_results and option_results:
        plot_data(stock_results, option_results)
    else:
        logging.error("Insufficient data to plot. Ensure both SPY and option data are available.")


if __name__ == "__main__":
    main()
