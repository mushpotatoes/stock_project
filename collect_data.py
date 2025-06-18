import os
import time
import logging
import requests
import sqlite3
from trading_helpers import get_git_repo_root
import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(api_key, symbol, start_date, end_date, next_url=None, limit=5000):
    """Fetches stock data from Polygon.io API.

    Args:
        api_key (str): Polygon.io API key.
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        next_url (str, optional): URL for the next page of results. Defaults to None.
        limit (int, optional): Number of results to return per request. Defaults to 5000.

    Returns:
        list: List of dictionaries, each representing a data point.
    """

    if next_url:
        url = next_url
        url += f"&adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
    else:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
        url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"

    # logging.debug(f"Getting URL: {url}")
    response = requests.get(url)
    response_json = response.json()

    if response.status_code == 200 and "results" in response_json:
        return response_json["results"], response_json.get("next_url")
    else:
        logging.error(f"Error fetching data: {response.status_code}")
        return [], None

def store_stock_data(data, symbol, repo_root):
    """Stores stock data in a SQLite database, with each year in a separate table.

    Args:
        data (list): List of dictionaries, each representing a data point.
        symbol (str): Stock ticker symbol.
        repo_root (str): Root directory of the Git repository.
    """

    conn = sqlite3.connect(os.path.join(repo_root, f"big_{symbol}_data.db"))
    cursor = conn.cursor()

    for row in data:
        epoch_time = row['t'] / 1000
        dt_object = datetime.datetime.fromtimestamp(epoch_time)
        year = dt_object.year  # Extract the year
        date_str = dt_object.strftime('%Y-%m-%d')
        time_str = dt_object.strftime('%H:%M:%S')

        table_name = f"{symbol}_prices_{year}"

        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                epoch_time INTEGER PRIMARY KEY,
                date TEXT,
                time TEXT,
                volume REAL,
                v_weight_price REAL,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                tx_cnt INTEGER
            )
        """
        cursor.execute(create_table_sql)

        insert_sql = f"""
            INSERT OR IGNORE INTO {table_name} (epoch_time, date, time, volume, v_weight_price, open, close, high, low, tx_cnt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_sql, (epoch_time, date_str, time_str, row['v'], row['vw'], row['o'], row['c'], row['h'], row['l'], row['n']))

    conn.commit()
    conn.close()

def main():
    api_key = os.environ.get('API_KEY')
    symbol = 'SPY'
    # symbol = 'DIA'
    # symbol = 'QQQ'
    # symbol = 'IWM' # Small Cap
    # symbol = 'MDY' # Small Cap
    # for symbol in ['DIA', 'QQQ', 'MDY', 'IWM']:
    for symbol in ['SPY']:
        start_date = '2025-02-13'
        today = datetime.date.today()
        # end_date = today.strftime("%Y-%m-%d") + datetime.timedelta(days=1)
        # end_date = '2025-01-28'
        delta = datetime.timedelta(days=14)

        repo_root = get_git_repo_root()
        if not repo_root:
            logging.error("Not inside a Git repository.")
            exit()
        current_date = datetime.date(2025, 6, 10)
        while current_date <= datetime.date(today.year, today.month, today.day):
            start_date = current_date
            end_date = min(current_date + delta, datetime.date(today.year, today.month, today.day) + datetime.timedelta(days=1))
            next_url = None

            logging.debug(f"Getting data for start date: {start_date}")
            while True:
                data, next_url = fetch_stock_data(api_key, symbol, start_date.strftime('%Y-%m-%d'), end_date, next_url)
                if not data:
                    break

                store_stock_data(data, symbol, repo_root)
                if not next_url:
                    break
                time.sleep(1)
            current_date += delta + datetime.timedelta(days=1)
            time.sleep(1)

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    main()
    