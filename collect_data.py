import os
import time
import logging
import pandas as pd
import requests
import sqlite3
from trading_helpers import get_git_repo_root

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(api_key, symbol, start_date, end_date):
    """Fetches stock data from Polygon.io API.

    Args:
        api_key (str): Polygon.io API key.
        symbol (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        list: List of dictionaries, each representing a data point.
    """

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{start_date}/{end_date}"
    url += f"?adjusted=false&sort=asc&limit=50000&apiKey={api_key}"

    response = requests.get(url)
    response_json = response.json()

    if response.status_code == 200 and "results" in response_json:
        return response_json["results"]
    else:
        logging.error(f"Error fetching data: {response.status_code}")
        return []

def store_stock_data(data, symbol, repo_root):
    """Stores stock data in a SQLite database.

    Args:
        data (list): List of dictionaries, each representing a data point.
        symbol (str): Stock ticker symbol.
        repo_root (str): Root directory of the Git repository.
    """

    conn = sqlite3.connect(os.path.join(repo_root, f"{symbol}_data.db"))
    cursor = conn.cursor()

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {symbol}_prices (
            epoch_time INTEGER PRIMARY KEY,
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
        INSERT OR REPLACE INTO {symbol}_prices (epoch_time, volume, v_weight_price, open, close, high, low, tx_cnt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    cursor.executemany(insert_sql, [(row['t'] / 1000, row['v'], row['vw'], row['o'], row['c'], row['h'], row['l'], row['n']) for row in data])

    conn.commit()
    conn.close()

if __name__ == "__main__":
    repo_root = get_git_repo_root()
    if not repo_root:
        logging.error("Not inside a Git repository.")
        exit()

    api_key = os.environ.get('API_KEY')
    symbol = 'SPY'
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    logging.debug(f"API KEY: {api_key}")
    logging.debug(f"Repo Root: {repo_root}")
    # data = fetch_stock_data(api_key, symbol, start_date, end_date)
    # store_stock_data(data, symbol, repo_root)