import os
import sqlite3
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from trading_helpers import get_git_repo_root

def load_and_sort_stock_data(symbol, repo_root):
    """Loads stock data from the SQLite database into a Pandas DataFrame and sorts it by date.

    Args:
        symbol (str): Stock ticker symbol.
        repo_root (str): Root directory of the Git repository.

    Returns:
        pd.DataFrame: DataFrame containing the sorted stock data.
    """

    conn = sqlite3.connect(os.path.join(repo_root, f"{symbol}_data.db"))

    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    df_list = []
    for table_name in table_names:
        query = f"SELECT * FROM {table_name}"
        df_list.append(pd.read_sql_query(query, conn))

    conn.close()

    # Concatenate all DataFrames and sort
    df = pd.concat(df_list, ignore_index=True)
    df = df.sort_values(by='epoch_time').reset_index(drop=True) 
    return df

def get_range(from_time, to_time, df):
    return

def convert_to_epoch(date_str, time_str):
    """
    Takes two strings:
      date_str in the format "YYYY-MM-DD"
      time_str in the format "HH:MM:SS"
    and returns the corresponding Unix epoch time.
    """
    # Combine the date and time strings
    datetime_str = date_str + " " + time_str
    
    # Parse them into a datetime object
    dt_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    
    # Convert the datetime object to epoch (float)
    return dt_object.timestamp()

def filter_epoch_range(df, start, end):
    """
    Filters a dataframe to return rows where 'epoch_time'
    is between start and end (inclusive).

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing an 'epoch_time' column.
    start : float or int
        The lower bound of epoch time (in seconds).
    end : float or int
        The upper bound of epoch time (in seconds).

    Returns:
    --------
    pd.DataFrame
        Rows where df['epoch_time'] >= start and df['epoch_time'] <= end.
    """
    ret_df = df[((df['epoch_time'] >= start) & (df['epoch_time'] <= end))]
    ret_df = ret_df.sort_values(by='epoch_time').reset_index(drop=True)
    return ret_df
    # return df[(df['epoch_time'] >= start)]
    # return df['epoch_time'].between(start, end)

def find_last_below_threshold(df, threshold):
    """
    Returns the integer index of the last row whose 'volume' is below 'threshold',
    provided that every row after that index has 'volume' >= threshold.

    If no such crossing point is found, returns None.

    Assumes the DataFrame has a simple RangeIndex (0, 1, 2, ...).
    """
    n = len(df)

    # We iterate from the second-to-last row backward to row 0
    # (since we compare i+1 in the check)
    for i in range(n - 2, -1, -1):
        if df.iloc[i]["volume"] < threshold:
            # Check if from row i+1 to the end, volume is always >= threshold
            if (df.iloc[i + 1 :]["volume"] >= threshold).all():
                return i

    return None

def calculate_rsi(df, window=14):
    """
    Calculates the RSI (Relative Strength Index) based on the 'close' column.
    Uses an Exponential Moving Average approach for gains/losses.
    Returns a Pandas Series with the RSI values.
    """
    # 1. Calculate the difference (delta) between consecutive close prices
    delta = df['close'].diff()

    # 2. Separate gains and losses
    gain = delta.where(delta > 0, 0)  # positive deltas
    loss = -delta.where(delta < 0, 0) # negative deltas (multiplied by -1)

    # 3. Calculate Exponential Moving Averages for gains and losses
    alpha = 1 / window
    avg_gain = gain.ewm(alpha=alpha, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=window).mean()

    # 4. Compute the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # 5. Compute the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi

def main():
    symbol = 'SPY'
    repo_root = get_git_repo_root()
    if not repo_root:
        print("Not inside a Git repository.")
        exit()

    df = load_and_sort_stock_data(symbol, repo_root)
    df['rsi'] = calculate_rsi(df, window=14)
    df['sma_15'] = df['close'].rolling(window=15).mean()
    df['sma_25'] = df['close'].rolling(window=25).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    print(df)


    inital_time = int(convert_to_epoch(f"2024-12-26", "8:30:00"))
    end_time = int(convert_to_epoch(f"2024-12-26", "16:00:00"))
    # print(inital_time, end_time)
    selected_df = filter_epoch_range(df, inital_time, end_time)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # --- Plot 1: Close ---
    ax1.plot(selected_df.index, selected_df["close"], color='blue')
    ax1.plot(selected_df.index, selected_df['sma_15'], label='SMA 15', color='orange')
    ax1.plot(selected_df.index, selected_df['sma_25'], label='SMA 25', color='red')
    ax1.plot(selected_df.index, selected_df['sma_100'], label='SMA 100', color='green')
    ax1.set_ylabel("Close Price")
    ax1.set_title("Close vs. RSI")

    # --- Plot 2: RSI ---
    ax2.plot(selected_df.index, selected_df["rsi"], color='red')
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Index (Time)")
    ax2.axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
    ax2.axhline(y=70, color='orange', linestyle='--', label='Overbought (70)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    main()
