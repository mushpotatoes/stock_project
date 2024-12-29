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

def main():
    symbol = 'SPY'
    repo_root = get_git_repo_root()
    if not repo_root:
        print("Not inside a Git repository.")
        exit()

    df = load_and_sort_stock_data(symbol, repo_root)

    crossing_indices = []
    for i in range(1,13):
        for i in range(1,31):
            inital_time = int(convert_to_epoch(f"2024-12-{i:02d}", "8:30:00"))
            end_time = int(convert_to_epoch(f"2024-12-{i:02d}", "16:00:00"))
            # print(inital_time, end_time)
            selected_df = filter_epoch_range(df, inital_time, end_time)
            if selected_df.empty:
                continue
            # print(selected_df.head())
            # print(selected_df.tail())
            subset_df = selected_df.iloc[50:330]
            volume_mean = subset_df['volume'].mean()
            volume_std = subset_df['volume'].std()
            # print(subset_df['volume'])
            # print(f"Mean: {volume_mean}, STD: {volume_std}")
            # print(subset_df.head())
            # exit()
            last_cross = find_last_below_threshold(selected_df[0:390], volume_mean+volume_std)
            if last_cross is not None:
                crossing_indices.append(last_cross)
            else:
                continue

            # print(f"Last crossing point is {last_cross} minutes.")
            plt.figure(figsize=(16, 10))
            plt.plot(selected_df['volume'], label='Volume', color='blue')
            plt.axhline(y=volume_mean, color='red', linestyle='--', label=f'Average = {volume_mean:.2f}')
            plt.axhline(y=volume_mean+volume_std, color='red', linestyle='--', label=f'1 Std')
            plt.axvline(x=0, color='red', linestyle='--', label='Vertical line at x=0')
            plt.axvline(x=390, color='red', linestyle='--', label='Vertical line at x=390')
            plt.axvline(x=last_cross, color='green', linestyle='-', label=f'Vertical line at x={last_cross}')
            plt.xlabel('Index')
            plt.ylabel('Volume')
            plt.title('Volume Chart')
            plt.legend()
            plt.show()
    
    # Once the loop is done, plot a histogram of crossing_indices
    plt.figure(figsize=(7, 5))
    plt.hist(crossing_indices, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Index of Last Crossing')
    plt.ylabel('Frequency')
    plt.title('Distribution of Last Crossing Points')
    plt.show()

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    main()
