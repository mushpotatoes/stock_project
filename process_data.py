import os
import sqlite3
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from trading_helpers import get_git_repo_root

FULL_DAY_MINS = 300
NUM_MINUTES_INFER = 40
NUM_MINUTES_IN_PRED = 20
LAST_INDEX_PRED = NUM_MINUTES_INFER + NUM_MINUTES_IN_PRED

def load_and_sort_stock_data(symbol, repo_root, start_date=None):
    """Loads stock data from the SQLite database into a Pandas DataFrame,
       sorts it by epoch_time, and optionally filters out rows before start_date.
    """
    conn = sqlite3.connect(os.path.join(repo_root, f"{symbol}_data.db"))
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
    
    # Filter by start_date if provided
    if start_date is not None:
        # Convert start_date to epoch
        start_epoch = convert_to_epoch(start_date, "00:00:00")
        df = df[df['epoch_time'] >= start_epoch]

    return df

def filter_epoch_range(df, start, end):
    """
    Filters a dataframe to return rows where 'epoch_time'
    is between 'start' and 'end' (inclusive).
    """
    ret_df = df[((df['epoch_time'] >= start) & (df['epoch_time'] <= end))]
    ret_df = ret_df.sort_values(by='epoch_time').reset_index(drop=True)
    return ret_df

def convert_to_epoch(date_str, time_str):
    """
    Takes two strings:
      date_str in the format "YYYY-MM-DD"
      time_str in the format "HH:MM:SS"
    and returns the corresponding Unix epoch time.
    """
    datetime_str = date_str + " " + time_str
    dt_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return dt_object.timestamp()

def calculate_rsi(df, window=14):
    """
    Calculates the RSI (Relative Strength Index) based on the 'close' column.
    Uses an Exponential Moving Average approach for gains/losses.
    Returns a Pandas Series with the RSI values.
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    alpha = 1 / window
    avg_gain = gain.ewm(alpha=alpha, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    symbol = 'SPY'
    repo_root = get_git_repo_root()
    if not repo_root:
        print("Not inside a Git repository.")
        exit()

    # 1) Load data and calculate indicators
    df = load_and_sort_stock_data(symbol, repo_root, start_date="2020-07-01")
    df['rsi'] = calculate_rsi(df, window=14)
    df['sma_15'] = df['close'].rolling(window=15).mean()
    df['sma_25'] = df['close'].rolling(window=25).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    
    # 2) Create a day-only column (YYYY-MM-DD)
    df['date_only'] = df['epoch_time'].apply(
        lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d')
    )

    # ------------------------------------------------
    # PART A: Collect 40-sample windows for each day.
    # ------------------------------------------------
    # We'll store each day's array in a list; each element is (5, 40)
    daily_tensors = []
    target_tensors = []
    num_days_above_threshold = 0
    threshold = 1.8

    # Get all unique trading dates in sorted order
    all_dates = df['date_only'].unique()
    all_dates = sorted(all_dates)
    leave = False
    min_vals = [float("inf"), float("inf"), float("inf"), float("inf")]
    for date_str in tqdm(all_dates, desc="Processing days", unit="day"):
        # if date_str in ["2020-02-27"]:
        #     continue

        # print(f"processing {date_str}")
        start_time = convert_to_epoch(date_str, "09:00:00")
        end_time   = convert_to_epoch(date_str, "14:00:00")
        day_slice  = filter_epoch_range(df, start_time, end_time)
        
        # Focus on the 5 columns of interest and drop NaN rows
        day_slice  = day_slice[['close', 'rsi', 'sma_15', 'sma_25', 'sma_100']].dropna()
        day_slice['rsi'] = day_slice['rsi'] / 100
        
        total_minutes = (end_time - start_time) / 60
        num_eval_points = int(total_minutes - LAST_INDEX_PRED)
        # print(total_minutes)
        # print(num_eval_points)
        # exit()
        # print(len(day_slice))
        count = 0
        if len(day_slice) >= FULL_DAY_MINS:
            for initial_idx in range(num_eval_points):
                count = count + 1
                subslice_start = initial_idx
                subslice_end = NUM_MINUTES_INFER + initial_idx
                predict_end = LAST_INDEX_PRED + initial_idx
                # Use the first 40
                # day_subslice = day_slice.iloc[subslice_start:subslice_end]
                day_subslice = day_slice.iloc[subslice_start:subslice_end].copy(deep=True).reset_index(drop=True)
                day_slice_targets = day_slice['close'].iloc[subslice_end:predict_end].to_numpy()
                # day_subslice['sma_15'] = (day_subslice['sma_15'] / day_subslice['close'].iloc[0] - 1) * 1000
                # day_subslice['sma_25'] = (day_subslice['sma_25'] / day_subslice['close'].iloc[0] - 1) * 1000
                # day_subslice['sma_100'] = (day_subslice['sma_100'] / day_subslice['close'].iloc[0] - 1) * 1000
                # day_subslice['close'] = (day_subslice['close'] / day_subslice['close'].iloc[0] - 1) * 1000
                first_close = day_subslice['close'].iloc[0]
                # print(f"\nWhile processing {date_str}")
                # print(f"Processing {subslice_start} to {subslice_end} and {predict_end}")
                # print(first_close)
                # # print(day_subslice)
                # # exit()
                # print(day_subslice.loc[:15], 'sma_15')
                # print((day_subslice.loc[:, 'sma_15'] / first_close) - 1)
                # if count > 5:
                #     exit()
                data_offset = 0
                day_subslice.loc[:, 'sma_15'] = (
                    (day_subslice.loc[:, 'sma_15'] / first_close) - 1
                ) * 100 + data_offset
                if min_vals[0] > day_subslice.loc[:, 'sma_15'].min():
                    min_vals[0] = day_subslice.loc[:, 'sma_15'].min()
                day_subslice.loc[:, 'sma_25'] = (
                    (day_subslice.loc[:, 'sma_25'] / first_close) - 1
                ) * 100 + data_offset
                if min_vals[1] > day_subslice.loc[:, 'sma_25'].min():
                    min_vals[1] = day_subslice.loc[:, 'sma_25'].min()
                day_subslice.loc[:, 'sma_100'] = (
                    (day_subslice.loc[:, 'sma_100'] / first_close) - 1
                ) * 100 + data_offset
                if min_vals[2] > day_subslice.loc[:, 'sma_100'].min():
                    min_vals[2] = day_subslice.loc[:, 'sma_100'].min()
                day_subslice.loc[:, 'close'] = (
                    (day_subslice.loc[:, 'close'] / first_close) - 1
                ) * 100 + data_offset
                if min_vals[3] > day_subslice.loc[:, 'close'].min():
                    min_vals[3] = day_subslice.loc[:, 'close'].min()

                # print(day_slice_targets)
                # print(day_subslice)
                # non_numeric_df = day_subslice.select_dtypes(exclude=['number'])
                # print(non_numeric_df)

                any_abs_greater_than_one = (day_subslice.abs() > threshold).any().any()
                # print(day_subslice)
                # print(any_abs_greater_than_one)
                # exit()
                if any_abs_greater_than_one:
                    num_days_above_threshold = num_days_above_threshold + 1
                    # print(f"\nWhile processing {date_str}")
                    # print("Dataframe contains value > 1")
                    # print(day_subslice)
                    # exit()
                    # leave = True
                    # break

                # the magic number 1 is a minute lag until purchase
                minutes_offset = 1
                subarray = day_slice_targets[minutes_offset:]
                max_idx = np.argmax(subarray)
                min_idx = np.argmin(subarray)

                max_change = subarray[max_idx] / day_slice_targets[0]
                min_change = subarray[min_idx] / day_slice_targets[0]
                # print(f"Max: {1 - max_change} at {max_idx + minutes_offset} minutes")
                # print(f"Min: {1 - min_change} at {min_idx + minutes_offset} minutes")
                if (np.abs(1 - max_change) > np.abs(1 - min_change)):
                    # print("greater max")
                    # print([1 - max_change, (max_idx + minutes_offset)/60])
                    target_tensors.append([((1 - max_change)*100) + data_offset, (max_idx + minutes_offset)/60])
                else:
                    # print("greater min")
                    # print([1 - min_change, (min_idx + minutes_offset)/60])
                    target_tensors.append([((1 - min_change)*100) + data_offset, (min_idx + minutes_offset)/60])
                # exit()
                
                # Convert to numpy, shape becomes (40, 5). Then transpose for (5, 40).
                subslice_tensor = day_subslice.to_numpy().T  # shape: (5, 40)
                # print(subslice_tensor)
                # exit()
                # Append to our list
                daily_tensors.append(subslice_tensor)
        # if leave:
        #     break
    print(f"Min values are: {min_vals}")
    print(f"{num_days_above_threshold} with values above {threshold}")
    # Now daily_tensors is a list of arrays, each shape = (5, 40).
    # You could convert to a single 3D NumPy array, if desired, by:
    daily_tensors = np.array(daily_tensors)  # shape = (num_days, 5, 40)
    target_tensors = np.array(target_tensors)  # shape = (num_days, 5, 40)

    # print(f"Collected {len(daily_tensors)} daily tensors with shape (5, {NUM_MINUTES_INFER}).")
    # print(daily_tensors)
    # print(f"Collected {len(target_tensors)} daily tensors with shape {target_tensors[0].shape}.")
    # print(target_tensors)

    filename = "daily_tensors.npy"
    np.save(filename, daily_tensors)

    # loaded_tensors = np.load(filename)
    # print(loaded_tensors.shape)

    filename = "target_tensors.npy"
    np.save(filename, target_tensors)

    # loaded_tensors = np.load(filename)
    # print(loaded_tensors.shape)

    # ------------------------------------------------
    # PART B: Example plot for a single day (unchanged).
    # ------------------------------------------------
    # You can keep your plotting logic as needed.
    # start_time = int(convert_to_epoch("2020-02-20", "08:30:00"))
    # end_time    = int(convert_to_epoch("2020-02-20", "16:00:00"))
    selected_df = filter_epoch_range(df, start_time, end_time)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    # --- Plot 1: Close + SMAs ---
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
