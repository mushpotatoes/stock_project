import os
import datetime
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading_helpers import get_git_repo_root


def load_and_sort_stock_data(symbol, repo_root):
    """
    Load stock data from an SQLite database into a Pandas DataFrame,
    then sort it by 'epoch_time'.

    Parameters
    ----------
    symbol : str
        The stock ticker symbol (e.g., 'SPY').
    repo_root : str
        The root directory path of the Git repository.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all tables within the DB for the given symbol,
        sorted by 'epoch_time'.
    """
    db_path = os.path.join(repo_root, f"{symbol}_data.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names in the SQLite database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    # Load each table into a list of DataFrames
    df_list = []
    for table_name in table_names:
        query = f"SELECT * FROM {table_name}"
        df_list.append(pd.read_sql_query(query, conn))

    conn.close()

    # Concatenate all DataFrames and sort by 'epoch_time'
    df = pd.concat(df_list, ignore_index=True)
    df.sort_values(by='epoch_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def convert_to_epoch(date_str, time_str):
    """
    Convert a date string plus a time string into Unix epoch time.

    Parameters
    ----------
    date_str : str
        Date in "YYYY-MM-DD" format.
    time_str : str
        Time in "HH:MM:SS" format.

    Returns
    -------
    float
        The Unix epoch time corresponding to the input date/time.
    """
    datetime_str = f"{date_str} {time_str}"
    dt_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return dt_object.timestamp()


def filter_epoch_range(df, start, end):
    """
    Filter rows of a DataFrame whose 'epoch_time' values lie between
    'start' and 'end' (inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an 'epoch_time' column.
    start : float
        Starting epoch time (inclusive).
    end : float
        Ending epoch time (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with rows in ascending order of 'epoch_time'.
    """
    mask = (df['epoch_time'] >= start) & (df['epoch_time'] <= end)
    filtered_df = df.loc[mask].sort_values(by='epoch_time').reset_index(drop=True)
    return filtered_df


def find_last_below_threshold(df, threshold):
    """
    Return the integer index of the last row whose 'volume' is below
    the specified 'threshold', with the constraint that all subsequent rows
    have 'volume' >= threshold. If no crossing is found, return None.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'volume' column.
    threshold : float
        Threshold value for 'volume'.

    Returns
    -------
    int or None
        The index of the last row below threshold, or None if no such crossing exists.
    """
    n = len(df)
    # Iterate backward (excluding the very last row)
    for i in range(n - 2, -1, -1):
        if df.iloc[i]["volume"] < threshold:
            # Check that everything after i is >= threshold
            if (df.iloc[i + 1:]["volume"] >= threshold).all():
                return i
    return None


def plot_volume_chart_for_day(day_df, date_str):
    """
    Plot the volume chart for a single day's DataFrame, focusing on normal trading hours.

    * Retains vertical lines at x=0 and x=390
    * Labels the crossing index where volume rises above threshold
    * Draws horizontal lines at mean and (mean + std) of volume

    Parameters
    ----------
    day_df : pd.DataFrame
        DataFrame for a specific day, with a 'volume' column.
    date_str : str
        The date string (YYYY-MM-DD), used as the chart title.

    Returns
    -------
    int or None
        The crossing index if found, otherwise None.
    """

    # Consider the first 390 rows (08:30 - 16:00) as normal trading hours
    subset_df = day_df.iloc[50:330]  # subset used to compute mean & std
    volume_mean = subset_df['volume'].mean()
    volume_std = subset_df['volume'].std()

    # Find crossing within the first 390 rows
    last_cross = find_last_below_threshold(day_df.iloc[0:390], volume_mean + volume_std)
    if last_cross is None:
        return None

    # Plot
    plt.figure(figsize=(16, 10))

    # Actual volume
    plt.plot(day_df['volume'], label='Volume', color='blue')

    # Mean and mean+std horizontal lines
    plt.axhline(y=volume_mean, color='red', linestyle='--', label=f'Mean = {volume_mean:.2f}')
    plt.axhline(y=volume_mean + volume_std, color='red', linestyle='--', label='Mean + 1 Std')

    # Vertical lines at x=0, x=390, and at crossing
    plt.axvline(x=0, color='red', linestyle='--', label='Start (x=0)')
    plt.axvline(x=390, color='red', linestyle='--', label='End (x=390)')
    plt.axvline(x=last_cross, color='green', linestyle='-', label=f'Crossing (x={last_cross})')

    plt.xlabel('Index')
    plt.ylabel('Volume')
    plt.title(f'Volume Chart - {date_str}')
    plt.legend()
    plt.show()

    return last_cross


def slope_of_best_fit(values):
    """
    Compute the slope of the best-fit line for a 1D array of values,
    assuming x-coordinates [0, 1, 2, ..., len(values) - 1].

    Parameters
    ----------
    values : array-like
        The data over which to compute a linear best-fit.

    Returns
    -------
    float
        The slope of the best-fit line.
    """
    x = np.arange(len(values))
    y = values
    slope, intercept = np.polyfit(x, y, 1)
    return slope


def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given Series.

    Parameters
    ----------
    series : pd.Series
        The price series (e.g., close prices).
    period : int
        The period over which to compute the RSI, default is 14.

    Returns
    -------
    pd.Series
        A Series containing RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def find_local_minima(series, threshold=-0.01):
    """
    Identify indices where 'series' has a local minimum below `threshold`.

    A local minimum is defined by:
      series[i] < series[i - 1]
      series[i] < series[i + 1]
      series[i] < threshold

    Parameters
    ----------
    series : pd.Series
        The data to examine for local minima.
    threshold : float
        The cutoff below which a point must be to be considered a local min.

    Returns
    -------
    list of int
        Indices of local minima meeting the threshold criterion.
    """
    minima_indices = []
    arr = series.to_numpy()

    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < threshold:
            minima_indices.append(i)

    return minima_indices


def plot_close_chart_for_day(day_df, date_str, ma_windows=None, plot_8_30_to_3=False, should_find_local_minima = False):
    """
    Create three subplots for a single day:
      1) Close prices + optional moving averages
      2) 14-period RSI
      3) Slope of the last 10 values of the 25-minute MA

    Additionally, it:
      - Marks local minima in the MA_25 slope between 9:00 AM and 1:30 PM.
      - Prints min, mean, and mode of the slope in that window.

    Parameters
    ----------
    day_df : pd.DataFrame
        Must contain 'time' (HH:MM:SS) and 'close' columns.
    date_str : str
        A string representing the day for chart labeling (YYYY-MM-DD).
    ma_windows : list of int, optional
        Window sizes for moving averages to compute and plot. E.g., [25].
    plot_8_30_to_3 : bool, optional
        If True, restrict plotted range to 08:30 - 15:00; else plot entire day.
    """
    # Convert 'time' to a proper datetime so Matplotlib can do time-based plotting
    day_df['time'] = pd.to_datetime(day_df['time'], format="%H:%M:%S")

    # Compute requested MAs
    if ma_windows:
        for window in ma_windows:
            day_df[f"MA_{window}"] = day_df['close'].rolling(window).mean()
    else:
        ma_windows = []

    # Ensure we have a 25-minute MA, used for slope calculations
    if 25 not in ma_windows:
        day_df['MA_25'] = day_df['close'].rolling(25).mean()

    # Compute 14-period RSI
    day_df['RSI_14'] = compute_rsi(day_df['close'], period=14)

    # Compute slope of the last 10 values of the 25-minute MA
    day_df['MA_25_slope'] = day_df['MA_25'].rolling(10).apply(slope_of_best_fit, raw=True)

    # Filter data for plotting, if requested
    if plot_8_30_to_3:
        start_plot_time = pd.to_datetime("08:30:00").time()
        end_plot_time = pd.to_datetime("15:00:00").time()
        mask_plot = (day_df['time'].dt.time >= start_plot_time) & (day_df['time'].dt.time <= end_plot_time)
        day_df_plot = day_df.loc[mask_plot].copy().reset_index(drop=True)
    else:
        day_df_plot = day_df.copy()

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    fig.suptitle(f"Close + RSI + MA_25 Slope - {date_str}", fontsize=14)

    # --- Top subplot: Close + MAs ---
    ax1.plot(day_df_plot['time'], day_df_plot['close'], label='Close', color='blue')
    for window in ma_windows:
        ax1.plot(day_df_plot['time'], day_df_plot[f"MA_{window}"], label=f"MA_{window}")
    ax1.set_ylabel('Close')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # --- Middle subplot: RSI ---
    ax2.plot(day_df_plot['time'], day_df_plot['RSI_14'], color='purple', label='RSI (14)')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
    ax2.set_ylabel('RSI')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    # --- Bottom subplot: MA_25 Slope ---
    ax3.plot(day_df_plot['time'], day_df_plot['MA_25_slope'], label='MA_25 Slope', color='orange')
    ax3.set_ylabel('Slope')
    ax3.set_xlabel('Time of Day')
    ax3.grid(True)
    ax3.legend(loc='upper left')

    # --- Identify consecutive runs of negative slope (below 0) ---
    neg_mask = (day_df_plot['MA_25_slope'] < 0) #& (day_df_plot['MA_25_slope'] > -0.03)
    start_idx = None
    runs = []

    for i in range(len(neg_mask)):
        if neg_mask[i] and start_idx is None:
            # We’re starting a new negative run
            start_idx = i
            # print(f"Found start {start_idx} {day_df_plot.iloc[start_idx]['time']}")
        elif not neg_mask[i] and start_idx is not None:
            # We just ended a negative run at i-1
            run_length = i - start_idx
            # print(f"Found end {i} {day_df_plot.iloc[i]['time']}; len = {run_length}")
            if (run_length > 30):
                runs.append((start_idx, i - 1))
            start_idx = None

    # If a negative run continues till the very end of the data
    if start_idx is not None:
        run_length = len(neg_mask) - start_idx
        if run_length > 30:
            runs.append((start_idx, len(neg_mask) - 1))

    # --- Shade each run on all subplots (ax1, ax2, ax3) ---
    for (start, end) in runs:
        start_time = day_df_plot.iloc[start + 30]['time']
        end_time   = day_df_plot.iloc[end]['time']

        # axvspan adds a shaded vertical rectangle from start_time to end_time
        ax1.axvspan(start_time, end_time, color='lightgray', alpha=0.3)
        ax2.axvspan(start_time, end_time, color='lightgray', alpha=0.3)
        ax3.axvspan(start_time, end_time, color='lightgray', alpha=0.3)

    # --- Identify consecutive runs of 25 below 100 ---
    below_mask = (day_df_plot['MA_100'] - day_df_plot['MA_25']) > 0
    start_idx = None
    runs = []

    for i in range(len(below_mask)):
        if below_mask[i] and start_idx is None:
            # We’re starting a new negative run
            start_idx = i
            # print(f"Found start {start_idx} {day_df_plot.iloc[start_idx]['time']}")
        elif not below_mask[i] and start_idx is not None:
            # We just ended a negative run at i-1
            run_length = i - start_idx
            # print(f"Found end {i} {day_df_plot.iloc[i]['time']}; len = {run_length}")
            if run_length > 30:
                runs.append((start_idx, i - 1))
            start_idx = None

    # If a negative run continues till the very end of the data
    if start_idx is not None:
        run_length = len(below_mask) - start_idx
        if run_length > 30:
            runs.append((start_idx, len(below_mask) - 1))

    # --- Shade each run on all subplots (ax1, ax2, ax3) ---
    for (start, end) in runs:
        start_time = day_df_plot.iloc[start + 30]['time']
        end_time   = day_df_plot.iloc[end]['time']

        # axvspan adds a shaded vertical rectangle from start_time to end_time
        ax1.axvspan(start_time, end_time, color='lightgreen', alpha=0.3)
        ax2.axvspan(start_time, end_time, color='lightgreen', alpha=0.3)
        ax3.axvspan(start_time, end_time, color='lightgreen', alpha=0.3)


    # (1) and (2): find runs in overlap_mask
    overlap_mask = neg_mask & below_mask
    start_idx = None
    runs_overlap = []

    for i in range(len(overlap_mask)):
        if overlap_mask[i] and start_idx is None:
            start_idx = i
            print(f"Found ov start {start_idx} {day_df_plot.iloc[start_idx]['time']}")
        elif not overlap_mask[i] and start_idx is not None:
            run_length = i - start_idx
            print(f"Found ov end {i} {day_df_plot.iloc[i]['time']}; len = {run_length}")
            if run_length > 30:
                runs_overlap.append((start_idx + 30, i - 1))
            start_idx = None

    # Edge case: still "in a run" at the end
    if start_idx is not None:
        run_length = len(overlap_mask) - start_idx
        if run_length > 30:
            runs_overlap.append((start_idx + 30, len(overlap_mask) - 1))

    # print(runs_overlap)
    for run_idx, (start, end) in enumerate(runs_overlap):
        if(day_df_plot.iloc[start:end]['MA_25_slope'].min() > -0.03):
            local_min_indices = find_local_minima(day_df_plot.iloc[start:end]['MA_25_slope'], threshold=0)
            
            if len(local_min_indices) > 0:
                first_idx = local_min_indices[0]
                t_min = day_df_plot.iloc[start + first_idx]['time']
                ax1.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)
                ax2.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)
                ax3.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)
        else:
            print(f"Max drop rate was {day_df_plot.iloc[start:end]['MA_25_slope'].min()} for {day_df_plot.iloc[start]['time']}")
        


    start_time_localmin = pd.to_datetime("09:00:00").time()
    end_time_localmin = pd.to_datetime("13:30:00").time()
    if should_find_local_minima:
        # Mark local minima in slope < -0.01 between 9:00 and 13:30
        mask_localmin = (
            (day_df['time'].dt.time >= start_time_localmin) &
            (day_df['time'].dt.time <= end_time_localmin)
        )
        df_localmin = day_df.loc[mask_localmin].copy().reset_index(drop=True)
        local_min_indices = find_local_minima(df_localmin['MA_25_slope'], threshold=-0.01)

        for i_min in local_min_indices:
            t_min = df_localmin.loc[i_min, 'time']
            ax1.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)
            ax2.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)
            ax3.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)

    # Set up hour-based tick labels
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()

    # Compute min, mean, mode of slope from 9:00 to 13:30
    mask_stats = (
        (day_df['time'].dt.time >= start_time_localmin) &
        (day_df['time'].dt.time <= end_time_localmin)
    )
    slopes_filtered = day_df.loc[mask_stats, 'MA_25_slope'].dropna()
    if not slopes_filtered.empty:
        min_slope = slopes_filtered.min()
        mean_slope = slopes_filtered.mean()
        mode_values = slopes_filtered.mode()
        mode_slope = mode_values.iloc[0] if not mode_values.empty else None

        print("\n--- MA_25 Slope Statistics between 09:00 and 13:30 ---")
        print(f"Min slope:  {min_slope:.5f}")
        print(f"Mean slope: {mean_slope:.5f}")
        print(f"Mode slope: {mode_slope:.5f}" if mode_slope is not None else "Mode slope: None")
    else:
        print("\nNo slope data available between 09:00 and 13:30.")
    # exit()


def main(plot_type='volume'):
    """
    Main function to load data and plot either:
      - 'volume' charts (for normal trading hours), or
      - 'close' charts (for the entire calendar day).

    Parameters
    ----------
    plot_type : str
        Either 'volume' or 'close'. Defaults to 'volume'.
    """
    symbol = 'SPY'
    repo_root = get_git_repo_root()

    if not repo_root:
        print("Not inside a Git repository. Exiting.")
        return

    df = load_and_sort_stock_data(symbol, repo_root)
    crossing_indices = []

    # Example loop for days in January (adjust as needed)
    for day in range(1, 32):
        date_str = f"2024-02-{day:02d}"

        if plot_type == 'volume':
            # Normal trading hours
            start_epoch = int(convert_to_epoch(date_str, "08:30:00"))
            end_epoch = int(convert_to_epoch(date_str, "16:00:00"))
        elif plot_type == 'close':
            # Full day
            start_epoch = int(convert_to_epoch(date_str, "00:00:00"))
            end_epoch = int(convert_to_epoch(date_str, "23:59:59"))
        else:
            print("Unrecognized plot_type. Use 'volume' or 'close'.")
            return

        # Filter data for the day
        selected_df = filter_epoch_range(df, start_epoch, end_epoch)
        if selected_df.empty:
            continue

        # Generate the plots
        if plot_type == 'volume':
            last_cross = plot_volume_chart_for_day(selected_df, date_str)
            if last_cross is not None:
                crossing_indices.append(last_cross)
        elif plot_type == 'close':
            plot_close_chart_for_day(
                selected_df,
                date_str,
                ma_windows=[15, 25, 100],
                plot_8_30_to_3=True
            )

    # If we did volume plots, show histogram of crossing indices
    if plot_type == 'volume' and crossing_indices:
        plt.figure(figsize=(7, 5))
        plt.hist(crossing_indices, bins=20, color='blue', edgecolor='black')
        plt.xlabel('Index of Last Crossing')
        plt.ylabel('Frequency')
        plt.title('Distribution of Last Crossing Points (Volume)')
        plt.show()


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    # Choose which plot to generate:
    #   'volume' -> restricted to normal trading hours
    #   'close'  -> entire day
    # main(plot_type='volume')
    main(plot_type='close')
