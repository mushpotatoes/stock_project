import os
import datetime
import sqlite3
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import base64
import mimetypes
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from trading_helpers import get_git_repo_root

SCOPES = [
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send"
]

def gmail_create_draft_with_attachment(creds):
    """
    Create a draft email with attachment, then return the draft object.
    """
    try:
        # Build the Gmail API client
        service = build("gmail", "v1", credentials=creds)
        mime_message = EmailMessage()
        today = datetime.date.today()
        formatted_date = today.strftime("%Y-%m-%d")

        # Email headers
        mime_message["To"] = "wilkpaulc@gmail.com"
        mime_message["From"] = "wilkpaulc@gmail.com"
        mime_message["Subject"] = f"POI on {formatted_date}"

        # Email text
        mime_message.set_content(
            "Hello,\n\nThis is an automated mail with attachment.\nPlease do not reply."
        )

        # Identify the path and attach a file
        attachment_filename = f"plots/close_{formatted_date}.png"
        type_subtype, _ = mimetypes.guess_type(attachment_filename)
        maintype, subtype = type_subtype.split("/")

        with open(attachment_filename, "rb") as fp:
            attachment_data = fp.read()
        mime_message.add_attachment(
            attachment_data, 
            maintype, 
            subtype, 
            filename=f"poi_{formatted_date}.png"
        )

        # Encode in base64
        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        create_draft_request_body = {"message": {"raw": encoded_message}}

        # Create the draft in Gmail
        draft = (
            service.users()
                   .drafts()
                   .create(userId="me", body=create_draft_request_body)
                   .execute()
        )
        print(f"Draft created. ID: {draft['id']}")
        return draft

    except HttpError as error:
        print(f"An error occurred while creating the draft: {error}")
        return None

def gmail_send_draft(draft_id, creds):
    """
    Send an existing draft by its draft_id.
    """
    try:
        service = build("gmail", "v1", credentials=creds)
        sent_message = (
            service.users()
                   .drafts()
                   .send(userId="me", body={"id": draft_id})
                   .execute()
        )
        print(f"Draft with ID {draft_id} has been sent successfully.")
        return sent_message

    except HttpError as error:
        print(f"An error occurred while sending the draft: {error}")
        return None
    
###############################################################################
# Callback placeholders
###############################################################################
def on_interval_elapsed(df):
    """
    Called every N minutes (e.g., every 5 minutes).
    You can place code here for tasks you'd like to run at that interval:
      - Logging
      - Additional data analysis
      - Printing status
      - Etc.
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Interval event triggered.")


def on_new_local_minimum_found(min_idx, min_value, df):
    """
    Called each time a NEW local minimum is found (an index that wasn't found before).
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New local minimum at index={min_idx}, value={min_value}")
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    # 3. Create a draft with attachment
    draft = gmail_create_draft_with_attachment(creds)
    if draft is None:
        print("Failed to create draft.")
        return

    # 4. Send the draft
    draft_id = draft["id"]
    sent_msg = gmail_send_draft(draft_id, creds)
    if sent_msg:
        print("Email successfully sent.")
    else:
        print("Failed to send the email.")



###############################################################################
#                         1. DATA LOADING & FILTERING                         #
###############################################################################

def load_and_sort_stock_data(symbol, repo_root, use_yahoo=False, start_date=None, end_date=None):
    """
    Load stock data either from a local SQLite DB (default) or from Yahoo Finance
    if use_yahoo=True.  For Yahoo Finance, you can optionally provide date range
    strings (YYYY-MM-DD) for start_date/end_date.
    Returns a DataFrame with at least:
       ['epoch_time', 'open', 'high', 'low', 'close', 'volume', 'time']
    sorted by 'epoch_time'.
    """

    if use_yahoo:
        import yfinance as yf
        import datetime

        # If no date range is given, pick something reasonable
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')

        # Download daily data (use interval="1m" if you actually need intraday 
        # data and have yfinance privileges to do so)
        data = yf.download(symbol, start=start_date, end=end_date, interval="1m", prepost=True)

        if data.empty:
            print(f"No Yahoo Finance data retrieved for {symbol} from {start_date} to {end_date}.")
            return pd.DataFrame()

        # Reset index so 'Date' becomes a regular column
        data.reset_index(inplace=True)

        # Rename columns to align with existing usage
        data.rename(
            columns={
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',  # in case you want it
                'Volume': 'volume'
            },
            inplace=True
        )
        # print(data)
        # exit()

        # Convert date to epoch_time
        data['date'] = data['date'] - pd.Timedelta(hours=1)
        data['epoch_time'] = data['date'].apply(lambda dt: dt.timestamp())

        # Generate a 'time' column if downstream code expects it:
        # (for daily data, just use "00:00:00" or format the date if you like)
        data['time'] = data['date'].dt.strftime("%H:%M:%S")

        # Sort and re-index
        data.sort_values(by='epoch_time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    else:
        # Original LOCAL-SQLITE logic:
        import sqlite3
        import os

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
    """
    datetime_str = f"{date_str} {time_str}"
    dt_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return dt_object.timestamp()


def filter_epoch_range(df, start, end):
    """
    Filter rows of a DataFrame whose 'epoch_time' values lie between
    'start' and 'end' (inclusive).
    """
    mask = (df['epoch_time'] >= start) & (df['epoch_time'] <= end)
    filtered_df = df.loc[mask].sort_values(by='epoch_time').reset_index(drop=True)
    return filtered_df


###############################################################################
#                         2. UTILITY / ANALYSIS HELPERS                       #
###############################################################################

def find_last_below_threshold(df, threshold):
    """
    Return the integer index of the last row whose 'volume' is below
    the specified 'threshold', provided all subsequent rows meet volume >= threshold.
    If no crossing is found, return None.
    """
    n = len(df)
    # Iterate backward (excluding the very last row)
    for i in range(n - 2, -1, -1):
        if df.iloc[i]["volume"] < threshold:
            # Check that everything after i is >= threshold
            if (df.iloc[i + 1:]["volume"] >= threshold).all():
                return i
    return None


def slope_of_best_fit(values):
    """
    Compute the slope of a best-fit line for a 1D array of values,
    assuming x-coordinates [0, 1, 2, ..., len(values) - 1].
    """
    x = np.arange(len(values))
    y = values
    slope, _ = np.polyfit(x, y, 1)
    return slope


def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given Series.
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
    """
    minima_indices = []
    arr = series.to_numpy()

    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < threshold:
            minima_indices.append(i)

    return minima_indices

def find_local_maxima(series):
    """
    Return indices of points that are local maxima.
    A local max at i is defined by:
      series[i] > series[i-1] and series[i] > series[i+1]
    """
    maxima_indices = []
    arr = series.to_numpy()

    for i in range(1, len(arr) - 1):
        if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]):
            maxima_indices.append(i)
    return maxima_indices


###############################################################################
#                         3. PLOT CONFIG & PLOTTING                           #
###############################################################################

class PlotConfig:
    """
    Configuration object controlling which features appear in the plots.
    """
    def __init__(
        self,
        # Shared toggles:
        show_plots: bool = True,             # if False, no plot display at all
        save_plots: bool = False,            # if True, save to file instead of show
        output_dir: str = "./plots",

        # Volume plot toggles:
        plot_volume: bool = False,
        volume_threshold_lines: bool = True, # Show mean ± std lines

        # Close plot toggles:
        plot_close: bool = False,
        add_rsi: bool = True,
        add_moving_averages: bool = True,
        ma_windows = (15, 25, 100),
        restrict_830_to_300: bool = True,

        # Choose slope source
        slope_source: str = "MA_25",   # or "close", or "MA_15", etc.
        slope_lookback: int = 10,

        # More advanced features:
        highlight_negative_slope_runs: bool = True,
        highlight_below_ma_runs: bool = True,
        find_local_minima_in_slope: bool = False
    ):
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.output_dir = output_dir

        self.plot_volume = plot_volume
        self.volume_threshold_lines = volume_threshold_lines

        self.plot_close = plot_close
        self.add_rsi = add_rsi
        self.add_moving_averages = add_moving_averages
        self.ma_windows = list(ma_windows) if ma_windows else []

        self.restrict_830_to_300 = restrict_830_to_300
        self.slope_source = slope_source
        self.slope_lookback = slope_lookback
        self.highlight_negative_slope_runs = highlight_negative_slope_runs
        self.highlight_below_ma_runs = highlight_below_ma_runs
        self.find_local_minima_in_slope = find_local_minima_in_slope


def plot_volume_chart_for_day(day_df, date_str, config: PlotConfig):
    """
    Plot volume data for a single day, with optional threshold lines, crossing index,
    etc., based on the config toggles.
    """
    # For demonstration, always compute volume mean/std, but only
    # show them if config.volume_threshold_lines is True
    subset_df = day_df.iloc[50:330]  # subset used to compute mean & std
    volume_mean = subset_df['volume'].mean()
    volume_std = subset_df['volume'].std()

    # Identify crossing index
    crossing_idx = find_last_below_threshold(day_df.iloc[0:390], volume_mean + volume_std)

    if not config.show_plots and not config.save_plots:
        return crossing_idx

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)

    # Plot actual volume
    ax.plot(day_df['volume'], label='Volume', color='blue')

    # Horizontal lines
    if config.volume_threshold_lines:
        ax.axhline(y=volume_mean, color='red', linestyle='--', label=f'Mean={volume_mean:.2f}')
        ax.axhline(y=volume_mean + volume_std, color='red', linestyle='--', label='Mean+Std')

    # Vertical lines for day boundary
    ax.axvline(x=0, color='red', linestyle='--', label='Start (x=0)')
    ax.axvline(x=390, color='red', linestyle='--', label='End (x=390)')

    # Crossing index
    if crossing_idx is not None:
        ax.axvline(x=crossing_idx, color='green', linestyle='-', label=f'Cross @ x={crossing_idx}')

    ax.set_xlabel('Index')
    ax.set_ylabel('Volume')
    ax.set_title(f'Volume Chart - {date_str}')
    ax.legend()

    # Save or show
    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        fname = os.path.join(config.output_dir, f"volume_{date_str}.png")
        plt.savefig(fname)
        plt.close(fig)
    elif config.show_plots:
        plt.show()

    return crossing_idx


def plot_close_chart_for_day(day_df, date_str, config: PlotConfig):
    """
    Plot up to three subplots: (1) close + MAs, (2) RSI, (3) slope of last 10 MAs,
    with optional shading features, local minima, etc.
    """
    # Convert 'time' to a proper datetime so Matplotlib can do time-based plotting
    day_df['time'] = pd.to_datetime(day_df['time'], format="%H:%M:%S")

    # Compute MAs if requested
    if config.add_moving_averages and config.ma_windows:
        for window in config.ma_windows:
            ma_col = f"MA_{window}"
            day_df[ma_col] = day_df['close'].rolling(window).mean()
    # Ensure we have a 25-minute MA if we'll check slopes
    if 25 not in config.ma_windows:
        day_df['MA_25'] = day_df['close'].rolling(25).mean()

    # Compute RSI if requested
    if config.add_rsi:
        day_df['RSI_14'] = compute_rsi(day_df['close'], period=14)
    # Decide which series to use for slope
    slope_series = None
    if config.slope_source in day_df.columns:
        # e.g., "MA_25" or "MA_15" or "MA_100" ...
        slope_series = day_df[config.slope_source]
    elif config.slope_source.lower() == "close":
        slope_series = day_df['close']
    else:
        # If your code might default or raise an error
        print(f"Warning: slope_source '{config.slope_source}' not recognized; using 'close' by default.")
        slope_series = day_df['close']

    # Compute slope for chosen series, using the specified lookback
    day_df['chosen_slope'] = slope_series.rolling(config.slope_lookback).apply(
        slope_of_best_fit, raw=True
    )

    # Possibly restrict plotting range
    if config.restrict_830_to_300:
        start_plot_time = pd.to_datetime("08:30:00").time()
        end_plot_time   = pd.to_datetime("15:00:00").time()
        mask_plot = (day_df['time'].dt.time >= start_plot_time) & (day_df['time'].dt.time <= end_plot_time)
        day_df_plot = day_df.loc[mask_plot].copy().reset_index(drop=True)
    else:
        day_df_plot = day_df.copy()

    delay_before_trade=0
    delay_before_close_pos=15
    # Just an example: find local minima between 09:00 and 13:30
    start_local = pd.to_datetime("9:00:00").time()
    end_local   = pd.to_datetime("12:00:00").time()
    mask_local = (day_df['time'].dt.time >= start_local) & (day_df['time'].dt.time <= end_local)
    df_local = day_df.loc[mask_local].copy().reset_index(drop=True)
    local_min_indices = find_local_minima(df_local['chosen_slope'], threshold=-0.05)

    range_to_inspect_after = 15 # minutes
    minimum_drop_period = 0
    if len(local_min_indices) > 0:
        first_idx = min(local_min_indices)
        if (first_idx >= 120) or (first_idx < 30):
            return
        
        if (df_local.loc[first_idx-minimum_drop_period:first_idx, "chosen_slope"] > 0).any():
            return

        trade_idx = first_idx + delay_before_trade
        # print(first_idx, trade_idx, len(df_local))

        try:
            stats = [
                        df_local['date'].iloc[first_idx],
                        df_local['time'].iloc[first_idx].strftime('%H:%M'),
                        df_local['chosen_slope'].iloc[first_idx],
                        df_local['close'].iloc[trade_idx],
                        np.average(df_local['close'].iloc[trade_idx+1:trade_idx+range_to_inspect_after])
            ]
            print(stats)
        except:
            print(df_local)
        # exit()
        return stats
    else:
        return

    if not config.show_plots and not config.save_plots:
        return  # We won't do any actual plotting

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(16, 12))
    fig.suptitle(f"Close + RSI + MA_25 Slope - {date_str}", fontsize=14)

    # 1) Top subplot: Close + MAs
    ax1.plot(day_df_plot['time'], day_df_plot['close'], label='Close', color='blue')
    if config.add_moving_averages and config.ma_windows:
        for window in config.ma_windows:
            ax1.plot(day_df_plot['time'], day_df_plot[f"MA_{window}"], label=f"MA_{window}")

    ax1.set_ylabel('Close')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # 2) Middle subplot: RSI
    if config.add_rsi:
        ax2.plot(day_df_plot['time'], day_df_plot['RSI_14'], color='purple', label='RSI (14)')
        ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
        ax2.set_ylabel('RSI')
        ax2.legend(loc='upper left')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, "RSI Disabled", ha='center', va='center', fontsize=12)
        ax2.set_ylabel('RSI')
        ax2.grid(True)

    # 3) Bottom subplot: chosen slope
    ax3.plot(day_df_plot['time'], day_df_plot['chosen_slope'], label='Chosen Slope', color='orange')
    ax3.set_ylabel('Slope')
    ax3.set_xlabel('Time of Day')
    ax3.grid(True)
    ax3.legend(loc='upper left')

    # Highlight runs of negative slope if enabled
    if config.highlight_negative_slope_runs:
        neg_mask = (day_df_plot['chosen_slope'] < 0)
        start_idx = None
        runs = []

        for i in range(len(neg_mask)):
            if neg_mask[i] and start_idx is None:
                start_idx = i
            elif not neg_mask[i] and start_idx is not None:
                run_length = i - start_idx
                if run_length > 30:
                    runs.append((start_idx, i - 1))
                start_idx = None
        # Edge case: if it extends to the end
        if start_idx is not None:
            run_length = len(neg_mask) - start_idx
            if run_length > 30:
                runs.append((start_idx, len(neg_mask) - 1))

        # Shade the runs
        for (start, end) in runs:
            # shift by 30 if you want the run to start 30 bars later, etc.
            start_time = day_df_plot.iloc[start]['time']
            end_time   = day_df_plot.iloc[end]['time']
            for ax in (ax1, ax2, ax3):
                ax.axvspan(start_time, end_time, color='lightgray', alpha=0.3)

    # Highlight runs where MA_25 is below some other MA if enabled
    # For demonstration, let's highlight where MA_25 < MA_100.
    if config.highlight_below_ma_runs and ("MA_100" in day_df_plot.columns):
        below_mask = (day_df_plot['MA_100'] - day_df_plot['MA_25']) > 0
        start_idx = None
        runs = []

        for i in range(len(below_mask)):
            if below_mask[i] and start_idx is None:
                start_idx = i
            elif not below_mask[i] and start_idx is not None:
                run_length = i - start_idx
                if run_length > 30:
                    runs.append((start_idx, i - 1))
                start_idx = None
        # Edge case: if it extends to the end
        if start_idx is not None:
            run_length = len(below_mask) - start_idx
            if run_length > 30:
                runs.append((start_idx, len(below_mask) - 1))

        for (start, end) in runs:
            start_time = day_df_plot.iloc[start]['time']
            end_time   = day_df_plot.iloc[end]['time']
            for ax in (ax1, ax2, ax3):
                ax.axvspan(start_time, end_time, color='lightgreen', alpha=0.3)

    # Mark local minima in slope if requested
    if config.find_local_minima_in_slope:
        # Just an example: find local minima between 09:00 and 13:30
        start_local = pd.to_datetime("09:30:00").time()
        end_local   = pd.to_datetime("13:30:00").time()
        mask_local = (day_df['time'].dt.time >= start_local) & (day_df['time'].dt.time <= end_local)
        df_local = day_df.loc[mask_local].copy().reset_index(drop=True)
        local_min_indices = find_local_minima(df_local['chosen_slope'], threshold=-0.12)

        # for index in local_min_indices:
        #     print(f"{index}: {day_df['chosen_slope'].iloc[index]}")
        # exit()
        for i_min in local_min_indices:
            t_min = df_local.loc[i_min, 'time']
            for ax in (ax1, ax2, ax3):
                ax.axvline(x=t_min, color='red', linestyle='--', alpha=0.7)

    # Format the x-axis as times
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()

    plt.tight_layout()

    # Save or show
    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        fname = os.path.join(config.output_dir, f"close_{date_str}.png")
        plt.savefig(fname)
        plt.close(fig)
    elif config.show_plots:
        plt.show()



###############################################################################
#                         4. STATISTICS-COLLECTION                            #
###############################################################################
def compute_daily_stats(
    day_df, 
    date_str, 
    slope_source='close',  
    slope_lookback=10,
    slope_diff_threshold=0.025
):
    """
    Returns stats for a single trading day, including:
      - daily_min_slope of the chosen slope source
      - local min -> next local max intervals (in minutes) 
        (only if the max slope is at least slope_diff_threshold
         higher than the slope at the local min)
    """

    if not pd.api.types.is_datetime64_any_dtype(day_df['time']):
        day_df['time'] = pd.to_datetime(day_df['time'], format="%H:%M:%S")

    # Ensure we have the slope_source column
    if slope_source not in day_df.columns:
        if slope_source == 'close':
            pass
        else:
            raise ValueError(f"Column '{slope_source}' not found in DataFrame.")

    # Compute slope over the chosen series
    from numpy import polyfit, arange
    def slope_of_best_fit(values):
        x = arange(len(values))
        y = values
        slope, _ = polyfit(x, y, 1)
        return slope

    series_for_slope = day_df[slope_source]
    day_df['chosen_slope'] = series_for_slope.rolling(slope_lookback).apply(slope_of_best_fit, raw=True)
    daily_min_slope = day_df['chosen_slope'].min(skipna=True)

    # Simple local minima / maxima detectors
    def find_local_minima(series, threshold=None):
        minima_indices = []
        arr = series.to_numpy()
        for i in range(1, len(arr) - 1):
            if arr[i] < arr[i-1] and arr[i] < arr[i+1]:
                if threshold is None or arr[i] < threshold:
                    minima_indices.append(i)
        return minima_indices

    def find_local_maxima(series, threshold=None):
        maxima_indices = []
        arr = series.to_numpy()
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                # maxima_indices.append(i)
                if threshold is None or arr[i] > threshold:
                    maxima_indices.append(i)
        return maxima_indices

    # Find local minima below 0.0 slope, for example
    local_min_indices = find_local_minima(day_df['chosen_slope'], threshold=-0.1)
    
    maxima_all = find_local_maxima(day_df['chosen_slope'], threshold=0.03)
    max_set = set(maxima_all)

    time_intervals = []
    for min_idx in local_min_indices:
        slope_min = day_df['chosen_slope'].iloc[min_idx]

        # Find the *first* local max after min_idx
        # whose slope is at least slope_diff_threshold above slope_min
        next_max_idx = None
        for forward_i in range(min_idx + 1, len(day_df) - 1):
            if forward_i in max_set:
                slope_max = day_df['chosen_slope'].iloc[forward_i]
                if slope_max - slope_min >= slope_diff_threshold:
                    next_max_idx = forward_i
                    break

        # If found, measure the time difference in minutes
        if next_max_idx is not None:
            t_min = day_df.loc[min_idx, 'time']
            t_max = day_df.loc[next_max_idx, 'time']
            delta_minutes = (t_max - t_min).total_seconds() / 60.0
            time_intervals.append(delta_minutes)

    # If no intervals, average is None
    avg_interval = np.mean(time_intervals) if time_intervals else None

    stats = {
        "date": date_str,
        "daily_min_slope": daily_min_slope,
        "min_to_max_intervals": time_intervals,
        "avg_min_to_max_interval": avg_interval
    }
    return stats



###############################################################################
#                         5. MASTER ANALYSIS FUNCTION                         #
###############################################################################

def run_analysis(
    symbol="SPY",
    start_date="2024-01-01",
    end_date="2024-01-31",
    plot_config=None,
    collect_stats: bool = True,
    use_yahoo: bool = False
):
    """
    Demonstrates iterating over a range of dates, filtering data for each day,
    optionally plotting with PlotConfig, and optionally collecting daily statistics.
    """
    if plot_config is None:
        # Default config: no volume, only close
        plot_config = PlotConfig(plot_close=True)

    repo_root = get_git_repo_root()
    if not repo_root:
        print("Not inside a Git repository. Exiting.")
        return

    df = load_and_sort_stock_data(symbol, repo_root, use_yahoo=use_yahoo, start_date=start_date, end_date=end_date)
    ret_df = df.copy(deep=True)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_stats = []

    collection = []
    for single_date in date_range:
        date_str = single_date.strftime('%Y-%m-%d')

        if plot_config.plot_volume:
            # Normal trading hours
            start_epoch = int(convert_to_epoch(date_str, "08:30:00"))
            end_epoch   = int(convert_to_epoch(date_str, "16:00:00"))
        else:
            # Full day if we are doing 'close' or no volume
            start_epoch = int(convert_to_epoch(date_str, "00:00:00"))
            end_epoch   = int(convert_to_epoch(date_str, "23:59:59"))

        # Filter data
        selected_df = filter_epoch_range(df, start_epoch, end_epoch)
        # print(selected_df)
        if selected_df.empty:
            continue

        # Plot volume if requested
        if plot_config.plot_volume:
            _ = plot_volume_chart_for_day(selected_df, date_str, plot_config)

        # Plot close if requested
        if plot_config.plot_close:
            close_stats = plot_close_chart_for_day(selected_df, date_str, plot_config)
            if close_stats is not None:
                collection.append(close_stats)

        # # Collect stats if desired
        # if collect_stats:
        #     day_stats = compute_daily_stats(selected_df, date_str)
        #     all_stats.append(day_stats)

    # print(collection)
    # print(len(collection))
    # print(len(collection[0]))
    # print(collection)
    cs_df = pd.DataFrame(collection, columns=['Date', 'Time', 'Slope', 'Close', 'NextAverage'])
    print(cs_df)
    col1 = 'NextAverage'
    col2 = 'Close'
    bins = 20
    cs_df['Difference'] = cs_df[col1] - cs_df[col2]  # Calculate the difference
    plt.hist(cs_df['Difference'], bins=bins)
    plt.xlabel(f'{col1} - {col2}')  # Dynamic x-axis label
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {col1} - {col2}')
    plt.grid(True)
    plt.show()
    negative_diff = cs_df[cs_df['Difference'] < 0]
    positive_diff = cs_df[cs_df['Difference'] >= 0]  # Including 0 for completeness

    # 2. Calculate statistics for negative Difference
    def calculate_stats(df, column_name):
        """Calculates statistics for a given column in a DataFrame."""
        mean = df[column_name].mean()
        median = df[column_name].median()
        try:  # Handle potential error if no mode can be found.
            mode = df[column_name].mode()[0] # mode() can return multiple values, we only need one.
        except IndexError:
            mode = "No unique mode" # Handle cases with no mode.
        std_dev = df[column_name].std()
        return mean, median, mode, std_dev

    neg_mean, neg_median, neg_mode, neg_std = calculate_stats(negative_diff, 'Slope')
    pos_mean, pos_median, pos_mode, pos_std = calculate_stats(positive_diff, 'Slope')


    # 3. Print the results
    print("Statistics for Negative Difference (Slope):")
    print(f"  Mean: {neg_mean}")
    print(f"  Median: {neg_median}")
    print(f"  Mode: {neg_mode}")
    print(f"  Standard Deviation: {neg_std}")
    print(f"  Count: {len(negative_diff)}")

    print("\nStatistics for Positive or Zero Difference (Slope):")
    print(f"  Mean: {pos_mean}")
    print(f"  Median: {pos_median}")
    print(f"  Mode: {pos_mode}")
    print(f"  Standard Deviation: {pos_std}")
    print(f"  Count: {len(positive_diff)}")


    # 4. (Optional) Visual Comparison: Box Plots
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed
    plt.boxplot([negative_diff['Slope'], positive_diff['Slope']], labels=['Negative Difference', 'Positive/Zero Difference'])
    plt.ylabel('Slope')
    plt.title('Comparison of Slope Distributions')
    plt.show()

    # 5. (Optional) Visual Comparison: Histograms
    plt.figure(figsize=(8, 6))
    plt.hist(negative_diff['Slope'], alpha=0.5, label='Negative Difference', bins=bins)  # alpha for transparency
    plt.hist(positive_diff['Slope'], alpha=0.5, label='Positive/Zero Difference', bins=bins)
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Comparison of Slope Distributions')
    plt.legend()
    plt.show()

    
    exit()
    
    # Convert stats to DataFrame for further use
    stats_df = pd.DataFrame(all_stats)
    # print(selected_df)
    return ret_df


###############################################################################
# Core loop that runs every minute
###############################################################################
def start_monitoring(symbol="SPY", minutes_interval=5, config: PlotConfig = None):
    """
    Example function that (1) fetches or analyzes data every minute,
    (2) triggers on_interval_elapsed() every 'minutes_interval' minutes,
    (3) triggers on_new_local_minimum_found() if a new local min appears.

    You may want to incorporate your own data-fetch routines here:
       - either from your local database or from Yahoo
       - perhaps only fetch the most recent minute's data
    """

    iteration_count = 0
    known_minima = set()  # Keep track of local-min indices found so far
    current_minima = 0
    while True:
        iteration_count += 1

        # ----------------------------------------------------
        # 1) Gather or update your data for analysis
        #    (Below is just placeholder logic.)
        # ----------------------------------------------------
        #
        # In your real code, you might call:
        #     new_df = load_and_sort_stock_data(symbol, repo_root, use_yahoo=True, ...)
        # or you might just append the latest minute’s data to a running DataFrame.
        #
        # For demonstration, assume we have some df we can pass to a function:
        
        # You can adjust the date range to multiple months or years
        df = run_analysis(
            symbol="SPY",
            start_date="2025-01-30",
            end_date="2025-02-01",
            plot_config=config,
            collect_stats=True,
            use_yahoo = True
        )
        # print(df)

        # # Fake DataFrame with random 'close' values for this demonstration
        # # Replace with your real data or partial updates each minute
        # data_length = 100
        # df = pd.DataFrame({
        #     "close": np.random.randn(data_length).cumsum()  # random walk
        # })

        # ----------------------------------------------------
        # 2) Analyze the data to find local minima
        # ----------------------------------------------------
        # This re-uses your existing local-min detection logic:
        #   find_local_minima() from your code expects:
        #   def find_local_minima(series, threshold=-0.01): ...
        slope_series = None
        if config.slope_source in df.columns:
            # e.g., "MA_25" or "MA_15" or "MA_100" ...
            slope_series = df[config.slope_source]
        elif config.slope_source.lower() == "close":
            slope_series = df['close']
        else:
            # If your code might default or raise an error
            print(f"Warning: slope_source '{config.slope_source}' not recognized; using 'close' by default.")
            slope_series = df['close']
        df['chosen_slope'] = slope_series.rolling(config.slope_lookback).apply(
            slope_of_best_fit, raw=True
        )
        
        local_min_indices = find_local_minima(df['chosen_slope'], threshold=-0.1)
        # local_min_indices = []
        # arr = df["close"].to_numpy()
        # for i in range(1, len(arr) - 1):
        #     if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
        #         local_min_indices.append(i)
        # print(local_min_indices)

        # Convert to a set for easier comparison
        current_idx = max(local_min_indices)
        current_time = df["epoch_time"].iloc[max(local_min_indices)]
        # print(df["epoch_time"].iloc[max(local_min_indices)])
        # exit()
        if (len(local_min_indices) > 0) and (current_minima != current_time):
            on_new_local_minimum_found(current_idx, df["close"].iloc[current_idx], df)
            current_minima = current_time

        # # Find newly discovered minima
        # new_minima = current_minima - known_minima
        # if new_minima:
        #     for nm in new_minima:
        #         on_new_local_minimum_found(nm, df["close"].iloc[nm], df)
        #     # Update our global record of known minima
        #     known_minima.update(new_minima)

        # ----------------------------------------------------
        # 3) Trigger event if we've hit the interval
        # ----------------------------------------------------
        if iteration_count % minutes_interval == 0:
            on_interval_elapsed(df)

        # ----------------------------------------------------
        # 4) Sleep until next minute
        # ----------------------------------------------------
        time.sleep(60)  # Sleep for 60 seconds

        # NOTE: This loop will run indefinitely.
        # Press Ctrl+C (or otherwise terminate) to stop.
        
###############################################################################
#                             6. SCRIPT ENTRY POINT                            #
###############################################################################

if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    # Example usage:
    #   1) Basic config: just close, no volume
    config = PlotConfig(
        plot_volume=False,
        plot_close=True,
        add_rsi=True,
        add_moving_averages=True,
        ma_windows=(15, 25, 100),
        restrict_830_to_300=True,
        highlight_negative_slope_runs=True,
        highlight_below_ma_runs=True,
        find_local_minima_in_slope=True,
        save_plots=True,
        slope_source='close',
        slope_lookback=10
    )

    df = run_analysis(
        symbol="SPY",
        start_date="2020-01-01",
        end_date="2025-02-01",
        plot_config=config,
        collect_stats=True,
        use_yahoo = False
    )
    # start_monitoring(config=config)

    # # stats now holds a DataFrame with columns like ["date", "daily_min_slope"]
    # print(stats)
