import os
import datetime
import pickle
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
import warnings

import pytz
import requests

warnings.filterwarnings("ignore", category=DeprecationWarning)


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

def gmail_create_draft_with_attachment(creds, body="Hello,\n\nThis is an automated mail with attachment.\nPlease do not reply."):
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
        mime_message.set_content(body)

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
    creds = None
    # flow = InstalledAppFlow.from_client_secrets_file(
    #     "credentials.json", SCOPES
    # )
    # creds = flow.run_local_server(port=0)
    # with open("token.json", "w") as token:
    #     token.write(creds.to_json())
    # exit()

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
    draft = gmail_create_draft_with_attachment(creds, "Triggered by interval")
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


def on_new_local_minimum_found(min_idx, min_value, df):
    """
    Called each time a NEW local minimum is found (an index that wasn't found before).
    """
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New local minimum at index={min_idx} [{df["time"].iloc[min_idx]}], value={min_value}")
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
    draft = gmail_create_draft_with_attachment(creds, "Triggered by minimum slope")
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


def get_recent_purple_start_in_green(day_df_plot, predicted_classes):
    """
    Returns the timestamp of the beginning of the most recent purple section 
    (i.e. a segment where predicted_classes==1 for at least 3 consecutive points)
    that occurs inside of a green section (i.e. where MA_25 is below MA_100 for 
    more than 30 consecutive points, with a 30-point offset).
    
    If no such purple section is found, returns None.
    """
    # --- Identify Purple Runs ---
    purple_runs = []
    n = len(predicted_classes)
    i = 0
    while i < n:
        if predicted_classes[i] == 1:
            start = i
            while i < n and predicted_classes[i] == 1:
                i += 1
            end = i - 1
            if (end - start + 1) >= 3:
                purple_runs.append((start, end))
        else:
            i += 1

    # Get the starting timestamp for each purple run
    purple_start_times = []
    for start, _ in purple_runs:
        # Get the timestamp at which this purple section starts.
        p_time = day_df_plot.iloc[start]['time']
        purple_start_times.append(p_time)

    # --- Identify Green Runs ---
    # Here we assume that a green run is defined using:
    # below_mask = (MA_100 - MA_25) > 0, and only runs longer than 30 points are considered.
    below_mask = (day_df_plot['MA_100'] - day_df_plot['MA_25']) > 0
    green_runs = []
    start_idx = None
    for i in range(len(below_mask)):
        if below_mask[i] and start_idx is None:
            start_idx = i
        elif not below_mask[i] and start_idx is not None:
            run_length = i - start_idx
            if run_length > 30:
                # Note the green overlay starts at start_idx+30, per your plotting code.
                green_start = day_df_plot.iloc[start_idx + 30]['time']
                green_end = day_df_plot.iloc[i - 1]['time']
                green_runs.append((green_start, green_end))
            start_idx = None
    # Check if the green run extends to the end of the DataFrame.
    if start_idx is not None:
        run_length = len(below_mask) - start_idx
        if run_length > 30:
            green_start = day_df_plot.iloc[start_idx + 30]['time']
            green_end = day_df_plot.iloc[len(below_mask) - 1]['time']
            green_runs.append((green_start, green_end))

    # --- Find Purple Sections That Occur Inside a Green Section ---
    purple_inside_green = []
    for p_time in purple_start_times:
        for g_start, g_end in green_runs:
            if g_start <= p_time <= g_end:
                purple_inside_green.append(p_time)
                break  # once found inside a green section, no need to check further

    if not purple_inside_green:
        return None

    # --- Return the Most Recent Purple Section Start Timestamp ---
    # "Most recent" is the one with the maximum (latest) timestamp.
    most_recent_timestamp = max(purple_inside_green)
    return most_recent_timestamp
import pandas as pd  # ensure pandas is imported

def get_recent_purple_start_longer_than_5_minutes(day_df_plot, predicted_classes):
    """
    Returns the timestamp at which the most recent purple section (i.e. a consecutive
    segment where predicted_classes == 1) that lasts longer than 5 minutes begins.
    
    A purple section's duration is determined by subtracting the timestamp at its start
    from the timestamp at its end (as found in day_df_plot['time']).
    
    If no purple section longer than 5 minutes is found, returns None.
    """
    
    # --- Identify Purple Runs and Filter by Duration ---
    purple_runs = []  # to hold tuples of (start_index, end_index)
    n = len(predicted_classes)
    i = 0
    
    while i < n:
        if predicted_classes[i] == 1:
            # Mark the start of a purple run.
            start = i
            while i < n and predicted_classes[i] == 1:
                i += 1
            # The run ends at the previous index.
            end = i - 1
            
            # Retrieve start and end timestamps.
            start_time = day_df_plot.iloc[start]['time']
            end_time = day_df_plot.iloc[end]['time']
            
            # Check if the run lasts longer than 5 minutes.
            if end_time - start_time > pd.Timedelta(minutes=5):
                purple_runs.append((start, end))
        else:
            i += 1

    if not purple_runs:
        return None

    # --- Retrieve the Start Times of the Qualified Purple Runs ---
    purple_start_times = [day_df_plot.iloc[start]['time'] for start, _ in purple_runs]
    
    # --- Return the Most Recent (Latest) Start Time ---
    most_recent_timestamp = max(purple_start_times)
    return most_recent_timestamp

###############################################################################
#                         1. DATA LOADING & FILTERING                         #
###############################################################################
central = pytz.timezone('US/Central')

# def load_and_sort_stock_data(symbol, repo_root, use_yahoo=False, start_date=None, end_date=None):
#     """
#     Load stock data either from a local SQLite DB (default) or from Yahoo Finance
#     if use_yahoo=True.  For Yahoo Finance, you can optionally provide date range
#     strings (YYYY-MM-DD) for start_date/end_date.
#     Returns a DataFrame with at least:
#        ['epoch_time', 'open', 'high', 'low', 'close', 'volume', 'time']
#     sorted by 'epoch_time'.
#     """

#     if use_yahoo:
#         import yfinance as yf
#         import datetime

#         # If no date range is given, pick something reasonable
#         if start_date is None:
#             start_date = "2020-01-01"
#         if end_date is None:
#             end_date = datetime.datetime.today().strftime('%Y-%m-%d')

#         # Download daily data (use interval="1m" if you actually need intraday 
#         # data and have yfinance privileges to do so)
#         try:
#             data = yf.download(symbol, start=start_date, end=end_date, interval="1m", prepost=True)
#         except:
#             print("Failed to download data")
#             return None

#         if data.empty:
#             print(f"No Yahoo Finance data retrieved for {symbol} from {start_date} to {end_date}.")
#             return pd.DataFrame()

#         # Reset index so 'Date' becomes a regular column
#         data.reset_index(inplace=True)

#         # Rename columns to align with existing usage
#         data.rename(
#             columns={
#                 'Datetime': 'date',
#                 'Open': 'open',
#                 'High': 'high',
#                 'Low': 'low',
#                 'Close': 'close',
#                 'Adj Close': 'adj_close',  # in case you want it
#                 'Volume': 'volume'
#             },
#             inplace=True
#         )
#         # print(data)
#         # exit()

#         # Convert date to epoch_time
#         data['date'] = data['date'] - pd.Timedelta(hours=1)
#         data['epoch_time'] = data['date'].apply(lambda dt: dt.timestamp())

#         # Generate a 'time' column if downstream code expects it:
#         # (for daily data, just use "00:00:00" or format the date if you like)
#         data['time'] = data['date'].dt.strftime("%H:%M:%S")
#         data['close'] = data['close'].round(2)


#         # Sort and re-index
#         data.sort_values(by='epoch_time', inplace=True)
#         data.reset_index(drop=True, inplace=True)
#         return data

#     else:
#         # Original LOCAL-SQLITE logic:
#         import sqlite3
#         import os

#         db_path = os.path.join(repo_root, f"{symbol}_data.db")
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # Get all table names in the SQLite database
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         table_names = [row[0] for row in cursor.fetchall()]

#         # Load each table into a list of DataFrames
#         df_list = []
#         for table_name in table_names:
#             query = f"SELECT * FROM {table_name}"
#             df_list.append(pd.read_sql_query(query, conn))

#         conn.close()

#         # Concatenate all DataFrames and sort by 'epoch_time'
#         df = pd.concat(df_list, ignore_index=True)
#         df.sort_values(by='epoch_time', inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         return df

def load_and_sort_stock_data(symbol, repo_root, use_yahoo=False, start_date=None, end_date=None):
    """
    Load stock data either from Yahoo Finance (if use_yahoo=True) or from the Polygon API by default.
    For Yahoo Finance, you can optionally provide date range strings (YYYY-MM-DD) for start_date/end_date.
    
    Returns a DataFrame with at least:
       ['epoch_time', 'open', 'high', 'low', 'close', 'volume', 'time']
    sorted by 'epoch_time'.
    """
    if use_yahoo:
        import yfinance as yf

        # If no date range is given, pick something reasonable
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')

        # Download daily data (use interval="1m" if you need intraday data)
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval="1m", prepost=True)
        except Exception as e:
            print("Failed to download data from Yahoo Finance:", e)
            return None

        if data.empty:
            print(f"No Yahoo Finance data retrieved for {symbol} from {start_date} to {end_date}.")
            return pd.DataFrame()

        data.reset_index(inplace=True)
        data.rename(
            columns={
                'Datetime': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',  # optional
                'Volume': 'volume'
            },
            inplace=True
        )
        # Adjust time, convert to epoch, and create a 'time' column.
        data['date'] = data['date'] - pd.Timedelta(hours=1)
        data['epoch_time'] = data['date'].apply(lambda dt: dt.timestamp())
        data['time'] = data['date'].dt.strftime("%H:%M:%S")
        data['close'] = data['close'].round(2)
        data.sort_values(by='epoch_time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    else:
        # --- POLYGON API BRANCH ---
        # Use Polygon API by default rather than local SQLite.
        # If no date range is provided, use defaults.
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
            
        api_key = os.environ.get('API_KEY')
        if not api_key:
            print("Polygon API key not found in environment variable 'API_KEY'.")
            return pd.DataFrame()

        limit = 5000  # or adjust as needed
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
        url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
        
        response = requests.get(url)
        response_json = response.json()
        
        if response.status_code == 200 and "results" in response_json:
            results = response_json["results"]
        else:
            print(f"No Polygon data retrieved for {symbol} from {start_date} to {end_date}.")
            return pd.DataFrame()
        
        # Convert results (a list of dictionaries) into a DataFrame.
        data = pd.DataFrame(results)
        # Polygon's field definitions:
        #   t: timestamp (in milliseconds)
        #   o: open, h: high, l: low, c: close, v: volume, n: number of trades.
        new_data = pd.DataFrame({
            'epoch_time': [item['t'] / 1000.0 for item in results],  # Convert ms to seconds
            # 'date': [pd.to_datetime(item['t'], unit='ms') for item in results],
            'date': [pd.to_datetime(item['t'], unit='ms', utc=True)
                    .tz_convert(central).strftime("%Y-%m-%d") for item in results],
            'time': [pd.to_datetime(item['t'], unit='ms', utc=True)
                    .tz_convert(central).strftime("%H:%M:%S") for item in results],
            # 'date': [pd.to_datetime(item['t'], unit='ms').strftime("%Y-%m-%d") for item in results],
            # 'time': [pd.to_datetime(item['t'], unit='ms').strftime("%H:%M:%S") for item in results],
            'open':  [item['o'] for item in results],
            'high':  [item['h'] for item in results],
            'low':   [item['l'] for item in results],
            'close': [round(item['c'], 2) for item in results],
            'volume':[item['v'] for item in results]
        })

        # data['epoch_time'] = data['t'] / 1000.0   # Convert ms to seconds.
        # data['open']  = data['o']
        # data['high']  = data['h']
        # data['low']   = data['l']
        # data['close'] = data['c'].round(2)
        # data['volume'] = data['v']
        # Create a 'time' column based on the epoch_time.
        # new_data['time'] = pd.to_datetime(new_data['epoch_time'], unit='s').dt.strftime("%H:%M:%S")
        
        new_data.sort_values(by='epoch_time', inplace=True)
        new_data.reset_index(drop=True, inplace=True)
        # print(new_data)
        # print(new_data.columns)
        # exit()
        return new_data



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
        find_local_minima_in_slope: bool = False,
        rand_forest_class = None
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
        self.rand_forest_class = rand_forest_class


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

    # print(f"processing {date_str}")
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

    # print(day_df_plot)
    if not config.show_plots and not config.save_plots:
        return  # We won't do any actual plotting
    # print(len(day_df_plot))
    # exit()
    if len(day_df_plot) != 0:
        forest_day_df = prepare_normalized_features(day_df_plot)
        predicted_classes = config.rand_forest_class.predict(forest_day_df)
    else:
        predicted_classes = []

    # print(forest_day_df)
    # exit()
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    ax3.sharex(ax1)
    # fig.suptitle(f"Close + RSI + MA_25 Slope - {date_str}", fontsize=14)
    fig.suptitle(f"Close + FFT(Slope) + Slope - {date_str}", fontsize=14)

    # 1) Top subplot: Close + MAs
    ax1.plot(day_df_plot['time'], day_df_plot['close'], label='Close', color='blue')
    if config.add_moving_averages and config.ma_windows:
        for window in config.ma_windows:
            ax1.plot(day_df_plot['time'], day_df_plot[f"MA_{window}"], label=f"MA_{window}")

    ax1.set_ylabel('Close')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    # --- Overlay light purple boxes on ax1 for segments where predicted class == 1 for at least 3 consecutive points.
    runs = []
    n = len(predicted_classes)
    i = 0
    while i < n:
        if predicted_classes[i] == 1:
            start = i
            while i < n and predicted_classes[i] == 1:
                i += 1
            end = i - 1
            if (end - start + 1) >= 3:
                runs.append((start, end))
        else:
            i += 1
    for start, end in runs:
        start_time_val = day_df_plot.iloc[start]['time']
        end_time_val = day_df_plot.iloc[end]['time']
        # axvspan will overlay a box between these times.
        ax1.axvspan(start_time_val, end_time_val, facecolor='plum', alpha=0.3, edgecolor='purple', linewidth=2)
    # --- End of classification overlay.

    # ------------------ 2) MIDDLE SUBPLOT: FFT of 'chosen_slope' ------------------
    # Drop NaNs before FFT
    slope_data = day_df_plot['chosen_slope'].dropna()
    N = len(slope_data)
    if N > 1:
        fft_values = np.fft.fft(slope_data)
        freqs = np.fft.fftfreq(N, d=1/60)  # d=1 => each sample is "1 step" apart

        # Plot only the first half of frequencies (0 .. N/2) if you want the standard real-valued spectrum
        half_N = N // 2
        abs_vals = np.abs(fft_values)[:half_N]
        idx_top = np.argsort(abs_vals)[-3:]   # grabs the last 3 indices (largest)
        top_freqs = freqs[:half_N][idx_top]
        top_mags = abs_vals[idx_top]
        info_str = "Top 3 peaks:\n"
        for f, m in sorted(zip(top_freqs, top_mags), key=lambda x: -x[1]):
            info_str += f"freq = {f:.2f}, mag = {m:.2f}\n"
        ax2.text(
            0.1, 0.95,
            info_str,
            transform=ax2.transAxes,
            va='top', ha='left',
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )


        ax2.plot(freqs[:half_N], np.abs(fft_values)[:half_N], color='purple', label='FFT of Slope')
        # ax2.set_xlim([0, freqs[half_N - 1]])
        ax2.set_xlim([0, 10])
        ax2.set_ylabel('FFT Magnitude')
        # ax2.legend(loc='upper left')
        ax2.grid(True)
        tick_positions = np.arange(0, freqs[half_N - 1], 1)
        ax2.set_xticks(tick_positions)
        # optionally also set custom labels
        ax2.set_xticklabels([f"{pos:.1f}" for pos in tick_positions])

        f_min = 1.0  # 1 cycle/hour => once every 60 minutes
        f_max = 2.0  # 2 cycles/hour => once every 30 minutes

        # Add shading for that frequency band:
        ax2.axvspan(f_min, f_max, color='grey', alpha=0.3)


    else:
        ax2.text(0.5, 0.5, "Insufficient data for FFT", ha='center', va='center', fontsize=12)
        ax2.grid(True)
    # # 2) Middle subplot: RSI
    # if config.add_rsi:
    #     ax2.plot(day_df_plot['time'], day_df_plot['RSI_14'], color='purple', label='RSI (14)')
    #     ax2.axhline(70, color='red', linestyle='--', alpha=0.7)
    #     ax2.axhline(30, color='green', linestyle='--', alpha=0.7)
    #     ax2.set_ylabel('RSI')
    #     ax2.legend(loc='upper left')
    #     ax2.grid(True)
    # else:
    #     ax2.text(0.5, 0.5, "RSI Disabled", ha='center', va='center', fontsize=12)
    #     ax2.set_ylabel('RSI')
    #     ax2.grid(True)

    # 3) Bottom subplot: chosen slope
    ax3.plot(day_df_plot['time'], day_df_plot['chosen_slope'], label='Chosen Slope', color='orange')
    ax3.set_ylabel('Slope')
    ax3.set_xlabel('Time of Day')
    ax3.grid(True)
    ax3.legend(loc='upper left')
    average_slope = day_df_plot['chosen_slope'].mean()
    ax3.axhline(y=average_slope, color='red', linestyle='--', label=f'Average Slope: {average_slope:.2f}')  # Customize color and style



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
            start_time = day_df_plot.iloc[start+30]['time']
            end_time   = day_df_plot.iloc[end]['time']
            for ax in (ax1, ax2, ax3):
                ax.axvspan(start_time, end_time, color='lightgreen', alpha=0.3)

    # Mark local minima in slope if requested
    if config.find_local_minima_in_slope:
        # Just an example: find local minima between 09:00 and 13:30
        start_local = pd.to_datetime("09:00:00").time()
        end_local   = pd.to_datetime("13:30:00").time()
        mask_local = (day_df['time'].dt.time >= start_local) & (day_df['time'].dt.time <= end_local)
        df_local = day_df.loc[mask_local].copy().reset_index(drop=True)
        local_min_indices = find_local_minima(df_local['chosen_slope'], threshold=-0.09)

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
    recent_time = get_recent_purple_start_longer_than_5_minutes(day_df_plot, predicted_classes)
    if recent_time:
        print(f"[{date_str}]: Most recent purple section (long run) starts at: {recent_time.strftime('%H:%M')}")
        return recent_time.strftime('%H:%M')
    else:
        return None



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
    local_min_indices = find_local_minima(day_df['chosen_slope'], threshold=-0.09)
    
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

def prepare_normalized_features(df):
    """
    Processes an input DataFrame with columns:
      - 'date'
      - 'time' (string formatted as '%H:%M:%S')
      - 'close'
      - 'MA_15'  (to be renamed to 'SMA_15')
      - 'MA_25'  (to be renamed to 'SMA_25')
      - 'MA_100' (to be renamed to 'SMA_100')
      - 'chosen_slope' (to be renamed to 'slope_10')
    
    The function performs the following steps:
      1. Converts the 'time' column to datetime.time.
      2. For each day (grouped by 'date'), finds the close value at 8:30 and uses it to normalize
         the 'close' and MA columns.
      3. Renames the columns to 'SMA_15', 'SMA_25', 'SMA_100', and 'slope_10'.
      4. Computes three run-length features (per day):
           - sma_25_below_100_run_length: the run length (in rows) for which SMA_25 < SMA_100.
           - negative_slope_run_length: the run length for which slope_10 < 0.
           - positive_slope_run_length: the run length for which slope_10 > 0.
      5. Returns a new DataFrame with columns in the following order:
         [close, SMA_100, SMA_25, SMA_15, slope_10,
          sma_25_below_100_run_length, negative_slope_run_length, positive_slope_run_length]
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The processed DataFrame with normalized values and run-length features.
    """
    # Ensure the 'time' column is in datetime.time format.
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

    # Define the normalization time.
    norm_time = pd.to_datetime("08:30", format='%H:%M').time()
    
    # Function to normalize a day's data using the close at 8:30.
    def normalize_group(group):
        norm_rows = group[group['time'] == norm_time]
        if norm_rows.empty:
            # Fallback: if no row at 08:30, use the first row's close.
            norm_val = group.iloc[0]['close']
        else:
            norm_val = norm_rows.iloc[0]['close']
        group['close'] = group['close'] / norm_val
        group['MA_15'] = group['MA_15'] / norm_val
        group['MA_25'] = group['MA_25'] / norm_val
        group['MA_100'] = group['MA_100'] / norm_val
        return group

    # Normalize each day.
    df_norm = df.groupby('date', group_keys=False).apply(normalize_group)
    
    # Rename columns as specified.
    df_norm = df_norm.rename(columns={
        'MA_15': 'SMA_15',
        'MA_25': 'SMA_25',
        'MA_100': 'SMA_100',
        'chosen_slope': 'slope_10'
    })
    
    # Create a new DataFrame with only the desired columns.
    # (Keep 'date' and 'time' temporarily for groupby operations.)
    df_proc = df_norm[['date', 'time', 'close', 'SMA_100', 'SMA_25', 'SMA_15', 'slope_10']].copy()
    
    # Define a function to compute run-length features for each day.
    def compute_run_lengths(group):
        # Compute sma_25_below_100_run_length.
        condition = group['SMA_25'] < group['SMA_100']
        group['sma_25_below_100_run_length'] = condition.groupby((~condition).cumsum()).cumcount() + 1
        group.loc[~condition, 'sma_25_below_100_run_length'] = 0
        
        # Compute negative_slope_run_length.
        neg_condition = group['slope_10'] < 0
        group['negative_slope_run_length'] = neg_condition.groupby((~neg_condition).cumsum()).cumcount() + 1
        group.loc[~neg_condition, 'negative_slope_run_length'] = 0
        
        # Compute positive_slope_run_length.
        pos_condition = group['slope_10'] > 0
        group['positive_slope_run_length'] = pos_condition.groupby((~pos_condition).cumsum()).cumcount() + 1
        group.loc[~pos_condition, 'positive_slope_run_length'] = 0
        
        return group

    # print(df_proc)
    # exit()
    df_features = df_proc.groupby('date', group_keys=False).apply(compute_run_lengths)
    
    # Select and order the final columns as required.
    final_columns = ['close', 'SMA_100', 'SMA_25', 'SMA_15', 'slope_10',
                     'sma_25_below_100_run_length', 'negative_slope_run_length', 'positive_slope_run_length']
    result_df = df_features[final_columns].copy()
    return result_df

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
    if df is None:
        return None
    ret_df = df.copy(deep=True)

    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # print(date_range)
    all_stats = []

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
            recent_time = plot_close_chart_for_day(selected_df, date_str, plot_config)

        # Collect stats if desired
        if collect_stats:
            day_stats = compute_daily_stats(selected_df, date_str)
            all_stats.append(day_stats)

    # Convert stats to DataFrame for further use
    stats_df = pd.DataFrame(all_stats)
    # print(selected_df)
    return ret_df, recent_time


###############################################################################
# Core loop that runs every minute
###############################################################################
def start_monitoring(symbol="SPY", minutes_interval=15, config: PlotConfig = None):
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
    last_recent_time = None
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
        
        today = datetime.date.today()
        formatted_today = today.strftime("%Y-%m-%d")
        tomorrow = today + datetime.timedelta(days=1)
        formatted_tomorrow = tomorrow.strftime("%Y-%m-%d")

        threshold_time = datetime.time(15, 0)
        now_time = datetime.datetime.now().time()
        if now_time > threshold_time:
            exit()

        print(f"=========== [{now_time.strftime("%H:%M:%S")}] Fetching Stock Data ===========")
        try:
            df, recent_time = run_analysis(
                symbol="SPY",
                start_date=formatted_today,
                end_date=formatted_tomorrow,
                plot_config=config,
                collect_stats=True,
                use_yahoo = False
            )
        except:
            time.sleep(60)
            continue

        if df is None:
            time.sleep(60)
            continue
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
        
        local_min_indices = find_local_minima(df['chosen_slope'], threshold=-0.09)
        # local_min_indices = []
        # arr = df["close"].to_numpy()
        # for i in range(1, len(arr) - 1):
        #     if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
        #         local_min_indices.append(i)
        # print(local_min_indices)

        # Convert to a set for easier comparison
        if (len(local_min_indices) > 0):
            current_idx = max(local_min_indices)
            current_time = df["epoch_time"].iloc[max(local_min_indices)]
        # print(df["epoch_time"].iloc[max(local_min_indices)])
        # exit()
        if (len(local_min_indices) > 0) and (current_minima != current_time):
            on_new_local_minimum_found(current_idx, df["close"].iloc[current_idx], df)
            current_minima = current_time
        elif recent_time != last_recent_time:
            last_recent_time = recent_time
            on_new_local_minimum_found(current_idx, df["close"].iloc[current_idx], df)
        elif iteration_count % minutes_interval == 0:
            on_interval_elapsed(df)

        # # Find newly discovered minima
        # new_minima = current_minima - known_minima
        # if new_minima:
        #     for nm in new_minima:
        #         on_new_local_minimum_found(nm, df["close"].iloc[nm], df)
        #     # Update our global record of known minima
        #     known_minima.update(new_minima)

        # # ----------------------------------------------------
        # # 3) Trigger event if we've hit the interval
        # # ----------------------------------------------------
        # if iteration_count % minutes_interval == 0:
        #     on_interval_elapsed(df)

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
    
    model_path = "other_analysis/random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        rf_classifier = pickle.load(f)
    # new_sample = np.array([[1.0, 0.95, 0.97, 0.98, -0.02, 3, 0, 1]])

    # prediction = rf_classifier.predict(new_sample)
    # print("Prediction for new sample:", prediction)
    # exit()

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
        # show_plots=True,
        slope_source='close',
        slope_lookback=10,
        rand_forest_class = rf_classifier
    )

    # start_monitoring(config=config, minutes_interval=30)
    df, _ = run_analysis(
        symbol="SPY",
        # start_date="2020-01-01",
        start_date="2025-02-01",
        end_date="2025-02-14",
        plot_config=config,
        collect_stats=True,
        use_yahoo = False
    )

    # # stats now holds a DataFrame with columns like ["date", "daily_min_slope"]
    # print(stats)
