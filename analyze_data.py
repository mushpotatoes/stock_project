import os
import datetime
import pickle
import sqlite3
import time
import warnings
import logging
import base64
import mimetypes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import requests
import joblib
from email.message import EmailMessage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from sklearn.preprocessing import StandardScaler

# Google/Gmail imports
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

# -----------------------------------------------------------------------------
# Logging & Warnings
# -----------------------------------------------------------------------------
log_filename = 'analyze.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, 'a'),
        logging.StreamHandler()
    ]
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# -----------------------------------------------------------------------------
# Gmail Functions
# -----------------------------------------------------------------------------
def gmail_create_draft_with_attachment(creds, body="Hello,\n\nThis is an automated mail with attachment.\nPlease do not reply."):
    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        mime_message = EmailMessage()
        today = datetime.date.today()
        formatted_date = today.strftime("%Y-%m-%d")

        mime_message["To"] = "wilkpaulc@gmail.com"
        mime_message["From"] = "wilkpaulc@gmail.com"
        mime_message["Subject"] = f"POI on {formatted_date}"
        mime_message.set_content(body)

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

        encoded_message = base64.urlsafe_b64encode(mime_message.as_bytes()).decode()
        create_draft_request_body = {"message": {"raw": encoded_message}}
        draft = service.users().drafts().create(userId="me", body=create_draft_request_body).execute()
        logging.debug(f"Draft created. ID: {draft['id']}")
        return draft

    except HttpError as error:
        logging.error(f"An error occurred while creating the draft: {error}")
        return None

def gmail_send_draft(draft_id, creds):
    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        sent_message = service.users().drafts().send(userId="me", body={"id": draft_id}).execute()
        logging.debug(f"Draft with ID {draft_id} has been sent successfully.")
        return sent_message
    except HttpError as error:
        logging.error(f"An error occurred while sending the draft: {error}")
        return None

# -----------------------------------------------------------------------------
# Callback Placeholders
# -----------------------------------------------------------------------------
def on_interval_elapsed(df):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Interval event triggered.")
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    draft = gmail_create_draft_with_attachment(creds, "Triggered by interval")
    if draft is None:
        print("Failed to create draft.")
        return
    draft_id = draft["id"]
    sent_msg = gmail_send_draft(draft_id, creds)
    if sent_msg:
        logging.debug("Email successfully sent.")
    else:
        logging.error("Failed to send the email.")

def on_new_local_minimum_found(min_idx, min_value, df):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New local minimum at index={min_idx} [{df['time'].iloc[min_idx]}], value={min_value}")
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    draft = gmail_create_draft_with_attachment(creds, "Triggered by minimum slope")
    if draft is None:
        logging.error("Failed to create draft.")
        return
    draft_id = draft["id"]
    sent_msg = gmail_send_draft(draft_id, creds)
    if sent_msg:
        logging.debug("Email successfully sent.")
    else:
        logging.error("Failed to send the email.")

def on_event_message(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] New event: {message}")
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    draft = gmail_create_draft_with_attachment(creds, message)
    if draft is None:
        print("Failed to create draft.")
        return
    draft_id = draft["id"]
    sent_msg = gmail_send_draft(draft_id, creds)
    if sent_msg:
        print("Email successfully sent.")
    else:
        print("Failed to send the email.")

# -----------------------------------------------------------------------------
# Prediction Overlay Helper Functions
# -----------------------------------------------------------------------------
def scale_01_limited_stretch(arr, min_range_threshold=0.1):
    """
    Rescales a NumPy array to the range [0, 1], but limits stretching
    if the original range is smaller than a threshold.

    Args:
        arr (np.ndarray): Input array. Assumed to have values within [0, 1].
        min_range_threshold (float): The minimum range width to use for scaling.
                                     If the actual data range is smaller, scaling
                                     will be performed as if the range was this width,
                                     centered around the data's midpoint. Must be > 0.

    Returns:
        np.ndarray: Array scaled potentially to a sub-interval of [0, 1].
    """
    if min_range_threshold <= 0:
        raise ValueError("min_range_threshold must be greater than 0")

    min_val = arr.min()
    max_val = arr.max()
    actual_range = max_val - min_val

    # Handle the edge case where all values are the same initially
    if actual_range == 0:
         # If range is zero, no scaling needed or possible in a meaningful way.
         # Return a constant array. Since input is [0,1], returning the
         # constant value itself seems reasonable. Or map to 0.5? Let's return the value.
         # Clipping ensures it stays within [0,1] just in case.
         print("Warning: Input array has zero range.")
         # return np.full_like(arr, np.clip(min_val, 0.0, 1.0), dtype=np.float64)
         # Or return zeros consistent with the basic scaler
         return np.zeros_like(arr, dtype=np.float64)


    # Determine the effective min and max for scaling
    if actual_range < min_range_threshold:
        # Center the threshold range around the midpoint of the actual data
        mid = (min_val + max_val) / 2.0
        effective_min = mid - min_range_threshold / 2.0
        effective_max = mid + min_range_threshold / 2.0

        # Ensure the effective bounds stay within [0, 1]
        effective_min = max(0.0, effective_min)
        effective_max = min(1.0, effective_max)

        # Recalculate effective range, it might be clipped
        effective_range = effective_max - effective_min

         # Handle extremely rare case where clipping collapses the effective range to zero
        if effective_range <= 1e-9: # Use epsilon for float comparison
            print("Warning: Effective range collapsed to zero after thresholding and clipping.")
            # Fallback: return values mapped to the single effective point
            # This point is effective_min (which equals effective_max)
            # Scaled relative to [0,1], this value is just effective_min itself.
            # return np.full_like(arr, effective_min, dtype=np.float64)
            # Or return zeros consistent with other zero-range cases
            return np.zeros_like(arr, dtype=np.float64)

        print(f"Info: Actual range {actual_range:.4f} < threshold {min_range_threshold:.4f}. Using effective range [{effective_min:.4f}, {effective_max:.4f}].")
        scale_min = effective_min
        scale_range = effective_range
    else:
        # Use the actual data range for scaling
        scale_min = min_val
        scale_range = actual_range

    # Apply scaling formula using the chosen min and range
    scaled_arr = (arr - scale_min) / scale_range
    print(f"Scale min: {scale_min}; Scale range: {scale_range}")

    # Clip the final result to ensure it's strictly within [0, 1]
    # This is important especially when using the effective range,
    # as the original data might slightly exceed the calculated effective bounds.
    return np.clip(scaled_arr, 0.0, 1.0)

def center_mean_05_rescale(arr):
  """
  Shifts the array mean to 0.5 and rescales if necessary to fit [0, 1].

  Args:
    arr (np.ndarray): Input array.

  Returns:
    np.ndarray: Array with mean guaranteed to be 0.5 and values in [0, 1].
  """
  if arr.size == 0:
      return arr # Return empty array if input is empty
      
  # Ensure array is float for calculations
  arr = arr.astype(np.float64)

  target_mean = 0.5
  current_mean = arr.mean()

  # 1. Shift the array so the mean is target_mean (0.5)
  shifted_arr = arr + (target_mean - current_mean)

  # 2. Check the range of the shifted array
  min_shifted = shifted_arr.min()
  max_shifted = shifted_arr.max()

  # If already within [0, 1], we're done
  if min_shifted >= 0.0 and max_shifted <= 1.0:
      # Check for potential floating point issues very close to boundaries
      return np.clip(shifted_arr, 0.0, 1.0)

  # 3. If outside [0, 1], calculate the necessary scaling factor
  #    We want to scale around the target_mean.
  #    Formula: final = scale_factor * (shifted - target_mean) + target_mean
  #    We need:
  #    0 <= scale_factor * (min_shifted - target_mean) + target_mean  => scale_factor <= target_mean / (target_mean - min_shifted) [if min_shifted < 0]
  #    1 >= scale_factor * (max_shifted - target_mean) + target_mean  => scale_factor <= (1 - target_mean) / (max_shifted - target_mean) [if max_shifted > 1]

  scale_factor = 1.0
  epsilon = 1e-9 # To avoid division by zero if min/max is exactly 0.5

  if min_shifted < 0.0:
      # How much scaling is needed to bring min_shifted to 0?
      scale_low = target_mean / (target_mean - min_shifted + epsilon)
      scale_factor = min(scale_factor, scale_low)

  if max_shifted > 1.0:
      # How much scaling is needed to bring max_shifted to 1?
      scale_high = (1.0 - target_mean) / (max_shifted - target_mean + epsilon)
      scale_factor = min(scale_factor, scale_high)

  # Ensure scale factor is not negative (shouldn't happen with target_mean=0.5)
  scale_factor = max(0.0, scale_factor)

  # 4. Apply the scaling transformation
  final_arr = scale_factor * (shifted_arr - target_mean) + target_mean

  # Final clip for safety due to potential float inaccuracies
  return np.clip(final_arr, 0.0, 1.0)

def get_prediction_runs(predicted_classes, min_width):
    """
    Compute runs (start, end) where predicted_classes == 1 consecutively
    for at least min_width points.
    """
    runs = []
    n = len(predicted_classes)
    i = 0
    while i < n:
        if predicted_classes[i] == 1:
            start = i
            while i < n and predicted_classes[i] == 1:
                i += 1
            end = i - 1
            if (end - start + 1) >= min_width:
                runs.append((start, end))
        else:
            i += 1
    return runs

def shade_prediction_runs(ax, day_df, runs, facecolor, edgecolor, alpha, hatch=None, linewidth=2):
    """
    Shade regions for each run on the given axis.
    
    Parameters:
        ax: The axis on which to draw.
        day_df: DataFrame containing a 'datetime' column.
        runs: List of (start, end) index tuples.
        facecolor: Fill color.
        edgecolor: Edge color.
        alpha: Transparency.
        hatch: Hatch pattern (optional).
        linewidth: Line width for the edge.
    """
    for start, end in runs:
        start_time_val = day_df.iloc[start]['datetime']
        end_time_val = day_df.iloc[end]['datetime']
        ax.axvspan(start_time_val, end_time_val,
                   facecolor=facecolor, edgecolor=edgecolor,
                   alpha=alpha, hatch=hatch, linewidth=linewidth)

# -----------------------------------------------------------------------------
# Data Loading & Filtering
# -----------------------------------------------------------------------------
central = pytz.timezone('US/Central')

def load_and_sort_stock_data(symbol, repo_root, use_yahoo=False, start_date=None, end_date=None, use_sql=True):
    if use_yahoo:
        import yfinance as yf
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval="1m", prepost=True)
        except:
            print("Failed to download data")
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
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            },
            inplace=True
        )
        data['date'] = data['date'] - pd.Timedelta(hours=1)
        data['epoch_time'] = data['date'].apply(lambda dt: dt.timestamp())
        data['time'] = data['date'].dt.strftime("%H:%M:%S")
        data['close'] = data['close'].round(2)
        data.sort_values(by='epoch_time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    elif use_sql:
        db_path = os.path.join(repo_root, f"big_{symbol}_data.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_names = [row[0] for row in cursor.fetchall()]
        df_list = []
        for table_name in table_names:
            query = f"SELECT * FROM {table_name}"
            df_list.append(pd.read_sql_query(query, conn))
        conn.close()
        df = pd.concat(df_list, ignore_index=True)
        df.sort_values(by='epoch_time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        if start_date is None:
            start_date = "2020-01-01"
        if end_date is None:
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        api_key = os.environ.get('API_KEY')
        if not api_key:
            print("Polygon API key not found in environment variable 'API_KEY'.")
            return pd.DataFrame()
        limit = 5000
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
        url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"
        response = requests.get(url)
        response_json = response.json()
        if response.status_code == 200 and "results" in response_json:
            results = response_json["results"]
        else:
            print(f"No Polygon data retrieved for {symbol} from {start_date} to {end_date}.")
            return pd.DataFrame()
        new_data = pd.DataFrame({
            'epoch_time': [item['t'] / 1000.0 for item in results],
            'date': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(central).strftime("%Y-%m-%d") for item in results],
            'time': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(central).strftime("%H:%M:%S") for item in results],
            'open':  [item['o'] for item in results],
            'high':  [item['h'] for item in results],
            'low':   [item['l'] for item in results],
            'close': [round(item['c'], 2) for item in results],
            'volume':[item['v'] for item in results]
        })
        new_data.sort_values(by='epoch_time', inplace=True)
        new_data.reset_index(drop=True, inplace=True)
        return new_data

def convert_to_epoch(date_str, time_str):
    datetime_str = f"{date_str} {time_str}"
    dt_object = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    return dt_object.timestamp()

def filter_epoch_range(df, start, end):
    mask = (df['epoch_time'] >= start) & (df['epoch_time'] <= end)
    filtered_df = df.loc[mask].sort_values(by='epoch_time').reset_index(drop=True)
    return filtered_df

# -----------------------------------------------------------------------------
# Utility / Analysis Helpers
# -----------------------------------------------------------------------------
def find_last_below_threshold(df, threshold):
    n = len(df)
    for i in range(n - 2, -1, -1):
        if df.iloc[i]["volume"] < threshold:
            if (df.iloc[i + 1:]["volume"] >= threshold).all():
                return i
    return None

def slope_of_best_fit(values):
    x = np.arange(len(values))
    y = values
    slope, _ = np.polyfit(x, y, 1)
    return slope

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def find_local_minima(series, threshold=-0.01):
    minima_indices = []
    arr = series.to_numpy()
    for i in range(1, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1] and arr[i] < threshold:
            minima_indices.append(i)
    return minima_indices

def find_local_maxima(series):
    maxima_indices = []
    arr = series.to_numpy()
    for i in range(1, len(arr) - 1):
        if (arr[i] > arr[i - 1]) and (arr[i] > arr[i + 1]):
            maxima_indices.append(i)
    return maxima_indices

# -----------------------------------------------------------------------------
# Plot Config & Plotting
# -----------------------------------------------------------------------------
class PlotConfig:
    def __init__(
        self,
        show_plots: bool = True,
        save_plots: bool = False,
        output_dir: str = "./plots",
        plot_volume: bool = False,
        volume_threshold_lines: bool = True,
        plot_close: bool = False,
        add_rsi: bool = True,
        add_moving_averages: bool = True,
        ma_windows = (15, 25, 100),
        restrict_830_to_300: bool = True,
        slope_source: str = "MA_25",
        slope_lookback: int = 10,
        highlight_negative_slope_runs: bool = True,
        highlight_below_ma_runs: bool = True,
        highlight_buy_predictions : bool = True,
        find_local_minima_in_slope: bool = False,
        ma_run_len: int = 30,
        high_buy_overlay_min_width: int = 3,
        use_neg_pos_models: bool = False,
        use_down_models: bool = False,
        rand_forest_class = None,
        neg_classifier = None,
        pos_classifier = None,
        down_classifier = None,
        norm_time = pd.to_datetime("08:30:00").time()
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
        self.highlight_buy_predictions = highlight_buy_predictions
        self.ma_run_len = ma_run_len
        self.find_local_minima_in_slope = find_local_minima_in_slope
        self.high_buy_overlay_min_width = high_buy_overlay_min_width
        self.use_neg_pos_models = use_neg_pos_models
        self.use_down_models = use_down_models
        self.rand_forest_class = rand_forest_class
        self.neg_classifier = neg_classifier
        self.pos_classifier = pos_classifier
        self.down_classifier = down_classifier
        self.norm_time = norm_time

def plot_volume_chart_for_day(day_df, date_str, config: PlotConfig):
    subset_df = day_df.iloc[50:330]
    volume_mean = subset_df['volume'].mean()
    volume_std = subset_df['volume'].std()
    crossing_idx = find_last_below_threshold(day_df.iloc[0:390], volume_mean + volume_std)
    if not config.show_plots and not config.save_plots:
        return crossing_idx
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.plot(day_df['volume'], label='Volume', color='blue')
    if config.volume_threshold_lines:
        ax.axhline(y=volume_mean, color='red', linestyle='--', label=f'Mean={volume_mean:.2f}')
        ax.axhline(y=volume_mean + volume_std, color='red', linestyle='--', label='Mean+Std')
    ax.axvline(x=0, color='red', linestyle='--', label='Start (x=0)')
    ax.axvline(x=390, color='red', linestyle='--', label='End (x=390)')
    if crossing_idx is not None:
        ax.axvline(x=crossing_idx, color='green', linestyle='-', label=f'Cross @ x={crossing_idx}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Volume')
    ax.set_title(f'Volume Chart - {date_str}')
    ax.legend()
    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        fname = os.path.join(config.output_dir, f"volume_{date_str}.png")
        plt.savefig(fname)
        plt.close(fig)
    elif config.show_plots:
        plt.show()
    return crossing_idx

def compute_daily_slopes(group, slope_column, lookback):
    group = group.sort_values(by='time')
    group[f'slope_{lookback}'] = group[slope_column].rolling(window=lookback).apply(slope_of_best_fit, raw=True)
    return group

def add_slope_difference(df, slope_lookback=10, group_by_date=True):
    slope_col = f'slope_{slope_lookback}'
    diff_col = f'd2_{slope_lookback}'
    if group_by_date and 'date' in df.columns:
        df[diff_col] = df.groupby('date')[slope_col].diff()
    else:
        df[diff_col] = df[slope_col].diff()
    return df

def compute_slopes_for_timeframe(timeframe, df, use_saved_scaler=True):
    df_local = df.copy()
    df_local = df_local.groupby('date', group_keys=False).apply(
        lambda group: compute_daily_slopes(group, 'close', timeframe)
    )
    df_local = add_slope_difference(df_local, slope_lookback=timeframe)
    new_cols = [f'slope_{timeframe}', f'd2_{timeframe}']
    new_cols_df = df_local[new_cols].copy()
    scalers_dir = "C:\\Users\\deade\\OneDrive\\Desktop\\data_science\\stock_project\\other_analysis\\scalers"
    os.makedirs(scalers_dir, exist_ok=True)
    scaler_filename = os.path.join(scalers_dir, f"scaler_timeframe_{timeframe}.pkl")
    if use_saved_scaler and os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
        new_cols_standardized = scaler.transform(new_cols_df)
    else:
        df = pd.concat([df, new_cols_df], axis=1)
        return df
        print(f"No saved scaler found. Fitting a new scaler for timeframe {timeframe}.")
        scaler = StandardScaler()
        new_cols_standardized = scaler.fit_transform(new_cols_df)
        joblib.dump(scaler, scaler_filename)
    new_cols_df_standardized = pd.DataFrame(new_cols_standardized, 
                                            columns=[col for col in new_cols],
                                            index=new_cols_df.index)
    return new_cols_df_standardized

def add_slope_run_length(df, slope_column='slope_10'):
    def compute_run_length(group):
        mask_negative = group[slope_column] < 0
        group['negative_slope_run_length'] = mask_negative.groupby((~mask_negative).cumsum()).cumcount() + 1
        group.loc[~mask_negative, 'negative_slope_run_length'] = 0
        mask_positive = group[slope_column] > 0
        group['positive_slope_run_length'] = mask_positive.groupby((~mask_positive).cumsum()).cumcount() + 1
        group.loc[~mask_positive, 'positive_slope_run_length'] = 0
        return group
    df_with_runs = df.groupby('date', group_keys=False).apply(compute_run_length)
    return df_with_runs

def add_sma_run_length(df, sma25_col='SMA_25', sma100_col='SMA_100', new_col='sma_25_below_100_run_length'):
    def compute_run_length(group):
        condition = group[sma25_col] < group[sma100_col]
        group[new_col] = condition.groupby((~condition).cumsum()).cumcount() + 1
        group.loc[~condition, new_col] = 0
        return group
    df_with_run = df.groupby('date', group_keys=False).apply(compute_run_length)
    return df_with_run

def plot_close_chart_for_day(day_df, date_str, config: PlotConfig):
    in_green = False
    # Prepare datetime for plotting
    day_df['time'] = pd.to_datetime(day_df['time'], format="%H:%M:%S").dt.time
    day_df['date'] = pd.to_datetime(day_df['date'], format='%Y-%m-%d').dt.date
    day_df['datetime'] = day_df['time'].apply(lambda t: datetime.datetime.combine(day_df['date'].iloc[0], t))

    df_with_slopes = prepare_normalized_features(day_df)
    day_df = compute_slopes_for_timeframe(10, day_df, False)
    timeframes = [15, 25, 100]
    for timeframe in timeframes:
        day_df[f'MA_{timeframe}'] = day_df.groupby('date')['close'].transform(lambda x: x.rolling(window=timeframe).mean())
  
    timeframes = [10, 15, 25, 40, 90, 100, 120]
    results = {}
    for tf in timeframes:
        try:
            result_df = compute_slopes_for_timeframe(tf, df_with_slopes)
            results[tf] = result_df
        except Exception as exc:
            print(f"Timeframe {tf} generated an exception: {exc}")
    for tf, new_cols_df in results.items():
        df_with_slopes = df_with_slopes.join(new_cols_df)
    timeframes = [10, 25, 40, 90, 100, 120]
    for timeframe in timeframes:
        df_with_slopes[f'SMA_{timeframe}'] = df_with_slopes.groupby('date')['close'].transform(lambda x: x.rolling(window=timeframe).mean())
    
    forest_day_df = add_sma_run_length(
        add_slope_run_length(df_with_slopes, slope_column='slope_10'),
        sma25_col='SMA_25', sma100_col='SMA_100', new_col='sma_25_below_100_run_length'
    )
    
    if config.restrict_830_to_300:
        logging.debug("Limiting plot range")
        start_plot_time = pd.to_datetime("08:30:00").time()
        end_plot_time   = pd.to_datetime("15:00:00").time()
        mask_plot = (day_df['time'] >= start_plot_time) & (day_df['time'] <= end_plot_time)
        day_df_plot = day_df.loc[mask_plot].copy().reset_index(drop=True)
        mask_plot = (forest_day_df['time'] >= start_plot_time) & (forest_day_df['time'] <= end_plot_time)
        forest_day_df = forest_day_df.loc[mask_plot].copy().reset_index(drop=True)
    else:
        day_df_plot = day_df.copy()
    selected_features = [
        "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
        "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
        "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120",
        "sma_25_below_100_run_length", "negative_slope_run_length", "positive_slope_run_length"
    ]
    forest_day_df = forest_day_df[selected_features]
    
    if not config.show_plots and not config.save_plots:
        logging.info("Not printing plots.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    ax3.sharex(ax1)
    ax2.sharex(ax1)
    fig.suptitle(f"Close + FFT(Slope) + Slope - {date_str}", fontsize=14)

    # Top subplot: Close + MAs
    ax1.plot(day_df_plot['datetime'], day_df_plot['close'], label='Close', color='blue')
    if config.add_moving_averages and config.ma_windows:
        for window in config.ma_windows:
            ax1.plot(day_df_plot['datetime'], day_df_plot[f"MA_{window}"], label=f"MA_{window}")
    ax1.set_ylabel('Close')
    below_runs = False
    if config.highlight_below_ma_runs:
        # Create a boolean series: True where 25-min MA is below 100-min MA
        # condition = (day_df_plot['MA_25'] < day_df_plot['MA_100'])
        condition = (forest_day_df['SMA_25'] < forest_day_df['SMA_100']) & \
                    ((forest_day_df['SMA_10'] < forest_day_df['SMA_25']) | (forest_day_df['slope_120'] < -0.9))

        # Convert to integers (1 for True, 0 for False) so we can use get_prediction_runs
        condition_int = condition.astype(int).to_numpy()
        # Identify runs that last at least 30 minutes (i.e. 30 consecutive points)
        ma_runs = get_prediction_runs(condition_int, config.ma_run_len + 1)
        # Create a new list with the adjusted start times by adding 30 to each run's start index
        adjusted_ma_runs = [(start + config.ma_run_len, end) for start, end in ma_runs]
        # Shade these runs on ax1; adjust facecolor, edgecolor, and alpha as desired
        shade_prediction_runs(ax1, day_df_plot,
                              adjusted_ma_runs,
                              facecolor='green',
                              edgecolor='green',
                              alpha=0.1, hatch='//')
        
        # in_green = condition.values[-1]
        in_green = condition.tail(30).all()
        

    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # --- Prediction Overlays ---
    extra_in_window = False
    if config.highlight_buy_predictions:
        logging.debug("Applying prediction overlays")
        if len(day_df_plot) != 0:
            rand_predicted_classes = config.rand_forest_class.predict(forest_day_df)
        else:
            rand_predicted_classes = []
        rand_runs = get_prediction_runs(rand_predicted_classes, config.high_buy_overlay_min_width)
        shade_prediction_runs(ax1, day_df_plot, rand_runs, facecolor='blue', edgecolor='navy', alpha=0.1)
        
    if config.use_neg_pos_models and len(forest_day_df) != 0:
        logging.debug("Applying strict prediction overlays")
        if forest_day_df['slope_10'].mean() < 0:
            extra_predicted_classes = config.neg_classifier.predict(forest_day_df)
            extra_up_predicted_classes = config.pos_classifier.predict(forest_day_df)
            extra_up_predicted_probs = config.pos_classifier.predict_proba(forest_day_df)[:,1]
        else:
            extra_predicted_classes = config.neg_classifier.predict(forest_day_df)
            extra_up_predicted_classes = config.pos_classifier.predict(forest_day_df)
            extra_up_predicted_probs = config.pos_classifier.predict_proba(forest_day_df)[:,1]
        # print((extra_up_predicted_probs > 0.5).astype(int))
        # print(extra_up_predicted_classes)
        # exit()
        extra_runs = get_prediction_runs(extra_predicted_classes, config.high_buy_overlay_min_width)
        if (extra_predicted_classes[-1] == 1) or (extra_up_predicted_probs[-1] > 0.55):
            extra_in_window = True
        shade_prediction_runs(ax1, day_df_plot, extra_runs, facecolor='plum', edgecolor='purple', alpha=0.2, hatch='..')

        # if(forest_day_df['d2_120'].std() < 0.6) and (day_df['time'].iloc[-1] > ):
        #     logging.info(f"Processing {date_str} {day_df['time'].iloc[-1]}")
        #     # logging.info(f"Slope 120 mean: {forest_day_df['slope_120'].mean()}, std. dev.: {forest_day_df['slope_120'].std()}")
        #     logging.info(f"d2 120 mean: {forest_day_df['d2_120'].mean()}, std. dev.: {forest_day_df['d2_120'].std()}")
        #     scale_01_limited_stretch(extra_up_predicted_probs, min_range_threshold=0.1)
        #     scale_min = 0.01
        #     scale_range = 0.65
        #     scaled_arr = (extra_up_predicted_probs - scale_min) / scale_range
        #     # print(f"Scale min: {scale_min}; Scale range: {scale_range}")

        #     # Clip the final result to ensure it's strictly within [0, 1]
        #     # This is important especially when using the effective range,
        #     # as the original data might slightly exceed the calculated effective bounds.
        #     extra_up_predicted_probs = np.clip(scaled_arr, 0.0, 1.0)
            
    # scaled_arr = (arr - scale_min) / scale_range
    # print(f"Scale min: {scale_min}; Scale range: {scale_range}")
        # min_range_threshold=0.1
        # min_val = extra_up_predicted_probs.min()
        # max_val = extra_up_predicted_probs.max()
        # actual_range = max_val - min_val
        # print(extra_up_predicted_probs.mean())
        # if extra_up_predicted_probs.mean() > 0.7:
        #     extra_up_predicted_probs = center_mean_05_rescale(extra_up_predicted_probs)

        # print(min(extra_up_predicted_probs))
        # print(max(extra_up_predicted_probs))
        # print(type(extra_up_predicted_probs))
        # if min(extra_up_predicted_probs) > 0.25:
        #     extra_up_predicted_probs = extra_up_predicted_probs - 0.2
        
        # Set the gradient probabilities
        for probs in np.arange(0.55, 1.01, 0.05):
            # low_prob = (extra_up_predicted_probs > (0.5 + probs)).astype(int)
            low_prob = (extra_up_predicted_probs > (probs)).astype(int)
            # print(low_prob)
            # input()
            # if forest_day_df['slope_10'].mean() < 0:
            extra_up_runs = get_prediction_runs(low_prob, 3)
            # print(extra_up_runs)
            if probs == 0.55:
                ec = 'purple'
            else:
                ec = None
            # if probs > 0.2:
            if probs > 0.8:
                alpha_setting = 0.15
            else:
                alpha_setting = 0.07
            shade_prediction_runs(ax1, day_df_plot, extra_up_runs, 
                                  facecolor='red', edgecolor=ec, alpha=(alpha_setting))#, hatch='.')

        for probs in np.arange(0.45, -0.01, -0.05):
            # low_prob = (extra_up_predicted_probs > (0.5 + probs)).astype(int)
            low_prob = (extra_up_predicted_probs < (probs)).astype(int)
            # print(low_prob)
            # input()
            # if forest_day_df['slope_10'].mean() < 0:
            extra_up_runs = get_prediction_runs(low_prob, 3)
            # print(extra_up_runs)
            if probs == 0.45:
                ec = 'navy'
            else:
                ec = None
            # if probs > 0.2:
            if probs < 0.2:
                alpha_setting = 0.1
            else:
                alpha_setting = 0.05
            shade_prediction_runs(ax1, day_df_plot, extra_up_runs, 
                                  facecolor='blue', edgecolor=ec, alpha=(alpha_setting))#, hatch='.')

    if config.use_down_models and len(forest_day_df) != 0:
        down_predicted_classes = config.down_classifier.predict(forest_day_df)
        down_runs = get_prediction_runs(down_predicted_classes, config.high_buy_overlay_min_width)
        # print(down_runs)
        shade_prediction_runs(ax1, day_df_plot, down_runs, facecolor='gray', edgecolor='gray', alpha=0.2, hatch='\\\\')

    # Bottom subplot: 10 Minute Slope
    ax3.plot(day_df_plot['datetime'], forest_day_df['slope_10'], label='10 Minute Slope', color='orange')
    ax3.set_ylabel('Slope (10 minute)')
    ax3.set_xlabel('Time of Day')
    ax3.grid(True)
    ax3.legend(loc='upper right')
    average_slope = forest_day_df['slope_10'].mean()
    ax3.axhline(y=average_slope, color='red', linestyle='--', label=f'Average Slope: {average_slope:.2f}')
    ax3.set_ylim(-3,3)

    ax2.plot(day_df_plot['datetime'], forest_day_df['slope_120'], label='120 Minute Slope', color='black', alpha=0.2)
    ax2.set_ylabel('Slope (120 minute)')
    ax2.legend(loc='lower right')
    ax2.axhline(y=0, color='blue', linestyle='--')
    ax2.grid(True)
    ax2.set_ylim(-3,3)
    ax2.axhspan(-0.5, 0.5, color='lightgray', alpha=0.4)

    latest_datetime = day_df_plot['datetime'].max()
    threshold_time = datetime.time(9, 0)
    if latest_datetime.time() > threshold_time:
        y = forest_day_df['slope_120'].values
        N = len(y)
        fft_y = np.fft.fft(y)
        num_components = 5
        fft_y_filtered = np.zeros_like(fft_y)
        fft_y_filtered[:num_components] = fft_y[:num_components]
        fft_y_filtered[-(num_components-1):] = fft_y[-(num_components-1):]
        y_reconstructed = np.fft.ifft(fft_y_filtered)
        y_reconstructed = np.real(y_reconstructed)
        ax2.plot(day_df_plot['datetime'], y_reconstructed, 'b-', label='Reconstructed Signal (5 FFT components)')
        ax2.fill_between(day_df_plot['datetime'], y_reconstructed, 0, where=(y_reconstructed <= 0), facecolor='lightblue', alpha=0.5)

    ax2_secondary = ax2.twinx()
    forest_day_df['second_derivative'] = np.diff(forest_day_df['slope_120'], prepend=forest_day_df['slope_120'][0])
    ax2_secondary.plot(day_df_plot['datetime'], forest_day_df['second_derivative'], label='Second Derivative', color='red', alpha=0.2)
    ax2_secondary.set_ylabel('Second Derivative\n(Slope of Slope)', color='red')
    ax2_secondary.tick_params(axis='y', labelcolor='red')
    ax2_secondary.set_ylim(-0.06,0.06)
    ax2_secondary.legend(loc='upper right')

    if latest_datetime.time() > threshold_time:
        y = forest_day_df['second_derivative'].values
        N = len(y)
        fft_y = np.fft.fft(y)
        fft_y_filtered = np.zeros_like(fft_y)
        fft_y_filtered[:num_components] = fft_y[:num_components]
        fft_y_filtered[-(num_components-1):] = fft_y[-(num_components-1):]
        y_reconstructed = np.fft.ifft(fft_y_filtered)
        y_reconstructed = np.real(y_reconstructed)
        ax2_secondary.plot(day_df_plot['datetime'], y_reconstructed, 'r-', label='Reconstructed Signal (5 FFT components)')
        from scipy.signal import argrelextrema
        minima_indices = argrelextrema(y_reconstructed, np.less)[0]
        dates = day_df_plot['datetime']
        ax2_secondary.plot(dates.iloc[minima_indices], y_reconstructed[minima_indices], 
                           'o', markersize=10, markerfacecolor='none', markeredgecolor='black', 
                           label='Local Minima')
    
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    if config.restrict_830_to_300:
        ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax3.set_xlim(pd.to_datetime(f"{date_str} 08:30:00"), pd.to_datetime(f"{date_str} 15:00:00"))
    else:
        ax3.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax3.set_xlim(pd.to_datetime(f"{date_str} 03:30:00"), pd.to_datetime(f"{date_str} 19:00:00"))
        market_open = pd.to_datetime(f"{date_str} 08:30:00")
        market_close = pd.to_datetime(f"{date_str} 15:00:00")
        for ax in (ax1, ax3):
            ax.axvline(market_open, color='green', linestyle='-', linewidth=2, label='Market Open 8:30')
            ax.axvline(market_close, color='red', linestyle='-', linewidth=2, label='Market Close 15:00')
    fig.autofmt_xdate()
    plt.tight_layout()
    if config.save_plots:
        os.makedirs(config.output_dir, exist_ok=True)
        fname = os.path.join(config.output_dir, f"close_{date_str}.png")
        plt.savefig(fname)
        plt.close(fig)
    elif config.show_plots:
        plt.show()

    logging.debug("Finished plotting")
    if config.highlight_buy_predictions:
        recent_time = get_recent_purple_start_longer_than_3_minutes(day_df_plot, rand_predicted_classes)
    else:
        recent_time = None
    if recent_time:
        print(f"[{date_str}]: Most recent blue section (long run) starts at: {recent_time.strftime('%H:%M')}")
        return recent_time.strftime('%H:%M'), extra_in_window, in_green
    else:
        return None, extra_in_window, in_green

# -----------------------------------------------------------------------------
# Statistics Collection & Feature Normalization
# -----------------------------------------------------------------------------
def compute_daily_stats(day_df, date_str, slope_source='close', slope_lookback=10, slope_diff_threshold=0.025):
    if not pd.api.types.is_datetime64_any_dtype(day_df['time']):
        day_df['time'] = pd.to_datetime(day_df['time'], format="%H:%M:%S")
    from numpy import polyfit, arange
    def slope_of_best_fit(values):
        x = arange(len(values))
        y = values
        slope, _ = polyfit(x, y, 1)
        return slope
    series_for_slope = day_df[slope_source]
    day_df['slope_10'] = series_for_slope.rolling(slope_lookback).apply(slope_of_best_fit, raw=True)
    daily_min_slope = day_df['slope_10'].min(skipna=True)
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
                if threshold is None or arr[i] > threshold:
                    maxima_indices.append(i)
        return maxima_indices
    local_min_indices = find_local_minima(day_df['slope_10'], threshold=-0.09)
    maxima_all = find_local_maxima(day_df['slope_10'], threshold=0.03)
    max_set = set(maxima_all)
    time_intervals = []
    for min_idx in local_min_indices:
        slope_min = day_df['slope_10'].iloc[min_idx]
        next_max_idx = None
        for forward_i in range(min_idx + 1, len(day_df) - 1):
            if forward_i in max_set:
                slope_max = day_df['slope_10'].iloc[forward_i]
                if slope_max - slope_min >= slope_diff_threshold:
                    next_max_idx = forward_i
                    break
        if next_max_idx is not None:
            t_min = day_df.loc[min_idx, 'time']
            t_max = day_df.loc[next_max_idx, 'time']
            delta_minutes = (t_max - t_min).total_seconds() / 60.0
            time_intervals.append(delta_minutes)
    avg_interval = np.mean(time_intervals) if time_intervals else None
    stats = {
        "date": date_str,
        "daily_min_slope": daily_min_slope,
        "min_to_max_intervals": time_intervals,
        "avg_min_to_max_interval": avg_interval
    }
    return stats

def prepare_normalized_features(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
    # Note: Adjust norm_time as needed. Here we use the PlotConfig default.
    # norm_time = pd.to_datetime("08:30", format='%H:%M').time()
    norm_time = config.norm_time
    norm_cols = ['close'] + [col for col in df.columns if col.startswith('SMA_')]
    def normalize_group(group):
        base_rows = group[group['time'] == norm_time]
        if base_rows.empty:
            print(f"Warning: No row found at time '{norm_time}' for date {group.name}. Skipping normalization for this day.")
            return group
        base_value = base_rows.iloc[0]['close']
        if base_value == 0:
            print(f"Warning: Close value at {norm_time} is 0 for date {group.name}. Skipping normalization for this day.")
            return group
        group.loc[:, norm_cols] = group.loc[:, norm_cols] / base_value
        for col in ['high', 'low']:
            if col in group.columns and col not in norm_cols:
                group[col] = group[col] / base_value
        return group
    df_normalized = df.groupby('date', group_keys=False).apply(normalize_group)
    return df_normalized

def get_recent_purple_start_longer_than_3_minutes(day_df_plot, predicted_classes):
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
            start_time = day_df_plot.iloc[start]['datetime']
            end_time = day_df_plot.iloc[end]['datetime']
            
            # Check if the run lasts longer than some number of minutes.
            if end_time - start_time > pd.Timedelta(minutes=3):
                purple_runs.append((start, end))
        else:
            i += 1

    if not purple_runs:
        return None

    # --- Retrieve the Start Times of the Qualified Purple Runs ---
    purple_start_times = [day_df_plot.iloc[start]['datetime'] for start, _ in purple_runs]
    
    # --- Return the Most Recent (Latest) Start Time ---
    most_recent_timestamp = max(purple_start_times)
    return most_recent_timestamp

# -----------------------------------------------------------------------------
# Master Analysis Function & Monitor Loop
# -----------------------------------------------------------------------------
def run_analysis(symbol="SPY", start_date="2024-01-01", end_date="2024-01-31",
                 plot_config=None, collect_stats: bool = True,
                 use_yahoo: bool = False, use_sql=True):
    recent_time = None
    # Use default plot configuration if none provided
    if plot_config is None:
        plot_config = PlotConfig(plot_close=True)
        
    # Determine repository root for local data access; exit if not inside a Git repo
    repo_root = get_git_repo_root()
    if not repo_root:
        print("Not inside a Git repository. Exiting.")
        return

    logging.debug("Load and sort data")
    # Load stock data from the appropriate source (SQL, Yahoo, or API)
    df = load_and_sort_stock_data(symbol, repo_root, use_yahoo=use_yahoo,
                                  start_date=start_date, end_date=end_date, use_sql=use_sql)
    if df is None:
        return None

    # Make a deep copy to preserve original data for return
    ret_df = df.copy(deep=True)
    # Generate a date range for each day between the start and end dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    all_stats = []

    # Process data day-by-day
    for single_date in date_range:
        date_str = single_date.strftime('%Y-%m-%d')
        # Define the epoch range based on whether volume plotting is enabled
        if plot_config.plot_volume:
            start_epoch = int(convert_to_epoch(date_str, "08:30:00"))
            end_epoch   = int(convert_to_epoch(date_str, "15:00:00"))
        else:
            start_epoch = int(convert_to_epoch(date_str, "00:00:00"))
            end_epoch   = int(convert_to_epoch(date_str, "23:59:59"))
        # Filter the DataFrame to include only data within the specified epoch range
        selected_df = filter_epoch_range(df, start_epoch, end_epoch)
        if selected_df.empty:
            continue
        logging.info(f"Processing {date_str}")

        # Plot volume chart if enabled
        if plot_config.plot_volume:
            _ = plot_volume_chart_for_day(selected_df, date_str, plot_config)
        # Plot close chart and capture any prediction timing info if enabled
        if plot_config.plot_close:
            recent_time, in_purple, in_green = plot_close_chart_for_day(selected_df, date_str, plot_config)
        # Collect daily statistics if requested
        if collect_stats:
            day_stats = compute_daily_stats(selected_df, date_str)
            all_stats.append(day_stats)

    stats_df = pd.DataFrame(all_stats)
    return ret_df, recent_time, in_purple, in_green


def start_monitoring(symbol="SPY", minutes_interval=15, config: PlotConfig = None):
    iteration_count = -1
    known_minima = set()        # Track discovered local minima indices
    current_minima = 0
    last_recent_time = None
    threshold_time = datetime.time(14, 0)  # Monitoring stops after 15:00
    target_start = datetime.time(9, 5)    # Data collection starts just after market open
    target_wakeup = datetime.time(9, 5)  # Wake-up time slightly after start
    today = datetime.date.today()
    # Create datetime objects for target start and wake-up times
    start_dt = datetime.datetime.combine(today, target_start)
    wakeup_dt = datetime.datetime.combine(today, target_wakeup)
    last_in_purple = False
    in_purple = False

    logging.debug("Entering loop")
    while True:
        iteration_count += 1
        formatted_today = today.strftime("%Y-%m-%d")
        tomorrow = today + datetime.timedelta(days=1)
        formatted_tomorrow = tomorrow.strftime("%Y-%m-%d")
        now_time = datetime.datetime.now().time()
        today = datetime.date.today()

        if today.weekday() >= 5:
            exit()
        # Stop monitoring if current time is past the threshold
        if now_time > threshold_time or now_time.hour < 8:
            seconds_to_sleep = 1800
            now_dt = datetime.datetime.now()
            print(f"=========== [{now_dt.time().strftime('%H:%M:%S')}] Sleeping 30 minutes ===========")
            time.sleep(seconds_to_sleep)
            continue

        now_dt = datetime.datetime.now()
        # Wait until the target start time if current time is earlier
        start_dt = datetime.datetime.combine(today, target_start)
        if now_dt < start_dt:
            seconds_to_sleep = (wakeup_dt - now_dt).total_seconds()
            if seconds_to_sleep < 0:
                seconds_to_sleep = 600
            print(f"=========== [{now_dt.time().strftime('%H:%M:%S')}] Waiting until {target_wakeup.strftime('%H:%M')} ===========")
            time.sleep(seconds_to_sleep)
            now_dt = datetime.datetime.now()

        print(f"=========== [{now_time.strftime('%H:%M:%S')}] Fetching Stock Data ===========")
        try:
            # Run the analysis for today's data (from today to tomorrow)
            df, recent_time, in_purple, in_green = run_analysis(
                symbol="SPY",
                start_date=formatted_today,
                end_date=formatted_tomorrow,
                plot_config=config,
                collect_stats=True,
                use_yahoo=False,
                use_sql=False
            )
            logging.debug("Got stock data")
        except:
            # If an error occurs, wait 5 minutes and try again
            time.sleep(300)
            continue

        if df is None:
            # If no data was returned, wait a minute and try again
            time.sleep(60)
            continue

        # Determine which column to use for slope calculations
        if config.slope_source in df.columns:
            slope_series = df[config.slope_source]
        elif config.slope_source.lower() == "close":
            slope_series = df['close']
        else:
            print(f"Warning: slope_source '{config.slope_source}' not recognized; using 'close' by default.")
            slope_series = df['close']

        # Calculate rolling slope using the specified lookback period
        df['slope_10'] = slope_series.rolling(config.slope_lookback).apply(slope_of_best_fit, raw=True)
        # Identify indices where the slope meets the local minimum criteria
        local_min_indices = find_local_minima(df['slope_10'], threshold=-0.09)
        if len(local_min_indices) > 0:
            current_idx = max(local_min_indices)
            current_time = df["epoch_time"].iloc[max(local_min_indices)]

        logging.debug("Checking events")
        # Trigger an event if the purple region status has changed
        if last_in_purple != in_purple:
            last_in_purple = in_purple
            if in_purple:
                purp_str = f"Start of purple region {now_dt.strftime('%H:%M')}"
                on_event_message(purp_str)
            # else:
            #     purp_str = f"End of purple region {now_dt.strftime('%H:%M')}"
            # on_event_message(purp_str)
        # Trigger event if a new recent prediction time is detected
        elif recent_time != last_recent_time:
            last_recent_time = recent_time
            on_new_local_minimum_found(current_idx, df["close"].iloc[current_idx], df)
        # Otherwise, trigger an interval event at the specified minutes_interval
        elif now_time.minute % minutes_interval == 0:
            on_interval_elapsed(df)
        elif (now_time.minute % 5 == 0) and in_green:
            on_event_message("In green region")
        elif (now_time.minute % 5 == 0) and in_purple:
            on_event_message("In purple region")
        
        # if now_time > datetime.time(12, 0):
        #     minutes_interval = 15

        # Calculate delay until the start of the next minute and sleep for that duration
        now = datetime.datetime.now()
        next_minute = (now + datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
        delay = (next_minute - now).total_seconds()
        logging.debug(f"Sleeping for {delay:.2f} seconds until the start of the minute...")
        time.sleep(delay)


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)
    
    model_path = "other_analysis/high_buy_random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        rf_classifier = pickle.load(f)
    model_path = "other_analysis/neg_slope_model.pkl"
    with open(model_path, 'rb') as f:
        neg_rf_classifier = pickle.load(f)
    model_path = "other_analysis/pos_slope_model.pkl"
    with open(model_path, 'rb') as f:
        pos_rf_classifier = pickle.load(f)
    model_path = "other_analysis/low_close_random_forest_model.pkl"
    with open(model_path, 'rb') as f:
        down_rf_classifier = pickle.load(f)
    config = PlotConfig(
        plot_volume=False,
        plot_close=True,
        add_rsi=True,
        add_moving_averages=True,
        ma_windows=(15, 25, 100),
        restrict_830_to_300=True,
        highlight_negative_slope_runs=False,
        highlight_below_ma_runs=True,
        ma_run_len = 35,
        highlight_buy_predictions=False,
        use_down_models=False,
        find_local_minima_in_slope=True,
        high_buy_overlay_min_width=2,
        save_plots=True,
        slope_source='close',
        slope_lookback=10,
        use_neg_pos_models=True,
        rand_forest_class=rf_classifier,
        neg_classifier=neg_rf_classifier,
        pos_classifier=pos_rf_classifier,
        down_classifier=down_rf_classifier,
        # norm_time = pd.to_datetime("06:50:00").time()
    )
    # flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    # creds = flow.run_local_server(port=0)
    # with open("token.json", "w") as token:
    #     token.write(creds.to_json())
    start_monitoring(config=config, minutes_interval=10)

    # on_event_message("test")

    # df, _, _, _ = run_analysis(
    #     symbol="SPY",
    #     # start_date="2023-01-01",
    #     start_date="2025-06-16",
    #     # end_date="2025-04-02",
    #     end_date="2025-06-20",
    #     plot_config=config,
    #     collect_stats=True,
    #     use_yahoo=False,
    #     use_sql=True
    # )
    