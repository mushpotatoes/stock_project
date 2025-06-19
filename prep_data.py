import os
import datetime
import pickle
import sqlite3
import logging
import warnings
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# Import for slope calculation and scaling
from sklearn.preprocessing import StandardScaler
import joblib

# Assuming this is a custom helper module, ensure it's available
# from trading_helpers import get_git_repo_root # Commenting out as I don't have this module

# Dummy get_git_repo_root for demonstration purposes if trading_helpers is not available
def get_git_repo_root():
    """
    Dummy function to simulate getting the Git repository root.
    In a real environment, this would resolve the actual root path.
    For this example, it returns the current working directory.
    """
    return os.getcwd()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
log_filename = 'prep_data.log'
central_timezone = pytz.timezone('US/Central')
DEFAULT_NORM_TIME = datetime.time(8, 30) # 08:30 AM
DEFAULT_TIMEFRAMES = [10, 15, 25, 40, 90, 100, 120] # Default timeframes for slope calculation
# New: Default timeframes for SMA calculation for feature engineering (on 'close')
DEFAULT_SMA_TIMEFRAMES = [10, 25, 40, 90, 100, 120]
# Specific SMA timeframes for plotting (on 'original_close')
PLOT_SMA_TIMEFRAMES = [15, 25, 100]

# -----------------------------------------------------------------------------
# Logging & Warnings Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(log_filename, 'a'), # Uncomment to log to a file
        logging.StreamHandler() # Log to console
    ]
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class StockDataLoader:
    """
    A class to load stock data from various sources (Yahoo Finance, Polygon, SQLite).
    """
    def __init__(self, symbol, repo_root, start_date=None, end_date=None):
        self.symbol = symbol
        self.repo_root = repo_root
        # Set default dates if not provided
        self.start_date = start_date if start_date else "2020-01-01"
        self.end_date = end_date if end_date else datetime.datetime.today().strftime('%Y-%m-%d')
        self.central_timezone = pytz.timezone('US/Central')

    def _download_yahoo_finance_data(self):
        """
        Downloads historical stock data from Yahoo Finance.
        Handles data renaming, epoch time, and time string conversion.
        """
        try:
            import yfinance as yf
            data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval="1m", # 1-minute interval data
                prepost=True    # Include pre and post market data
            )
            if data.empty:
                logging.warning(f"No Yahoo Finance data retrieved for {self.symbol} from {self.start_date} to {self.end_date}.")
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
            # Adjust 'date' for Yahoo Finance timestamp discrepancies (often off by an hour)
            data['date'] = data['date'] - pd.Timedelta(hours=1)
            data['epoch_time'] = data['date'].apply(lambda dt: dt.timestamp())
            data['time'] = data['date'].dt.strftime("%H:%M:%S")
            data['close'] = data['close'].round(2)
            return data

        except Exception as e:
            logging.error(f"Failed to download Yahoo Finance data for {self.symbol}: {e}")
            return None

    def _load_sql_data(self):
        """
        Loads stock data from a SQLite database, filtered by date.
        Assumes the database contains tables with an 'epoch_time' column.
        """
        db_path = os.path.join(self.repo_root, f"big_{self.symbol}_data.db")
        if not os.path.exists(db_path):
            logging.warning(f"SQLite database not found at {db_path}")
            return pd.DataFrame()

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_names = [row[0] for row in cursor.fetchall()]
            df_list = []

            # Convert start_date and end_date to timestamps for SQL comparison
            # Ensure start_date and end_date are parsed as dates only to avoid time component issues
            start_timestamp = pd.to_datetime(self.start_date).timestamp()
            # Add a day's worth of seconds to end_timestamp to include data up to the end of the end_date
            end_timestamp = pd.to_datetime(self.end_date).timestamp() + (24 * 60 * 60) 

            for table_name in table_names:
                # Assuming 'epoch_time' is stored in the database
                query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE epoch_time >= {start_timestamp} AND epoch_time <= {end_timestamp}
                """
                temp_df = pd.read_sql_query(query, conn)
                if not temp_df.empty:
                    df_list.append(temp_df)
            conn.close()

            if not df_list:
                logging.warning(f"No tables or no data within the specified date range found in SQLite database {db_path}")
                return pd.DataFrame()

            # Concatenate all dataframes from the list
            df = pd.concat(df_list, ignore_index=True)
            return df
        except Exception as e:
            logging.error(f"Failed to load data from SQLite database {db_path}: {e}")
            return None

    def _download_polygon_data(self):
        """
        Downloads historical stock data from Polygon.io.
        Requires an 'API_KEY' environment variable.
        """
        api_key = os.environ.get('API_KEY')
        if not api_key:
            logging.error("Polygon API key not found in environment variable 'API_KEY'.")
            return pd.DataFrame()

        limit = 5000 # Max limit per request
        url = f"https://api.polygon.io/v2/aggs/ticker/{self.symbol}/range/1/minute/{self.start_date}/{self.end_date}"
        url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            response_json = response.json()

            if "results" not in response_json or not response_json["results"]:
                logging.warning(f"No Polygon data retrieved for {self.symbol} from {self.start_date} to {self.end_date}.")
                return pd.DataFrame()

            results = response_json["results"]
            new_data = pd.DataFrame({
                'epoch_time': [item['t'] / 1000.0 for item in results], # Convert ms to seconds
                'date': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(self.central_timezone).strftime("%Y-%m-%d") for item in results],
                'time': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(self.central_timezone).strftime("%H:%M:%S") for item in results],
                'open': [item['o'] for item in results],
                'high': [item['h'] for item in results],
                'low': [item['l'] for item in results],
                'close': [round(item['c'], 2) for item in results],
                'volume': [item['v'] for item in results]
            })
            return new_data

        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download Polygon data for {self.symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"An unexpected error occurred while processing Polygon data for {self.symbol}: {e}")
            return pd.DataFrame()

    def load_data(self, use_yahoo=False, use_sql=True):
        """
        Loads stock data based on the specified source (Yahoo Finance, SQLite, or Polygon).

        Args:
            use_yahoo (bool): If True, attempts to use Yahoo Finance.
            use_sql (bool): If True, attempts to use SQLite. If both use_yahoo and use_sql are False,
                            Polygon.io will be used as a fallback.

        Returns:
            pd.DataFrame: The loaded and sorted stock data, or an empty DataFrame/None on failure.
        """
        data = pd.DataFrame()
        if use_yahoo:
            logging.info(f"Loading data for {self.symbol} from Yahoo Finance...")
            data = self._download_yahoo_finance_data()
        elif use_sql:
            logging.info(f"Loading data for {self.symbol} from SQLite...")
            data = self._load_sql_data()
        else:
            logging.info(f"Loading data for {self.symbol} from Polygon.io...")
            data = self._download_polygon_data()

        if data is None or data.empty:
            logging.error(f"No data loaded for {self.symbol}.")
            return None

        # Sort data by epoch time and reset index for consistent processing
        data.sort_values(by='epoch_time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

class StockDataProcessor:
    """
    A class to process and analyze stock data, and make predictions using a Random Forest model.
    It handles feature engineering, normalization, standardization, and model inference.
    """
    def __init__(self, dataframe, repo_root=None, model_path=None, feature_columns=None):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = dataframe.copy(deep=True) # Work on a copy to avoid modifying original input DataFrame
        # Store original close prices before any normalization or modifications for plotting
        self.df['original_close'] = dataframe['close'].copy() 

        self.repo_root = repo_root # Base directory for saving/loading scalers
        self.model = None
        self.model_path = model_path
        self.feature_columns = feature_columns if feature_columns is not None else []

    @staticmethod
    def slope_of_best_fit(values):
        """
        Calculates the slope of the best-fit line for a given set of values using linear regression.
        Returns NaN if there are fewer than 2 valid (non-NaN) values.
        """
        if len(values) < 2 or pd.Series(values).isnull().all():
            return np.nan
        x = np.arange(len(values))
        y = values
        valid_indices = ~np.isnan(y)
        if not np.any(valid_indices): # Check if any non-NaN values exist
            return np.nan
        
        try:
            # Use numpy.polyfit to calculate the slope of a 1st-degree polynomial (line)
            slope, _ = np.polyfit(x[valid_indices], y[valid_indices], 1)
            return slope
        except np.linalg.LinAlgError:
            # Handle cases where linear algebra might fail (e.g., all x values are the same)
            return np.nan

    def _add_sma_features(self, timeframes=DEFAULT_SMA_TIMEFRAMES):
        """
        Adds Simple Moving Average (SMA) features to the DataFrame.
        Each SMA is calculated per day based on the 'close' column.
        These SMAs are intended for use as features in the model.

        Args:
            timeframes (list): A list of integer timeframes (window sizes) for SMA calculation.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot compute SMA features for model.")
            return

        # Ensure 'date' column is in datetime.date format for correct grouping
        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        
        if 'close' not in self.df.columns:
            logging.error("'close' column not found in DataFrame. Cannot compute SMAs for model features.")
            return

        logging.info("Starting SMA feature computation (for model features on 'close')...")
        for timeframe in timeframes:
            col_name = f'SMA_{timeframe}'
            logging.info(f"Calculating {col_name}...")
            # Group by date and apply rolling mean to 'close' price
            self.df[col_name] = self.df.groupby('date')['close'].transform(
                lambda x: x.rolling(window=timeframe, min_periods=1).mean()
            )
        logging.info("Completed SMA feature computation (for model features).")

    def _add_plotting_sma_features(self, timeframes=PLOT_SMA_TIMEFRAMES):
        """
        Adds Simple Moving Average (SMA) features to the DataFrame, specifically for plotting.
        These SMAs are calculated based on the 'original_close' column to represent unmodified data.

        Args:
            timeframes (list): A list of integer timeframes (window sizes) for SMA calculation for plotting.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot compute plotting SMA features.")
            return

        # Ensure 'date' column is in datetime.date format for correct grouping
        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        
        if 'original_close' not in self.df.columns:
            logging.error("'original_close' column not found in DataFrame. Cannot compute plotting SMAs.")
            return

        logging.info("Starting SMA feature computation (for plotting on 'original_close')...")
        for timeframe in timeframes:
            col_name = f'SMA_{timeframe}_plot' # Use a distinct suffix for plotting SMAs
            logging.info(f"Calculating {col_name}...")
            # Group by date and apply rolling mean to 'original_close' price
            self.df[col_name] = self.df.groupby('date')['original_close'].transform(
                lambda x: x.rolling(window=timeframe, min_periods=1).mean()
            )
        logging.info("Completed SMA feature computation (for plotting).")


    def _compute_daily_slopes(self, group, slope_column, lookback):
        """
        Computes the slope of the best-fit line for a given column within a daily group.
        This is a private helper method for internal use, typically applied via groupby.
        """
        group = group.sort_values(by='time') # Ensure chronological order for rolling window
        group[f'slope_{lookback}'] = group[slope_column].rolling(
            window=lookback, min_periods=1
        ).apply(self.slope_of_best_fit, raw=True) # Apply the static slope calculation
        return group

    def _add_slope_difference(self, df_in, slope_lookback):
        """
        Adds a column representing the difference from a previous slope.
        This helps capture the rate of change of the slope.
        """
        df = df_in.copy()
        slope_col = f'slope_{slope_lookback}'
        diff_col = f'd2_{slope_lookback}' # Name for the second derivative (difference of slopes)

        if slope_col in df.columns:
            if 'date' in df.columns:
                # Calculate difference within each daily group
                df[diff_col] = df.groupby('date')[slope_col].diff()
            else:
                # Calculate difference across the entire DataFrame if no date column
                df[diff_col] = df[slope_col].diff()
        else:
            logging.warning(f"Slope column '{slope_col}' not found for calculating difference. Setting '{diff_col}' to NaN.")
            df[diff_col] = np.nan

        return df

    def _add_slope_run_length(self, df_in, slope_column='slope_10'):
        """
        Adds columns for consecutive positive and negative slope run lengths.
        A run length indicates how many consecutive periods a condition (positive/negative slope) holds.
        """
        df = df_in.copy()
        
        if slope_column not in df.columns:
            logging.warning(f"Slope column '{slope_column}' not found. Cannot compute run lengths.")
            df['negative_slope_run_length'] = np.nan
            df['positive_slope_run_length'] = np.nan
            return df

        def compute_run_length_for_group(group):
            """Helper to compute run lengths within a single daily group."""
            group = group.sort_values(by='time') # Ensure data is sorted by time

            # Calculate negative slope run length:
            # Create a boolean mask where slope is negative.
            mask_negative = group[slope_column] < 0
            # Use cumsum to identify blocks of consecutive true/false values.
            # cumcount + 1 gives the count within each block.
            group['negative_slope_run_length'] = mask_negative.groupby((~mask_negative).cumsum()).cumcount() + 1
            # Reset run length to 0 where the condition is false (slope is not negative)
            group.loc[~mask_negative, 'negative_slope_run_length'] = 0

            # Calculate positive slope run length (similar logic)
            mask_positive = group[slope_column] > 0
            group['positive_slope_run_length'] = mask_positive.groupby((~mask_positive).cumsum()).cumcount() + 1
            group.loc[~mask_positive, 'positive_slope_run_length'] = 0
            
            return group

        # Apply the run length computation to each daily group
        df_with_runs = df.groupby('date', group_keys=False).apply(compute_run_length_for_group)
        return df_with_runs

    def _add_sma_run_length(self, df_in, sma_short_col, sma_long_col, new_col):
        """
        Adds a column for the run length where a short-term SMA is below a long-term SMA.
        This is a common technical analysis signal.
        """
        df = df_in.copy()

        if sma_short_col not in df.columns or sma_long_col not in df.columns:
            logging.warning(f"Required SMA columns '{sma_short_col}' or '{sma_long_col}' not found. Cannot compute SMA run length for '{new_col}'.")
            df[new_col] = np.nan
            return df

        def compute_run_length_for_group(group):
            """Helper to compute SMA run length within a single daily group."""
            group = group.sort_values(by='time') # Ensure data is sorted by time
            condition = group[sma_short_col] < group[sma_long_col]
            group[new_col] = condition.groupby((~condition).cumsum()).cumcount() + 1
            group.loc[~condition, new_col] = 0
            return group

        # Apply the run length computation to each daily group
        df_with_run = df.groupby('date', group_keys=False).apply(compute_run_length_for_group)
        return df_with_run

    def add_run_length_features(self, slope_column='slope_10', sma_cross_params=None):
        """
        Adds various run length features to the DataFrame, including slope run lengths
        and SMA cross-over run lengths.

        Args:
            slope_column (str): The name of the slope column to use for positive/negative slope run lengths.
            sma_cross_params (list of dict): Optional list of dictionaries defining parameters
                                              for SMA cross-over run length features. Each dict should
                                              contain 'sma_short', 'sma_long', and 'new_col_name'.
        """
        logging.info("Adding run length features...")
        
        # Add positive and negative slope run lengths
        self.df = self._add_slope_run_length(self.df, slope_column=slope_column)
        logging.info(f"Added slope run length features based on '{slope_column}'.")

        # Add SMA cross-over run lengths if parameters are provided
        if sma_cross_params:
            for params in sma_cross_params:
                sma_short = params.get('sma_short')
                sma_long = params.get('sma_long')
                new_col_name = params.get('new_col_name')
                
                if sma_short and sma_long and new_col_name:
                    self.df = self._add_sma_run_length(
                        self.df,
                        sma_short_col=sma_short,
                        sma_long_col=sma_long,
                        new_col=new_col_name
                    )
                    logging.info(f"Added SMA run length feature '{new_col_name}'.")
                else:
                    logging.warning(f"Invalid SMA cross parameters: {params}. Skipping this SMA run length feature.")
        else:
            logging.info("No SMA cross parameters provided. Skipping SMA run length features.")

        logging.info("Completed adding run length features.")


    def calculate_daily_stats(self):
        """
        Calculates basic daily statistics (open, close, high, low, volume)
        for each day in the DataFrame.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot calculate daily stats.")
            return []

        all_stats = []
        # Ensure 'date' column is in datetime format for grouping
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Group by date and calculate summary statistics
        for date, group in self.df.groupby(self.df['date'].dt.date):
            date_str = date.strftime('%Y-%m-%d')
            # Safely get first/last/max/min values, handling empty groups if any
            daily_open = group['open'].iloc[0] if not group.empty else np.nan
            daily_close = group['close'].iloc[-1] if not group.empty else np.nan
            daily_high = group['high'].max() if not group.empty else np.nan
            daily_low = group['low'].min() if not group.empty else np.nan
            daily_volume = group['volume'].sum() if not group.empty else np.nan

            stats = {
                'date': date_str,
                'open': daily_open,
                'close': daily_close,
                'high': daily_high,
                'low': daily_low,
                'volume': daily_volume
            }
            all_stats.append(stats)
            logging.debug(f"Processed daily stats for {date_str}")
        return all_stats

    def prepare_normalized_features(self, norm_time=DEFAULT_NORM_TIME):
        """
        Prepares normalized features by dividing 'close', 'open', 'high', 'low',
        and selected SMA columns by the 'close' price at a specific normalization time
        for each day. This helps in creating scale-invariant features.
        Days where normalization cannot be performed (e.g., no data at norm_time) are excluded.

        Args:
            norm_time (datetime.time): The time of day to use as the base for normalization.
                                        Defaults to DEFAULT_NORM_TIME (8:30 AM).
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot prepare normalized features.")
            return

        # Ensure 'time' column is datetime.time object for accurate comparison
        if 'time' in self.df.columns and not isinstance(self.df['time'].iloc[0], datetime.time):
            self.df['time'] = pd.to_datetime(self.df['time'], format='%H:%M:%S').dt.time

        # Ensure 'date' column is datetime.date object for grouping
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
             self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d').dt.date

        # Identify columns to normalize: 'close', 'open', 'high', 'low', and SMAs (excluding plotting SMAs)
        norm_cols = ['close', 'open', 'high', 'low'] + [col for col in self.df.columns if col.startswith('SMA_') and not col.endswith('_plot')]
        # Filter to only include columns that actually exist in the DataFrame
        norm_cols = [col for col in norm_cols if col in self.df.columns]

        def normalize_group(group):
            """Helper to normalize a single daily group, setting a flag if skipped."""
            # Add a temporary flag to indicate if normalization was successful for this day
            group['_normalization_skipped'] = False

            # Find the row corresponding to the normalization time
            base_rows = group.loc[group['time'] == norm_time]

            if base_rows.empty:
                logging.warning(f"No row found at time '{norm_time.strftime('%H:%M')}' for date {group.name}. Marking for exclusion.")
                group['_normalization_skipped'] = True # Mark this group to be skipped
                return group
            
            base_value = base_rows.iloc[0]['close'] # Get the close price at norm_time

            if base_value == 0:
                logging.warning(f"Close value at {norm_time.strftime('%H:%M')} is 0 for date {group.name}. Marking for exclusion to avoid division by zero.")
                group['_normalization_skipped'] = True # Mark this group to be skipped
                return group
            
            # Apply normalization to the identified columns
            group.loc[:, norm_cols] = group.loc[:, norm_cols] / base_value
            
            return group

        logging.info(f"Starting feature normalization using {norm_time.strftime('%H:%M')} as base...")
        # Apply normalization and flagging to each daily group
        self.df = self.df.groupby('date', group_keys=False).apply(normalize_group)

        # Filter out the days that were marked as skipped
        original_rows = len(self.df)
        self.df = self.df[self.df['_normalization_skipped'] == False].drop(columns=['_normalization_skipped'])
        skipped_rows = original_rows - len(self.df)
        
        if skipped_rows > 0:
            logging.info(f"Excluded {skipped_rows} rows from days where normalization could not be performed.")
        logging.info(f"Completed feature normalization. Remaining rows: {len(self.df)}")


    def _compute_slopes_for_timeframe_single(self, dataframe_day_group, timeframe, scalers_dir, use_saved_scaler=True):
        """
        Helper method to compute slopes and their second differences for a single timeframe
        on a daily group, and then standardize these computed features.
        This method is designed to be called by `apply` on groupby objects.
        """
        df_local = dataframe_day_group.copy()

        if 'date' in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local['date']):
            df_local['date'] = pd.to_datetime(df_local['date']).dt.date

        # 1. Compute daily slopes on the 'close' price
        df_local = self._compute_daily_slopes(df_local, 'close', timeframe)
        # 2. Add slope difference (second derivative)
        df_local = self._add_slope_difference(df_local, slope_lookback=timeframe)

        # Columns that were just added/computed
        new_cols_to_standardize = [f'slope_{timeframe}', f'd2_{timeframe}']
        
        # Filter to ensure we only try to standardize columns that exist
        existing_new_cols = [col for col in new_cols_to_standardize if col in df_local.columns]

        if not existing_new_cols:
            logging.warning(f"Slope and d2 columns ({new_cols_to_standardize}) not found for timeframe {timeframe} for date {dataframe_day_group.name}. Returning original group.")
            return dataframe_day_group # Return original if no new cols generated

        new_cols_df = df_local[existing_new_cols].copy()

        # Define path for the scaler file
        scaler_filename = os.path.join(scalers_dir, f"scaler_timeframe_{timeframe}.pkl")
        scaler = None

        # Load or fit StandardScaler
        if use_saved_scaler and os.path.exists(scaler_filename):
            try:
                scaler = joblib.load(scaler_filename)
                # Ensure columns for transformation match the scaler's features
                cols_for_transform = [col for col in scaler.feature_names_in_ if col in new_cols_df.columns]
                new_cols_standardized = scaler.transform(new_cols_df[cols_for_transform])
            except Exception as e:
                logging.error(f"Error loading or transforming with saved scaler for timeframe {timeframe}: {e}. Falling back to fitting a new scaler.")
                scaler = StandardScaler()
                new_cols_standardized = scaler.fit_transform(new_cols_df)
                joblib.dump(scaler, scaler_filename) # Save the newly fitted scaler
        else:
            logging.info(f"No saved scaler found or use_saved_scaler=False. Fitting a new scaler for timeframe {timeframe}.")
            scaler = StandardScaler()
            new_cols_standardized = scaler.fit_transform(new_cols_df)
            joblib.dump(scaler, scaler_filename) # Save the newly fitted scaler
        
        # Create a DataFrame for standardized columns
        # Remove '_norm' suffix if it was implicitly added by previous normalization steps
        standardized_col_names = [col.replace('_norm', '') for col in existing_new_cols]
        new_cols_df_standardized = pd.DataFrame(
            new_cols_standardized,
            columns=standardized_col_names,
            index=new_cols_df.index
        )
        
        # Merge the standardized columns back to the original daily group
        return dataframe_day_group.merge(new_cols_df_standardized, left_index=True, right_index=True, how='left')


    def compute_slopes_and_standardize(self, timeframes=DEFAULT_TIMEFRAMES, use_saved_scaler=True):
        """
        Computes slopes and their differences for a list of timeframes across the DataFrame,
        and standardizes these features using StandardScaler. Scalers are saved/loaded
        to/from a specified directory to ensure consistent scaling.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot compute slopes and standardize.")
            return

        if self.repo_root is None:
            logging.error("repo_root not set in StockDataProcessor. Cannot save/load scalers.")
            return

        scalers_dir = os.path.join(self.repo_root, "other_analysis", "scalers")
        os.makedirs(scalers_dir, exist_ok=True) # Ensure the directory exists
        logging.info(f"Scalers directory: {scalers_dir}")

        # Ensure 'date' column is in datetime.date format for grouping
        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date

        # Iterate through each specified timeframe to compute slopes and standardize
        for timeframe in timeframes:
            logging.info(f"Computing slopes and standardizing for timeframe: {timeframe}")
            
            # Apply the single-timeframe slope computation and standardization to each daily group
            temp_df_with_new_cols = self.df.groupby('date', group_keys=False).apply(
                lambda group: self._compute_slopes_for_timeframe_single(group, timeframe, scalers_dir, use_saved_scaler)
            )
            
            # Identify the newly added slope and d2 columns for the current timeframe
            newly_added_cols = [col for col in temp_df_with_new_cols.columns if col.startswith(f'slope_{timeframe}') or col.startswith(f'd2_{timeframe}')]

            if newly_added_cols:
                # Merge the newly computed and standardized columns back into the main DataFrame
                # Use 'epoch_time' for merging to ensure correct alignment
                self.df = self.df.merge(
                    temp_df_with_new_cols[['epoch_time'] + newly_added_cols],
                    on='epoch_time',
                    how='left'
                )
                
                # Drop potential duplicates that might arise from merging if epoch_time isn't strictly unique
                self.df.drop_duplicates(subset=['epoch_time'], inplace=True)
                self.df.reset_index(drop=True, inplace=True)

                logging.info(f"Added {len(newly_added_cols)} standardized columns for timeframe {timeframe}.")
            else:
                logging.warning(f"No standardized slope/d2 columns were generated for timeframe {timeframe}. Check for data issues or previous errors.")

        logging.info("Completed slope computation and standardization for all timeframes.")

    def load_model(self):
        """
        Loads the pre-trained Random Forest model from the specified path using pickle.
        """
        if not self.model_path:
            logging.error("Model path not set. Cannot load model.")
            return
        
        if not os.path.exists(self.model_path):
            logging.error(f"Model file not found at {self.model_path}")
            return

        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_path}: {e}")
            self.model = None

    def make_predictions(self, prediction_column_name='prediction_probability'):
        """
        Makes predictions using the loaded Random Forest model and adds the prediction
        probabilities (for the positive class) to a new column in the DataFrame.

        Args:
            prediction_column_name (str): The name for the new column containing prediction probabilities.
        """
        if self.model is None:
            logging.error("Model not loaded. Cannot make predictions.")
            self.df[prediction_column_name] = np.nan # Add NaN column if model isn't loaded
            return

        if not self.feature_columns:
            logging.warning("No feature columns specified for prediction. Skipping predictions.")
            self.df[prediction_column_name] = np.nan
            return

        # Check if all required feature columns exist in the DataFrame before predicting
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            logging.error(f"Missing required feature columns for prediction: {', '.join(missing_features)}. Cannot make predictions.")
            self.df[prediction_column_name] = np.nan
            return

        logging.info("Making predictions using the Random Forest model...")
        try:
            # Select only the features required by the model
            X = self.df[self.feature_columns]
            
            # Predict probabilities for the positive class (usually class 1)
            predictions = self.model.predict_proba(X)[:, 1]
            self.df[prediction_column_name] = predictions
            logging.info(f"Predictions added to column '{prediction_column_name}'.")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            self.df[prediction_column_name] = np.nan

    def get_dataframe(self):
        """
        Returns the current state of the processed DataFrame.
        """
        return self.df

def plot_daily_data(processed_df, start_plot_time, end_plot_time, symbol, save_dir="plots"):
    """
    Parses through the processed_df one day at a time and generates a plot for each day.
    Each plot has two subplots:
    1. Original close prices with 15, 25, and 100-minute Simple Moving Averages overlaid.
    2. Random Forest prediction probability values for the same time frame.

    Args:
        processed_df (pd.DataFrame): The DataFrame containing processed stock data,
                                     expected to include 'original_close', 'SMA_XX_plot' (e.g., SMA_15_plot),
                                     'rf_prediction_probability', 'date', and 'time' columns.
        start_plot_time (datetime.time): The start time (HH:MM:SS) for plotting data each day.
        end_plot_time (datetime.time): The end time (HH:MM:SS) for plotting data each day.
        symbol (str): The stock symbol (e.g., 'SPY') to be used in plot titles and filenames.
        save_dir (str): Directory where the generated plot images will be saved.
    """
    if processed_df.empty:
        logging.warning("Processed DataFrame is empty, no plots to generate.")
        return

    # Ensure 'date' is a datetime.date object for consistent grouping
    if 'date' in processed_df.columns and not pd.api.types.is_datetime64_any_dtype(processed_df['date']):
        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date
    
    # Ensure 'time' is a datetime.time object for accurate filtering
    if 'time' in processed_df.columns and not isinstance(processed_df['time'].iloc[0], datetime.time):
        processed_df['time'] = pd.to_datetime(processed_df['time'], format='%H:%M:%S').dt.time

    os.makedirs(save_dir, exist_ok=True) # Create the save directory if it doesn't exist
    logging.info(f"Plots will be saved in: {save_dir}")

    # Get unique dates from the DataFrame to iterate through each day
    unique_dates = processed_df['date'].unique()
    if len(unique_dates) == 0:
        logging.warning("No unique dates found in the DataFrame to plot.")
        return

    # Iterate over each unique date to generate a separate plot
    for date in unique_dates:
        date_str = date.strftime('%Y-%m-%d') # Format date for title and filename
        logging.info(f"Generating plot for date: {date_str}")

        # Filter data for the current day
        daily_data = processed_df[processed_df['date'] == date].copy()

        # Filter data within the specified plotting time range
        mask_plot = (daily_data['time'] >= start_plot_time) & \
                    (daily_data['time'] <= end_plot_time)
        daily_data_plot = daily_data.loc[mask_plot].copy()

        if daily_data_plot.empty:
            logging.warning(f"No data within plot time range ({start_plot_time}-{end_plot_time}) for {date_str}. Skipping plot.")
            continue

        # Combine date and time to create a single datetime column for the x-axis
        # This is essential for matplotlib to correctly plot time series
        daily_data_plot['datetime_plot'] = pd.to_datetime(
            daily_data_plot['date'].astype(str) + ' ' + daily_data_plot['time'].astype(str)
        )

        # Create a figure with two subplots, sharing the x-axis for consistent time alignment
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'{symbol} Daily Analysis - {date_str}', fontsize=16)

        # --- Subplot 1: Original Close Prices with SMAs ---
        ax1.plot(daily_data_plot['datetime_plot'], daily_data_plot['original_close'], 
                 label='Original Close', color='blue', linewidth=1.5, alpha=0.8)
        
        # Plot the specified SMAs on original close data
        for timeframe in PLOT_SMA_TIMEFRAMES:
            sma_col = f'SMA_{timeframe}_plot'
            if sma_col in daily_data_plot.columns:
                ax1.plot(daily_data_plot['datetime_plot'], daily_data_plot[sma_col], 
                         label=f'SMA {timeframe}min', linewidth=1, alpha=0.7)
            else:
                logging.warning(f"SMA column '{sma_col}' not found for plotting on {date_str}. Skipping this SMA.")

        ax1.set_ylabel('Price')
        ax1.set_title('Original Close Prices and Simple Moving Averages')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7) # Add a grid for readability

        # --- Subplot 2: RF Prediction Probability ---
        if 'rf_prediction_probability' in daily_data_plot.columns:
            ax2.plot(daily_data_plot['datetime_plot'], daily_data_plot['rf_prediction_probability'], 
                     label='RF Prediction Probability', color='red', linewidth=1.5, alpha=0.8)
            ax2.set_ylabel('Probability')
            ax2.set_title('Random Forest Prediction Probability')
            ax2.grid(True, linestyle='--', alpha=0.7)
            # Set y-axis limits for probability to be between 0 and 1
            ax2.set_ylim(0, 1) 
        else:
            logging.warning(f"'rf_prediction_probability' column not found for plotting on {date_str}. Plotting placeholder title.")
            ax2.set_title('RF Prediction Probability (Data Missing)')
            ax2.set_ylabel('Probability')


        # --- X-axis Formatting (shared for both subplots) ---
        xfmt = mdates.DateFormatter('%H:%M') # Format to show only hour and minute
        ax2.xaxis.set_major_formatter(xfmt)
        fig.autofmt_xdate() # Automatically format and rotate x-axis labels to prevent overlap
        ax2.set_xlabel('Time (HH:MM)')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap and provide space
        
        # Save the plot
        plot_filename = os.path.join(save_dir, f"{symbol}_{date_str}_analysis.png")
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free up memory after saving
        logging.info(f"Plot saved: {plot_filename}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    # Set the current working directory to the script's directory for relative paths
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir) 

    repo_root = get_git_repo_root()
    if not repo_root:
        logging.error("Not inside a Git repository or repo_root could not be determined. Exiting.")
        exit()

    global symbol # Make symbol global so plot_daily_data can access it directly
    symbol = "SPY"
    start_date = "2015-01-01" # Example start date for data loading
    end_date = "2025-06-13"   # Example end date for data loading

    use_sql = True # Set to True to try loading from SQLite, False to use Polygon

    # Initialize data loader and load raw data
    data_loader = StockDataLoader(
        symbol=symbol,
        repo_root=repo_root,
        start_date=start_date,
        end_date=end_date
    )

    df = data_loader.load_data(use_sql=use_sql, use_yahoo=False)
    if df is None:
        logging.error("Failed to load stock data. Exiting.")
        exit()

    logging.info(f"Successfully loaded {len(df)} records for {symbol}.")

    # Define the path to the pre-trained Random Forest model
    model_path = os.path.join(repo_root, "other_analysis", "pos_slope_model.pkl") 

    # Define the feature columns that the model expects
    selected_features = [
        "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
        "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
        "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120",
        "sma_25_below_100_run_length", "negative_slope_run_length", "positive_slope_run_length"
    ]

    # Initialize data processor
    data_processor = StockDataProcessor(
        df, # Pass the loaded DataFrame
        repo_root=repo_root,
        model_path=model_path,
        feature_columns=selected_features
    )

    # Add SMAs for model features (calculated on 'close' prices)
    data_processor._add_sma_features(timeframes=DEFAULT_SMA_TIMEFRAMES)
    # Add SMAs specifically for plotting (calculated on 'original_close' prices)
    data_processor._add_plotting_sma_features(timeframes=PLOT_SMA_TIMEFRAMES)
    
    # Normalize features (this will normalize the 'close' and SMA_ features for the model)
    # This method now also handles the exclusion of days where normalization fails.
    data_processor.prepare_normalized_features(norm_time=datetime.time(8, 30))
    
    logging.info("Starting slope computation and standardization...")
    # Compute slopes and standardize them for model features
    data_processor.compute_slopes_and_standardize(
        timeframes=DEFAULT_TIMEFRAMES,
        use_saved_scaler=True # Use saved scalers if available
    )
    logging.info("Slope computation and standardization complete.")

    # Define parameters for SMA run length features
    sma_run_length_params = [
        {'sma_short': 'SMA_25', 'sma_long': 'SMA_100', 'new_col_name': 'sma_25_below_100_run_length'},
    ]
    # Add various run length features
    data_processor.add_run_length_features(
        slope_column='slope_10', 
        sma_cross_params=sma_run_length_params
    )
    logging.info("Run length feature addition complete.")

    # Load the machine learning model and make predictions
    data_processor.load_model()
    data_processor.make_predictions(prediction_column_name='rf_prediction_probability')
    logging.info("Random Forest predictions added to DataFrame.")

    # Calculate and display daily statistics
    daily_stats = data_processor.calculate_daily_stats()
    if daily_stats:
        logging.info("Daily statistics calculated (possibly on normalized and standardized data):")
        for stat in daily_stats:
            logging.info(f"    Date: {stat['date']}, Close: {stat['close']:.4f}, Volume: {stat['volume']}")
    else:
        logging.info("No daily statistics to display.")

    # Get the final processed DataFrame
    processed_df = data_processor.get_dataframe()

    # Define plotting time window
    start_plot_time = pd.to_datetime("08:30:00").time()
    end_plot_time   = pd.to_datetime("15:00:00").time()
    
    # Display first few rows of the processed DataFrame (filtered for plot time range for display clarity)
    mask_display = (processed_df['time'] >= start_plot_time) & (processed_df['time'] <= end_plot_time)
    processed_df_for_display = processed_df.loc[mask_display].copy().reset_index(drop=True)

    file_path = "SPY_with_features.parquet"
    # --- Saving the DataFrame ---
    try:
        processed_df_for_display.to_parquet(file_path, engine='pyarrow', compression='snappy')
        print(f"DataFrame successfully saved to {file_path}")
    except ImportError:
        print("PyArrow or Fastparquet engine not found. Please install: pip install pyarrow")
    except Exception as e:
        print(f"Error saving DataFrame to Parquet: {e}")
    
    logging.info("\nFirst 20 rows of the processed (normalized and standardized) DataFrame with predictions:")
    print(processed_df_for_display.head(n=20))
    logging.info(f"\nColumns in final DataFrame: {processed_df_for_display.columns.tolist()}")

    # # Generate and save daily plots
    # plot_daily_data(processed_df, start_plot_time, end_plot_time, symbol, save_dir=os.path.join(repo_root, "daily_plots"))


if __name__ == "__main__":
    main()
