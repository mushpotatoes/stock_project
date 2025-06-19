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
from trading_helpers import get_git_repo_root

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
log_filename = 'prep_data.log'
central_timezone = pytz.timezone('US/Central')
DEFAULT_NORM_TIME = datetime.time(8, 30) # 08:30 AM
DEFAULT_TIMEFRAMES = [10, 15, 25, 40, 90, 100, 120] # Default timeframes for slope calculation
# New: Default timeframes for SMA calculation
DEFAULT_SMA_TIMEFRAMES = [10, 25, 40, 90, 100, 120]


# -----------------------------------------------------------------------------
# Logging & Warnings Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(log_filename, 'a'),
        logging.StreamHandler()
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
        self.start_date = start_date if start_date else "2020-01-01"
        self.end_date = end_date if end_date else datetime.datetime.today().strftime('%Y-%m-%d')
        self.central_timezone = pytz.timezone('US/Central')

    def _download_yahoo_finance_data(self):
        """
        Downloads historical stock data from Yahoo Finance.
        """
        try:
            import yfinance as yf
            data = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval="1m",
                prepost=True
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

            # Convert start_date and end_date to timestamps for comparison in SQL
            start_timestamp = pd.to_datetime(self.start_date).timestamp()
            end_timestamp = pd.to_datetime(self.end_date).timestamp()

            for table_name in table_names:
                # Assuming 'epoch_time' is stored in the database
                query = f"""
                    SELECT *
                    FROM {table_name}
                    WHERE epoch_time >= {start_timestamp} AND epoch_time <= {end_timestamp}
                """
                temp_df = pd.read_sql_query(query, conn)
                # FIX: Only append non-empty DataFrames
                if not temp_df.empty:
                    df_list.append(temp_df)
            conn.close()

            if not df_list:
                logging.warning(f"No tables or no data within the specified date range found in SQLite database {db_path}")
                return pd.DataFrame()

            # The concat operation will now only be on non-empty DataFrames
            df = pd.concat(df_list, ignore_index=True)
            return df
        except Exception as e:
            logging.error(f"Failed to load data from SQLite database {db_path}: {e}")
            return None

    def _download_polygon_data(self):
        """
        Downloads historical stock data from Polygon.io.
        """
        api_key = os.environ.get('API_KEY')
        if not api_key:
            logging.error("Polygon API key not found in environment variable 'API_KEY'.")
            return pd.DataFrame()

        limit = 5000
        url = f"https://api.polygon.io/v2/aggs/ticker/{self.symbol}/range/1/minute/{self.start_date}/{self.end_date}"
        url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            response_json = response.json()

            if "results" not in response_json or not response_json["results"]:
                logging.warning(f"No Polygon data retrieved for {self.symbol} from {self.start_date} to {self.end_date}.")
                return pd.DataFrame()

            results = response_json["results"]
            new_data = pd.DataFrame({
                'epoch_time': [item['t'] / 1000.0 for item in results],
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
        Loads stock data based on the specified source.

        Args:
            use_yahoo (bool): If True, use Yahoo Finance.
            use_sql (bool): If True, use SQLite. If False, and use_yahoo is False, use Polygon.

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

        data.sort_values(by='epoch_time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

class StockDataProcessor:
    """
    A class to process and analyze stock data, and make predictions using a Random Forest model.
    """
    def __init__(self, dataframe, repo_root=None, model_path=None, feature_columns=None):
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = dataframe.copy(deep=True) # Work on a copy
        self.repo_root = repo_root # Store repo_root for scaler directory
        self.model = None
        self.model_path = model_path
        self.feature_columns = feature_columns if feature_columns is not None else []

    @staticmethod
    def slope_of_best_fit(values):
        """
        Calculates the slope of the best-fit line for a given set of values.
        This is a static method as it does not depend on the instance state.
        """
        if len(values) < 2 or pd.Series(values).isnull().all():
            return np.nan
        x = np.arange(len(values))
        y = values
        valid_indices = ~np.isnan(y)
        if not np.any(valid_indices):
            return np.nan
        
        try:
            slope, _ = np.polyfit(x[valid_indices], y[valid_indices], 1)
            return slope
        except np.linalg.LinAlgError:
            return np.nan

    def _add_sma_features(self, timeframes=DEFAULT_SMA_TIMEFRAMES):
        """
        Adds Simple Moving Average (SMA) features to the DataFrame.
        Each SMA is calculated per day.

        Args:
            timeframes (list): A list of integer timeframes for SMA calculation.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot compute SMA features.")
            return

        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date
        
        # Ensure 'close' column exists for SMA calculation
        if 'close' not in self.df.columns:
            logging.error("'close' column not found in DataFrame. Cannot compute SMAs.")
            return

        logging.info("Starting SMA feature computation...")
        for timeframe in timeframes:
            col_name = f'SMA_{timeframe}'
            logging.info(f"Calculating {col_name}...")
            # Use transform to maintain the original DataFrame shape
            self.df[col_name] = self.df.groupby('date')['close'].transform(
                lambda x: x.rolling(window=timeframe, min_periods=1).mean()
            )
        logging.info("Completed SMA feature computation.")


    def _compute_daily_slopes(self, group, slope_column, lookback):
        """
        Computes the slope of the best-fit line for a given column within a daily group.
        This is a private helper method for internal use.
        """
        group = group.sort_values(by='time')
        group[f'slope_{lookback}'] = group[slope_column].rolling(
            window=lookback, min_periods=1
        ).apply(self.slope_of_best_fit, raw=True)
        return group

    def _add_slope_difference(self, df_in, slope_lookback):
        """
        Adds a column representing the difference from a previous slope.
        This is a private helper method for internal use.
        """
        df = df_in.copy()
        slope_col = f'slope_{slope_lookback}'
        diff_col = f'd2_{slope_lookback}'

        if slope_col in df.columns:
            if 'date' in df.columns:
                df[diff_col] = df.groupby('date')[slope_col].diff()
            else:
                df[diff_col] = df[slope_col].diff()
        else:
            df[diff_col] = np.nan

        return df

    def _add_slope_run_length(self, df_in, slope_column='slope_10'):
        """
        Adds columns for consecutive positive and negative slope run lengths.
        This is a private helper method for internal use.
        """
        df = df_in.copy()
        
        if slope_column not in df.columns:
            logging.warning(f"Slope column '{slope_column}' not found. Cannot compute run lengths.")
            df['negative_slope_run_length'] = np.nan
            df['positive_slope_run_length'] = np.nan
            return df

        def compute_run_length_for_group(group):
            # Ensure the group is sorted by time for accurate run length
            group = group.sort_values(by='time')

            # Negative slope run length
            mask_negative = group[slope_column] < 0
            group['negative_slope_run_length'] = mask_negative.groupby((~mask_negative).cumsum()).cumcount() + 1
            group.loc[~mask_negative, 'negative_slope_run_length'] = 0

            # Positive slope run length
            mask_positive = group[slope_column] > 0
            group['positive_slope_run_length'] = mask_positive.groupby((~mask_positive).cumsum()).cumcount() + 1
            group.loc[~mask_positive, 'positive_slope_run_length'] = 0
            
            return group

        # Apply to each date group
        df_with_runs = df.groupby('date', group_keys=False).apply(compute_run_length_for_group)
        return df_with_runs

    def _add_sma_run_length(self, df_in, sma_short_col, sma_long_col, new_col):
        """
        Adds a column for the run length where sma_short_col is below sma_long_col.
        This is a private helper method for internal use.
        """
        df = df_in.copy()

        # Check if SMA columns exist
        if sma_short_col not in df.columns or sma_long_col not in df.columns:
            logging.warning(f"Required SMA columns '{sma_short_col}' or '{sma_long_col}' not found. Cannot compute SMA run length for '{new_col}'.")
            df[new_col] = np.nan
            return df

        def compute_run_length_for_group(group):
            # Ensure the group is sorted by time for accurate run length
            group = group.sort_values(by='time')
            condition = group[sma_short_col] < group[sma_long_col]
            group[new_col] = condition.groupby((~condition).cumsum()).cumcount() + 1
            group.loc[~condition, new_col] = 0
            return group

        # Apply to each date group
        df_with_run = df.groupby('date', group_keys=False).apply(compute_run_length_for_group)
        return df_with_run

    def add_run_length_features(self, slope_column='slope_10', sma_cross_params=None):
        """
        Adds various run length features to the DataFrame.
        This is a public method orchestrating the run length calculations.

        Args:
            slope_column (str): The name of the slope column to use for slope run lengths.
            sma_cross_params (list of dict): A list of dictionaries, where each dict specifies
                                              'sma_short_col', 'sma_long_col', and 'new_col_name' for SMA run lengths.
                                              Example: [{'sma_short_col': 'SMA_25', 'sma_long_col': 'SMA_100', 'new_col_name': 'sma_25_below_100_run_length'}]
        """
        logging.info("Adding run length features...")
        
        # Add slope run lengths
        self.df = self._add_slope_run_length(self.df, slope_column=slope_column)
        logging.info(f"Added slope run length features based on '{slope_column}'.")

        # Add SMA cross run lengths if parameters are provided
        if sma_cross_params:
            for params in sma_cross_params:
                sma_short = params.get('sma_short')
                sma_long = params.get('sma_long')
                new_col_name = params.get('new_col_name')
                
                if sma_short and sma_long and new_col_name:
                    self.df = self._add_sma_run_length(
                        self.df,
                        sma_short_col=sma_short, # Changed parameter name for clarity
                        sma_long_col=sma_long,   # Changed parameter name for clarity
                        new_col=new_col_name
                    )
                    logging.info(f"Added SMA run length feature '{new_col_name}'.")
                else:
                    logging.warning(f"Invalid SMA cross parameters: {params}. Skipping.")
        else:
            logging.info("No SMA cross parameters provided. Skipping SMA run length features.")

        logging.info("Completed adding run length features.")


    def calculate_daily_stats(self):
        """
        Calculates various statistics for each day in the DataFrame.
        This is a placeholder for actual calculations.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot calculate daily stats.")
            return []

        all_stats = []
        # Ensure 'date' column is in datetime format for grouping
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            # Assuming 'date' is in 'YYYY-MM-DD' string format
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Group by date and calculate statistics
        for date, group in self.df.groupby(self.df['date'].dt.date):
            date_str = date.strftime('%Y-%m-%d')
            # Example statistics - replace with actual desired calculations
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
        Prepares normalized features by dividing 'close' and SMA_ columns by
        the 'close' price at a specific normalization time for each day.
        Other columns like 'high' and 'low' are also normalized.

        Args:
            norm_time (datetime.time): The time of day to use for normalization (e.g., datetime.time(8, 30)).
                                        Defaults to DEFAULT_NORM_TIME if not provided.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot prepare normalized features.")
            return

        # Ensure 'time' column is datetime.time object
        if 'time' in self.df.columns and not isinstance(self.df['time'].iloc[0], datetime.time):
            self.df['time'] = pd.to_datetime(self.df['time'], format='%H:%M:%S').dt.time

        # Ensure 'date' column is datetime.date object
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
             self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d').dt.date

        # Identify columns to normalize
        # Now SMA_ columns will exist if _add_sma_features was called previously
        norm_cols = ['close'] + [col for col in self.df.columns if col.startswith('SMA_')]
        for col in ['open', 'high', 'low']:
            if col in self.df.columns and col not in norm_cols:
                norm_cols.append(col)

        def normalize_group(group):
            base_rows = group.loc[group['time'] == norm_time]

            if base_rows.empty:
                logging.warning(f"No row found at time '{norm_time.strftime('%H:%M')}' for date {group.name}. Skipping normalization for this day.")
                return group
            
            base_value = base_rows.iloc[0]['close']

            if base_value == 0:
                logging.warning(f"Close value at {norm_time.strftime('%H:%M')} is 0 for date {group.name}. Skipping normalization for this day.")
                return group
            
            cols_to_apply = [col for col in norm_cols if col in group.columns]
            if cols_to_apply:
                group.loc[:, cols_to_apply] = group.loc[:, cols_to_apply] / base_value
            
            return group

        self.df = self.df.groupby('date', group_keys=False).apply(normalize_group)
        logging.info(f"Normalized features using {norm_time.strftime('%H:%M')} as base.")

    def _compute_slopes_for_timeframe_single(self, dataframe_day_group, timeframe, scalers_dir, use_saved_scaler=True):
        """
        Helper method to compute slopes and standardize for a single timeframe
        on a daily group. This is called by apply from compute_slopes_and_standardize.
        """
        df_local = dataframe_day_group.copy()

        if 'date' in df_local.columns and not pd.api.types.is_datetime64_any_dtype(df_local['date']):
            df_local['date'] = pd.to_datetime(df_local['date']).dt.date

        # 1. Compute daily slopes using the class's private method
        df_local = self._compute_daily_slopes(df_local, 'close', timeframe) # Using 'close' here assumes 'close' is the base for slopes
                                                                            # If you want slopes on normalized 'close', ensure normalization runs first.
        # 2. Add slope difference using the class's private method
        df_local = self._add_slope_difference(df_local, slope_lookback=timeframe)

        new_cols = [f'slope_{timeframe}', f'd2_{timeframe}']
        
        existing_new_cols = [col for col in new_cols if col in df_local.columns]

        if not existing_new_cols:
            logging.warning(f"Slope columns {new_cols} not found for timeframe {timeframe} for date {dataframe_day_group.name}. Returning original group.")
            return dataframe_day_group

        new_cols_df = df_local[existing_new_cols].copy()

        scaler_filename = os.path.join(scalers_dir, f"scaler_timeframe_{timeframe}.pkl")

        if use_saved_scaler and os.path.exists(scaler_filename):
            try:
                scaler = joblib.load(scaler_filename)
                cols_for_transform = [col for col in scaler.feature_names_in_ if col in new_cols_df.columns]
                new_cols_standardized = scaler.transform(new_cols_df[cols_for_transform])
            except Exception as e:
                logging.error(f"Error loading or transforming with saved scaler for timeframe {timeframe}: {e}. Fitting new scaler.")
                scaler = StandardScaler()
                new_cols_standardized = scaler.fit_transform(new_cols_df)
                joblib.dump(scaler, scaler_filename)
        else:
            logging.info(f"No saved scaler found or use_saved_scaler=False. Fitting a new scaler for timeframe {timeframe}.")
            scaler = StandardScaler()
            new_cols_standardized = scaler.fit_transform(new_cols_df)
            joblib.dump(scaler, scaler_filename)
        
        new_cols_df_standardized = pd.DataFrame(
            new_cols_standardized,
            columns=[f"{col.replace('_norm', '')}" for col in existing_new_cols],  # Remove '_norm' suffix if present
            index=new_cols_df.index
        )
        
        return dataframe_day_group.merge(new_cols_df_standardized, left_index=True, right_index=True, how='left')


    def compute_slopes_and_standardize(self, timeframes=DEFAULT_TIMEFRAMES, use_saved_scaler=True):
        """
        Computes slopes and their differences for a list of timeframes,
        and standardizes these features using StandardScaler.
        Scalers are saved/loaded based on the timeframe.
        """
        if self.df.empty:
            logging.warning("DataFrame is empty, cannot compute slopes and standardize.")
            return

        if self.repo_root is None:
            logging.error("repo_root not set in StockDataProcessor. Cannot save/load scalers.")
            return

        scalers_dir = os.path.join(self.repo_root, "other_analysis", "scalers")
        os.makedirs(scalers_dir, exist_ok=True)
        logging.info(f"Scalers directory: {scalers_dir}")

        if 'date' not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date']).dt.date

        for timeframe in timeframes:
            logging.info(f"Computing slopes and standardizing for timeframe: {timeframe}")
            
            temp_df_with_new_cols = self.df.groupby('date', group_keys=False).apply(
                lambda group: self._compute_slopes_for_timeframe_single(group, timeframe, scalers_dir, use_saved_scaler)
            )
            
            # The _compute_slopes_for_timeframe_single now returns columns without '_norm' suffix
            # So, we just look for original 'slope_' and 'd2_' columns if they were successfully added.
            newly_added_cols = [col for col in temp_df_with_new_cols.columns if col.startswith('slope_') or col.startswith('d2_')]

            if newly_added_cols:
                cols_to_merge_from_temp = newly_added_cols + ['epoch_time']
                
                self.df = self.df.merge(
                    temp_df_with_new_cols[cols_to_merge_from_temp],
                    on='epoch_time',
                    how='left',
                    suffixes=('', '_drop')
                )
                self.df.drop(columns=[col for col in self.df.columns if '_drop' in col], inplace=True)

                self.df.drop_duplicates(subset=['epoch_time'], inplace=True)
                self.df.reset_index(drop=True, inplace=True)

                logging.info(f"Added {len(newly_added_cols)} standardized columns for timeframe {timeframe}.")
            else:
                logging.warning(f"No standardized columns were generated for timeframe {timeframe}.")

        logging.info("Completed slope computation and standardization for all timeframes.")

    def load_model(self):
        """
        Loads the pre-trained Random Forest model from the specified path.
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
        Makes predictions using the loaded Random Forest model and adds them to the DataFrame.

        Args:
            prediction_column_name (str): The name for the new column containing prediction probabilities.
        """
        if self.model is None:
            logging.error("Model not loaded. Cannot make predictions.")
            self.df[prediction_column_name] = np.nan
            return

        if not self.feature_columns:
            logging.warning("No feature columns specified for prediction. Skipping predictions.")
            self.df[prediction_column_name] = np.nan
            return

        # Check if all required feature columns exist in the DataFrame
        missing_features = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_features:
            logging.error(f"Missing required feature columns for prediction: {', '.join(missing_features)}. Cannot make predictions.")
            self.df[prediction_column_name] = np.nan
            return

        logging.info("Making predictions using the Random Forest model...")
        try:
            # Ensure the DataFrame for prediction only contains the selected features
            X = self.df[self.feature_columns]
            
            # Predict probabilities for the positive class (class 1)
            predictions = self.model.predict_proba(X)[:, 1]
            self.df[prediction_column_name] = predictions
            logging.info(f"Predictions added to column '{prediction_column_name}'.")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            self.df[prediction_column_name] = np.nan

    def get_dataframe(self):
        """
        Returns the processed DataFrame.
        """
        return self.df

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir) # Change current working directory to script's directory

    repo_root = get_git_repo_root()
    if not repo_root:
        logging.error("Not inside a Git repository. Exiting.")
        exit()

    symbol = "SPY"
    start_date = "2025-06-10"
    end_date = "2025-06-13"

    use_sql = True

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

    model_path = os.path.join(repo_root, "other_analysis", "pos_slope_model.pkl") # Use os.path.join for path
    # model_path = os.path.join(repo_root, "other_analysis", "low_close_random_forest_model.pkl") # Example for another model

    selected_features = [
        "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
        "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
        "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120",
        "sma_25_below_100_run_length", "negative_slope_run_length", "positive_slope_run_length"
    ]

    data_processor = StockDataProcessor(
        df,
        repo_root=repo_root,
        model_path=model_path,
        feature_columns=selected_features
    )

    data_processor._add_sma_features(timeframes=DEFAULT_SMA_TIMEFRAMES)
    
    data_processor.prepare_normalized_features(norm_time=datetime.time(8, 30)) # This will now normalize SMAs too
    
    logging.info("Starting slope computation and standardization...")
    data_processor.compute_slopes_and_standardize(
        timeframes=DEFAULT_TIMEFRAMES,
        use_saved_scaler=True
    )
    logging.info("Slope computation and standardization complete.")

    sma_run_length_params = [
        {'sma_short': 'SMA_25', 'sma_long': 'SMA_100', 'new_col_name': 'sma_25_below_100_run_length'},
    ]
    data_processor.add_run_length_features(
        slope_column='slope_10', # This will now refer to the standardized slope_10 column
        sma_cross_params=sma_run_length_params
    )
    logging.info("Run length feature addition complete.")

    # Load the model and make predictions
    data_processor.load_model()
    data_processor.make_predictions(prediction_column_name='rf_prediction_probability')
    logging.info("Random Forest predictions added to DataFrame.")

    daily_stats = data_processor.calculate_daily_stats()
    if daily_stats:
        logging.info("Daily statistics calculated (possibly on normalized and standardized data):")
        for stat in daily_stats:
            logging.info(f"   Date: {stat['date']}, Close: {stat['close']:.4f}, Volume: {stat['volume']}")
    else:
        logging.info("No daily statistics to display.")

    processed_df = data_processor.get_dataframe()
    start_plot_time = pd.to_datetime("08:30:00").time()
    end_plot_time   = pd.to_datetime("15:00:00").time()
    mask_plot = (processed_df['time'] >= start_plot_time) & (processed_df['time'] <= end_plot_time)
    processed_df = processed_df.loc[mask_plot].copy().reset_index(drop=True)
    logging.info("\nFirst 20 rows of the processed (normalized and standardized) DataFrame with predictions:")
    print(processed_df.head(n=20))
    logging.info(f"\nColumns in final DataFrame: {processed_df.columns.tolist()}")

if __name__ == "__main__":
    main()