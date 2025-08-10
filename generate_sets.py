import pandas as pd
import numpy as np
import os
import logging
import warnings
from typing import List, Optional, Any

# -----------------------------------------------------------------------------
# Logging & Warnings Setup
# -----------------------------------------------------------------------------
log_filename = 'generate_sets.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler(log_filename, 'a'), # Uncomment to log to a file
        logging.StreamHandler()  # Log to console
    ]
)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class DataFrameProcessor:
    """
    A class to handle operations on financial dataframes,
    including loading, date extraction, and display.
    """
    def __init__(self, ticker: str):
        """
        Initializes the DataFrameProcessor with a given ticker and loads the dataframe.

        Args:
            ticker (str): The stock ticker symbol (e.g., 'SPY', 'QQQ', 'DIA').
        """
        self.ticker = ticker
        self.df: Optional[pd.DataFrame] = None
        self._load_dataframe()

    def _load_dataframe(self) -> None:
        """
        Loads the dataframe from a pickle file.
        Converts the 'date' column to datetime objects upon loading.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_file_path = os.path.join(script_dir, f"{self.ticker}_with_features.pkl")

        try:
            self.df = pd.read_pickle(pickle_file_path)
            logging.info(f"DataFrame for {self.ticker} successfully loaded from {pickle_file_path}.")
            # Convert 'date' column to datetime here to ensure consistency for all operations
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                logging.debug(f"Date column converted to datetime for {self.ticker}.")
            else:
                logging.warning(f"No 'date' column found in DataFrame for {self.ticker}.")
            logging.debug(f"Head of DataFrame for {self.ticker}:\n{self.df.head()}")
        except FileNotFoundError:
            logging.error(f"Pickle file not found for {self.ticker} at {pickle_file_path}. Please ensure it exists.")
            self.df = None
        except Exception as e:
            logging.error(f"Error reading DataFrame for {self.ticker} from Pickle: {e}")
            self.df = None

    def get_unique_dates(self) -> List[pd.Timestamp]:
        """
        Returns a sorted list of unique values from the 'date' column.
        Assumes 'date' column exists and is already converted to datetime.

        Returns:
            List[pd.Timestamp]: A sorted list of unique dates.
        """
        if self.df is not None and 'date' in self.df.columns:
            try:
                unique_dates_array = self.df['date'].unique()
                # Convert numpy datetime64 objects to pandas Timestamps for consistency if needed
                unique_dates_list = [pd.Timestamp(d) for d in unique_dates_array]
                return sorted(unique_dates_list)
            except KeyError:
                logging.error("The 'date' column does not exist in the DataFrame.")
                return []
            except Exception as e:
                logging.error(f"Error getting unique dates: {e}")
                return []
        elif self.df is None:
            logging.warning(f"DataFrame for {self.ticker} is not loaded. Cannot get unique dates.")
            return []
        else:
            logging.warning(f"The 'date' column does not exist in the DataFrame for {self.ticker}.")
            return []

    def print_first_n_dates(self, n: int = 3) -> List[pd.Timestamp]:
        """
        Prints the first N unique dates from the dataframe and returns them.

        Args:
            n (int): The number of first unique dates to print. Defaults to 3.

        Returns:
            List[pd.Timestamp]: A list containing the first N unique dates.
        """
        unique_dates = self.get_unique_dates()
        dates_to_return = []
        if unique_dates:
            logging.info(f"First {n} unique dates for {self.ticker}:")
            for i, date in enumerate(unique_dates[:n]):
                logging.info(f"- {date.strftime('%Y-%m-%d')}")
                dates_to_return.append(date)
        else:
            logging.info(f"No unique dates to display for {self.ticker}.")
        return dates_to_return

    def print_timeseries_for_dates(self, dates_to_print: List[pd.Timestamp]) -> None:
        """
        Prints the time series (all rows) from the DataFrame for the given list of dates.

        Args:
            dates_to_print (List[pd.Timestamp]): A list of pandas Timestamp objects
                                                 for which to print the time series data.
        """
        if self.df is None:
            logging.warning(f"DataFrame for {self.ticker} is not loaded. Cannot print time series.")
            return

        if not dates_to_print:
            logging.info(f"No specific dates provided to print time series for {self.ticker}.")
            return

        for date in dates_to_print:
            # Filter the DataFrame for the current date
            daily_data = self.df[self.df['date'] == date]

            if not daily_data.empty:
                logging.info(f"\n--- Time Series for {self.ticker} on {date.strftime('%Y-%m-%d')} ---")
                logging.info(daily_data.to_string()) # Use to_string() to print entire DataFrame
                logging.info("-" * 40)
            else:
                logging.info(f"\nNo data found for {self.ticker} on {date.strftime('%Y-%m-%d')}.")

    def process_data_by_day(self) -> Any:
        """
        Loads and processes data one day at a time.
        Iterates through each unique date in the dataframe and allows
        for custom processing logic to be added for each day's data.
        """
        selected_features = [
            "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
            "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
            "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120"
        ]
        if self.df is None:
            logging.warning(f"DataFrame for {self.ticker} is not loaded. Cannot process data by day.")
            return

        unique_dates = self.get_unique_dates()
        if not unique_dates:
            logging.info(f"No unique dates found in the DataFrame for {self.ticker}. Nothing to process.")
            return

        logging.info(f"\n--- Processing data by day for {self.ticker} ---")
        feature_array = []
        target_array = []
        idx = 0
        for date in unique_dates:
            idx = idx + 1
            daily_data = self.df[self.df['date'] == date].copy() # Use .copy() to avoid SettingWithCopyWarning
            logging.info(f"Processing data for date: {date.strftime('%Y-%m-%d')}")

            # --- Start of user-defined daily processing logic ---
            # You can add your custom processing steps here for 'daily_data'
            # For example, calculate daily aggregates, apply filters, run models, etc.
            if not daily_data.empty:
                # logging.info(f"  Rows for this day: {len(daily_data)}")
                
                if len(daily_data) == 391:
                    daily_data['time_dt'] = pd.to_datetime(daily_data['time'], format='%H:%M:%S')
                    daily_data = daily_data.sort_values(by='time_dt').reset_index(drop=True)
                    # Skip the first anomolously high volume data point each day
                    # Use 30 minutes of data to see if the close in another 30 minutes
                    # is higher than the close in 2 minutes.
                    for x in range(391 - 60):
                        # print(daily_data.iloc[x+1:x+30][selected_features].to_numpy())
                        # print(daily_data.iloc[x+1:x+31][["date", "time"]])
                        features = daily_data.iloc[x+1:x+31][selected_features].to_numpy()
                        future_ratio = daily_data.iloc[x+60]['close'] / daily_data.iloc[x+32]['close']
                        # print(features)
                        # print(future_ratio)
                        feature_array.append(features)
                        target_array.append(future_ratio)
                    # if idx > 20:
                    #     break
                        # input("")
                # Example: Print a summary of the daily data
                # logging.info(f"  Daily summary:\n{daily_data.describe()}")
                # Example: Access specific columns for the day
                # daily_open_price = daily_data['open'].iloc[0] if 'open' in daily_data.columns else 'N/A'
                # logging.info(f"  First open price for the day: {daily_open_price}")
            else:
                logging.info(f"  No data found for this specific date.")

            # --- End of user-defined daily processing logic ---
            logging.debug(f"Finished processing for {date.strftime('%Y-%m-%d')}.")
        print(f"Length of feature collection: {len(feature_array)}")
        print(f"Shape of feature collection: {feature_array[0].shape}")
        # exit()
        logging.info(f"--- Finished processing data by day for {self.ticker} ---")
        feature_array_np = np.array(feature_array)
        target_array_np = np.array(target_array)
        return feature_array_np, target_array_np


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure dummy pickle files exist for demonstration
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Process each ticker, loading and processing data day by day
    all_feature_data_chunks = []
    all_target_data_chunks = []
    for ticker in ['SPY', 'QQQ', 'DIA']:
        pickle_file_path = os.path.join(script_dir, f"{ticker}_with_features.pkl")
        if not os.path.exists(pickle_file_path):
            logging.warning(f"Creating a dummy pickle file for {ticker} as it doesn't exist.")
            dummy_data = {
                'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-04']),
                'time': ['09:30:00', '10:00:00', '09:30:00', '10:00:00', '09:30:00', '10:00:00', '09:30:00'],
                'open': np.random.rand(7) * 100,
                'high': np.random.rand(7) * 100 + 1,
                'low': np.random.rand(7) * 100 - 1,
                'close': np.random.rand(7) * 100,
                'volume': np.random.randint(10000, 100000, 7)
            }
            dummy_df = pd.DataFrame(dummy_data)
            # Combine date and time into a single datetime column for full timestamp
            dummy_df['datetime'] = pd.to_datetime(dummy_df['date'].astype(str) + ' ' + dummy_df['time'])
            dummy_df = dummy_df.drop(columns=['time']) # Drop the separate time column
            dummy_df.to_pickle(pickle_file_path)
            logging.warning(f"Dummy pickle file created at: {pickle_file_path}")


    for ticker in ['SPY', 'QQQ', 'DIA']:
        processor = DataFrameProcessor(ticker)
        if processor.df is not None:
            # feature_array and target_array returned here are already NumPy arrays for that ticker
            ticker_features, ticker_targets = processor.process_data_by_day()
            
            if ticker_features.shape[0] > 0: # Only append if there's actual data
                all_feature_data_chunks.append(ticker_features)
                all_target_data_chunks.append(ticker_targets)

    # Concatenate all collected NumPy arrays into final training datasets
    final_feature_data = np.concatenate(all_feature_data_chunks, axis=0) if all_feature_data_chunks else np.empty((0, 30, len([
            "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
            "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
            "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120"
        ])))
    final_target_data = np.concatenate(all_target_data_chunks, axis=0) if all_target_data_chunks else np.empty((0,))

    logging.info(f"\n--- Final Training Data Summary ---")
    logging.info(f"Overall Feature Data Shape: {final_feature_data.shape}")
    logging.info(f"Overall Target Data Shape: {final_target_data.shape}")
    
    # -------------------------------------------------------------------------
    # Saving the Training Data
    # -------------------------------------------------------------------------
    output_dir = os.path.join(script_dir, "training_data")
    os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

    feature_output_path = os.path.join(output_dir, "X_train.npy")
    target_output_path = os.path.join(output_dir, "y_train.npy")

    try:
        np.save(feature_output_path, final_feature_data)
        np.save(target_output_path, final_target_data)
        logging.info(f"Training features saved to: {feature_output_path}")
        logging.info(f"Training targets saved to: {target_output_path}")
    except Exception as e:
        logging.error(f"Error saving training data: {e}")
