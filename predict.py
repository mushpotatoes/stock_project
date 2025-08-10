from datetime import datetime, timedelta, time
from itertools import cycle
import os
import pandas as pd
import numpy as np
import pytz
import requests
import torch
import torch.nn as nn
import logging
from typing import List, Tuple
from prep_data import StockDataProcessor
import matplotlib.pyplot as plt
from itertools import cycle

# --- Basic Setup ---
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 1. YOUR MODEL DEFINITION
# This must match the architecture of your saved model file exactly.
# =============================================================================

class ResidualBlock(nn.Module):
    """A residual block for a fully connected network."""
    def __init__(self, in_features, out_features, dropout_prob=0.2):
        super(ResidualBlock, self).__init__()
        
        # Path F(x)
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(out_features, out_features) # Added another linear layer
        )
        
        # Path x (shortcut)
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
            
        # Final activation after the addition
        self.final_relu = nn.ReLU()
            
    def forward(self, x):
        # Calculate F(x) + x
        residual = self.main_path(x) + self.shortcut(x)
        # Apply final activation
        return self.final_relu(residual)

class ResidualFCNN(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_prob=0.2):
        super(ResidualFCNN, self).__init__()

        assert len(layer_sizes) > 1, "The architecture should have at least an input and output layer."

        first_layer_dim = layer_sizes[0]
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, first_layer_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )

        self.blocks = nn.ModuleList()
        
        # Create a sequence of residual blocks
        for i in range(len(layer_sizes) - 2):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            block = ResidualBlock(in_size, out_size, dropout_prob)
            self.blocks.append(block)
        self.final_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])


    def forward(self, x, additional_data=None):
        # Flatten and concatenate inputs
        x = x.view(x.size(0), -1)
        if additional_data is not None:
            x = torch.cat([x, additional_data], dim=1)
        
        x = self.input_layer(x)
        
        # Now proceed with the residual blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.final_layer(x)
        return x

# =============================================================================
# 2. YOUR DATA FETCHING & FEATURE CALCULATION
# Replace this section with your actual Polygon.io API calls and feature logic.
# =============================================================================
def generate_architecture(spec, input_dim, output_dim):
    """
    Generates a list of layer sizes based on a specification.

    Args:
        spec (list): A list of [node_count, layer_repeat] pairs.
        input_dim (int): The size of the input layer.
        output_dim (int): The size of the output layer.

    Returns:
        list: A list of integers representing the network architecture.
    """
    # Check for an even number of elements in the spec
    if len(spec) % 2 != 0:
        raise ValueError("The 'spec' list must contain an even number of elements (pairs of node_count and layer_repeat).")
    
    layers = [input_dim]
    for i in range(0, len(spec), 2):
        node_count = spec[i]
        layer_repeat = spec[i+1]
        layers.extend([node_count] * layer_repeat)
    
    layers.append(output_dim)
    return layers

def get_data_from_polygon(ticker: str, start_date: str) -> pd.DataFrame:
    api_key = os.environ.get('API_KEY')
    central_timezone = pytz.timezone('US/Central')
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    next_day = current_date + timedelta(days=1)
    next_day_string = next_day.strftime('%Y-%m-%d')

    if not api_key:
        logging.error("Polygon API key not found in environment variable 'API_KEY'.")
        return pd.DataFrame()

    limit = 5000 # Max limit per request
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{start_date}/{next_day_string}"
    url += f"?adjusted=false&sort=asc&limit={limit}&apiKey={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        response_json = response.json()

        if "results" not in response_json or not response_json["results"]:
            logging.warning(f"No Polygon data retrieved for {ticker} from {start_date} to {next_day_string}.")
            return pd.DataFrame()

        results = response_json["results"]
        new_data = pd.DataFrame({
            'epoch_time': [item['t'] / 1000.0 for item in results], # Convert ms to seconds
            'date': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(central_timezone).strftime("%Y-%m-%d") for item in results],
            'time': [pd.to_datetime(item['t'], unit='ms', utc=True).tz_convert(central_timezone).strftime("%H:%M:%S") for item in results],
            'open': [item['o'] for item in results],
            'high': [item['h'] for item in results],
            'low': [item['l'] for item in results],
            'close': [round(item['c'], 2) for item in results],
            'volume': [item['v'] for item in results]
        })
        return new_data

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download Polygon data for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing Polygon data for {ticker}: {e}")
        return pd.DataFrame()

# =============================================================================
# 3. YOUR DATA PROCESSOR
# This class reshapes the daily data into samples for the model.
# =============================================================================

class InferenceProcessor:
    """Processes a DataFrame for model inference."""
    def __init__(self, data: pd.DataFrame, selected_features):
        self.df = data
        self.selected_features = selected_features
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

    def process_data_into_samples(self) -> Tuple[np.ndarray, None]:
        """
        Creates sliding-window samples from the daily data.
        This is a simplified version of your original `process_data_by_day`.
        """
        selected_features = [
            "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
            "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
            "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120"
        ]
        
        if self.df.empty:
            logging.error("Input DataFrame is empty. Cannot create samples.")
            return np.array([]), None
        
        feature_array = []
        daily_data = self.df.copy()
        
        if len(daily_data) == 391:
            daily_data['time_dt'] = pd.to_datetime(daily_data['time'], format='%H:%M:%S')
            daily_data = daily_data.sort_values(by='time_dt').reset_index(drop=True)
            
            # Use 30 minutes of data to create samples
            # This loop creates 391 - 60 = 331 potential prediction opportunities in a day
            for x in range(391 - 60):
                features = daily_data.iloc[x+1:x+31][selected_features].to_numpy()
                feature_array.append(features)
        else:
            logging.warning(f"Data has {len(daily_data)} rows, expected 391. No samples created.")

        return np.array(feature_array), None

def process_data_by_day(df, ticker = "SPY"):
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
    if df is None:
        logging.warning(f"DataFrame for {ticker} is not loaded. Cannot process data by day.")
        return

    unique_dates_array = df['date'].unique()
    # Convert numpy datetime64 objects to pandas Timestamps for consistency if needed
    unique_dates_list = [pd.Timestamp(d) for d in unique_dates_array]
    unique_dates = sorted(unique_dates_list)
    if not unique_dates:
        logging.info(f"No unique dates found in the DataFrame for {ticker}. Nothing to process.")
        return
    print(unique_dates_list)
    logging.info(f"\n--- Processing data by day for {ticker} ---")
    feature_array = []
    for date in unique_dates:
        daily_data = df.copy() # Use .copy() to avoid SettingWithCopyWarning
        logging.info(f"Processing data for date: {date.strftime('%Y-%m-%d')}")

        # --- Start of user-defined daily processing logic ---
        # You can add your custom processing steps here for 'daily_data'
        # For example, calculate daily aggregates, apply filters, run models, etc.
        if not daily_data.empty:
            # logging.info(f"  Rows for this day: {len(daily_data)}")
            daily_data['time_dt'] = pd.to_datetime(daily_data['time'], format='%H:%M:%S')
            daily_data = daily_data.sort_values(by='time_dt').reset_index(drop=True)
            len_df = len(daily_data)
            for x in range(len_df - 30):
                features = daily_data.iloc[x+1:x+31][selected_features].to_numpy()
                feature_array.append(features)
        else:
            logging.info(f"  No data found for this specific date.")

    return np.array(feature_array)

# =============================================================================
# 4. MAIN INFERENCE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TICKER = "SPY"
    
    # --- USER DEFINED SETTINGS ---
    # 1. Define the date range to cycle through
    START_DATE = "2025-07-21"  # <-- CHANGE: Start of your date range
    END_DATE = "2025-08-05"    # <-- CHANGE: End of your date range

    # 2. Define the output directory and a prefix for saved files
    OUTPUT_DIR = "C:/Users/deade/OneDrive/Desktop/data_science/nn_models/predictions" #<-- CHANGE
    OUTPUT_PREFIX = f"{TICKER}_TRNS" # <-- CHANGE: Your desired file prefix

    # --- Create output directory if it doesn't exist ---
    os.makedirs(OUTPUT_DIR, exist_ok=True) # <-- CHANGE

    # These must match the model you saved
    arch_spec = [4096, 12, 1024, 8]
    separators = cycle(['x', '-'])
    str_nums = map(str, arch_spec)
    first_num = next(str_nums)
    architecture_string = first_num + "".join(sep + num for sep, num in zip(separators, str_nums))
    MODEL_LAYER_SIZES = generate_architecture(arch_spec, 630, 1)
    MODEL_PATH = f"C:/Users/deade/OneDrive/Desktop/data_science/nn_models/{architecture_string}_model.pth"

    INPUT_DIM = 630
    DEFAULT_NORM_TIME = time(8, 30)
    DEFAULT_TIMEFRAMES = [10, 15, 25, 40, 90, 100, 120]
    DEFAULT_SMA_TIMEFRAMES = [10, 25, 40, 90, 100, 120]
    PLOT_SMA_TIMEFRAMES = [15, 25, 100]
    selected_features = [ "close", "SMA_10", "SMA_25", "SMA_40", "SMA_90", "SMA_100", "SMA_120",
                          "slope_10", "slope_15", "slope_25", "slope_40", "slope_90", "slope_100", "slope_120",
                          "d2_10", "d2_15", "d2_25", "d2_40", "d2_90", "d2_100", "d2_120" ]

    # --- Step 1: Load Model (Done once before the loop) ---
    try:
        model = ResidualFCNN(input_dim=INPUT_DIM, layer_sizes=MODEL_LAYER_SIZES).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval() # Set model to evaluation mode
        logging.info(f"Model successfully loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
        exit()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit()

    # --- Step 2: Loop through the date range ---
    # Use pandas to easily iterate through business days, skipping weekends
    dates_to_process = pd.date_range(start=START_DATE, end=END_DATE, freq='B')

    for date_to_predict_dt in dates_to_process:
        date_to_predict = date_to_predict_dt.strftime('%Y-%m-%d') # Format date as string
        logging.info(f"--- Processing Date: {date_to_predict} ---")

        try: # <-- CHANGE: Added a try block for robust error handling per date
            # --- Step 3: Get and process data for the current date ---
            daily_df = get_data_from_polygon(TICKER, date_to_predict)
            
            if daily_df.empty:
                logging.warning(f"No data returned for {date_to_predict}. Skipping (may be a holiday).")
                continue

            daily_df = daily_df[daily_df['date'] == date_to_predict]
            market_open_str = '08:30:00'
            post_830_df = daily_df[daily_df['time'] >= market_open_str].copy()
            future_close = post_830_df['close'].shift(-30)

            post_830_df['future_ratio_30min'] = future_close / post_830_df['close']
            post_830_df = post_830_df[post_830_df['time'] <= '15:00:00']

            repo_root = os.getcwd()
            data_processor = StockDataProcessor(
                daily_df,
                repo_root=repo_root,
                model_path=None,
                feature_columns=selected_features
            )
            
            data_processor._add_sma_features(timeframes=DEFAULT_SMA_TIMEFRAMES)
            data_processor._add_plotting_sma_features(timeframes=PLOT_SMA_TIMEFRAMES)
            data_processor.prepare_normalized_features(norm_time=DEFAULT_NORM_TIME)
            data_processor.compute_slopes_and_standardize(
                timeframes=DEFAULT_TIMEFRAMES,
                use_saved_scaler=True
            )
            sma_run_length_params = [
                {'sma_short': 'SMA_25', 'sma_long': 'SMA_100', 'new_col_name': 'sma_25_below_100_run_length'},
            ]
            data_processor.add_run_length_features(
                slope_column='slope_10', 
                sma_cross_params=sma_run_length_params
            )
            
            daily_df = data_processor.get_dataframe()
            feature_array = process_data_by_day(daily_df)

            if feature_array.shape[0] == 0:
                logging.warning(f"No samples were generated for {date_to_predict}. Skipping.")
                continue
            
            # --- Step 4: Run Inference ---
            feature_tensor = torch.from_numpy(feature_array).float().to(device)
            logging.info(f"Running inference on {feature_tensor.shape[0]} samples for {date_to_predict}...")
            
            with torch.no_grad():
                predictions = model(feature_tensor).to("cpu")

            logging.info(f"--- Inference Complete for {date_to_predict} ---")

            # --- Step 5: Plot and Save the Output ---
            fig, axs = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 2]})
            fig.suptitle(f'{TICKER} Predictions for {date_to_predict}', fontsize=16)

            axs[0].plot(post_830_df['time'], post_830_df['close'], label='Actual Close Price', color='dodgerblue', linewidth=2)
            axs[0].set_title('SPY Close Prices')
            axs[0].set_ylabel('Price ($)')
            axs[0].legend()
            
            num_values_after_830 = len(post_830_df)
            predictions_to_plot = predictions.flatten().numpy()[-num_values_after_830:]
            ax2 = axs[0].twinx() 
            ax2.plot(predictions_to_plot, linestyle='--', color='green', label=f'Predicted Values')
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.set_ylim(0.996, 1.004)
            moving_avg = pd.Series(predictions_to_plot).rolling(window=10).mean()
            ax2.plot(moving_avg, linestyle='-', color='black', label='Moving Average', linewidth=2)
            ax2.axhline(y=1, color='black', linestyle='--', linewidth=1.5)
            ax2.axhspan(ax2.get_ylim()[0], 1, color='lightgray', alpha=0.5, zorder=0)
            
            tick_indices = np.linspace(0, len(post_830_df['time']) - 1, 10, dtype=int)
            axs[0].set_xticks(tick_indices)
            axs[0].set_xticklabels(post_830_df['time'].iloc[tick_indices], rotation=30)
            axs[0].set_xlabel('Time (US/Central)')

            axs[1].plot(predictions_to_plot, linestyle='--', color='green', label=f'Predicted Values')
            axs[1].plot(moving_avg, linestyle='-', color='black', label='Moving Average', linewidth=2)
            axs[1].plot(post_830_df['time'], post_830_df['future_ratio_30min'], linestyle='-', color='red', label=f'Actual Values')
            axs[1].set_title(f'Predicted Change vs Actual Change')
            axs[1].set_ylabel('Predicted Ratio')
            axs[1].set_xticks(tick_indices)
            axs[1].set_xticklabels(post_830_df['time'].iloc[tick_indices], rotation=30)
            axs[1].set_xlabel('Time (US/Central)')
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # --- SAVE THE PLOT INSTEAD OF SHOWING IT ---
            output_filename = f"{OUTPUT_PREFIX}_{architecture_string}_{date_to_predict}.png" # <-- CHANGE
            output_path = os.path.join(OUTPUT_DIR, output_filename) # <-- CHANGE
            print(f"Saving to: {output_path}")
            plt.savefig(output_path) # <-- CHANGE
            plt.close(fig) # <-- CHANGE: Close the figure to free up memory
            logging.info(f"Successfully saved plot to {output_path}")

        except Exception as e: # <-- CHANGE: Catch any errors for the date
            logging.error(f"An error occurred while processing {date_to_predict}: {e}")
            # Ensure plot is closed even if an error occurs mid-creation
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
            continue # Move to the next date

    logging.info("--- All dates processed. ---")






