import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class SPYDataLoader:
    def __init__(self, db_path):
        """
        Initialize with the path to the SQLite database.
        """
        self.db_path = db_path
        self.dataframe = None

    def load_data(self):
        """
        Connects to the SQLite database, loads all tables matching the
        'SPY_prices_%' pattern into a single DataFrame, sorts by 'epoch_time',
        and resets the index.
        
        Returns:
            pd.DataFrame: The concatenated and sorted DataFrame.
        """
        # Connect to the database.
        conn = sqlite3.connect(self.db_path)
        try:
            # Retrieve table names matching 'SPY_prices_%'
            query = """
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' AND name LIKE 'SPY_prices_%';
            """
            table_names = pd.read_sql_query(query, conn)['name'].tolist()

            # Load each table's data into a DataFrame.
            df_list = []
            for table in table_names:
                df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
                df_list.append(df)

            # Concatenate all data into one DataFrame.
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
            else:
                combined_df = pd.DataFrame()

            # Sort by 'epoch_time' and reset the index.
            combined_df.sort_values(by='epoch_time', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)

            # Save and return the final DataFrame.
            self.dataframe = combined_df
            return self.dataframe

        finally:
            conn.close()

    def iterate_days(self):
        """
        Generator that yields one day's worth of data at a time based on the
        'date' column (assumed to be in central time). Each yield is a tuple:
            (date_value, day_dataframe)
            
        Raises:
            ValueError: If the dataframe has not been loaded.
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Group by the 'date' field and yield each group.
        for date_value, group in self.dataframe.groupby('date'):
            yield date_value, group

    def add_sma(self, windows, price_column='close'):
        """
        Adds simple moving average (SMA) columns to the dataframe for the provided
        window sizes. The rolling calculations are done on the specified price column 
        (default is 'close') and computed separately for each day so that SMA calculations
        do not cross day boundaries.
        
        Parameters:
            windows (list of int): List of window sizes (in minutes) for which SMAs 
                                   should be calculated.
            price_column (str): The column to use for SMA calculations. Default is 'close'.
        
        Returns:
            pd.DataFrame: The dataframe with new SMA columns added.
            
        Raises:
            ValueError: If the dataframe has not been loaded.
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Work on a copy to avoid unintended side effects.
        df = self.dataframe.copy()

        # Compute the SMA for each specified window size within each day.
        for window in windows:
            sma_column = f'SMA_{window}'
            # Use groupby to ensure rolling windows do not cross day boundaries.
            df[sma_column] = df.groupby('date')[price_column].transform(
                lambda s: s.rolling(window, min_periods=window).mean()
            )

        # Update the instance's dataframe and return it.
        self.dataframe = df
        return self.dataframe

    def plot_day(self, day, save_path=None, price_column='close', start_time=None, end_time=None):
        """
        Plots a specific day's price values along with any SMA columns present,
        with the ability to restrict the plot to a given time range.

        Parameters:
            day (str): The date (in the same format as in the 'date' column) to plot.
            save_path (str, optional): Path to save the plot image. If not provided,
                                    the plot is displayed.
            price_column (str): The price column to plot as the main series (default 'close').
            start_time (str, optional): Starting time (e.g., '08:30') for the plot.
            end_time (str, optional): Ending time (e.g., '15:00') for the plot.

        Returns:
            None
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Filter the dataframe for the specified day.
        day_df = self.dataframe[self.dataframe['date'] == day].copy()
        if day_df.empty:
            print(f"No data found for day: {day}")
            return
        
        # Convert the 'time' column to datetime.time objects.
        day_df['time'] = pd.to_datetime(day_df['time'], format='%H:%M:%S').dt.time

        # Convert the input time strings to datetime.time objects.
        if start_time is not None:
            start_time_obj = pd.to_datetime(start_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] >= start_time_obj]
        if end_time is not None:
            end_time_obj = pd.to_datetime(end_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] <= end_time_obj]
        
        if day_df.empty:
            print(f"No data found for day: {day} within the time range {start_time} to {end_time}")
            return

        # Ensure the data is sorted by epoch_time.
        day_df.sort_values(by='epoch_time', inplace=True)
        
        # Prepare the x-axis using the DataFrame's index and label ticks with the 'time' column.
        x = day_df.index

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        
        # Plot the main price series.
        plt.plot(x, day_df[price_column], label=price_column, color='black', linewidth=1.5)
        
        # Identify and plot any SMA columns.
        sma_columns = [col for col in day_df.columns if col.startswith('SMA_')]
        for col in sma_columns:
            plt.plot(x, day_df[col], label=col, linestyle='--')
        
        plt.xlabel("Time (index; see tick labels for minute values)")
        plt.ylabel("Price")
        plt.title(f"{day} - {price_column.capitalize()} and SMA Values")
        plt.legend()
        plt.grid(True)
        
        # Optionally set x-axis ticks using the 'time' column if it exists.
        if 'time' in day_df.columns:
            tick_indices = x[::30]  # Adjust tick frequency as needed.
            tick_labels = day_df['time'].iloc[::30]
            plt.xticks(tick_indices, tick_labels, rotation=45)
        
        # Save or display the plot.
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def add_rolling_slope(self, windows, price_column='close'):
        """
        Adds columns to the dataframe for the rolling linear regression slope calculated
        using the specified number of points prior to each data point.
        
        For each window size provided (or a single integer), the function calculates the slope
        of the best-fit line (using linear regression) over that many consecutive points in the 
        `price_column`. The regression is performed separately for each day (based on the 'date'
        column) to ensure that the calculation does not cross day boundaries.
        
        Parameters:
            windows (int or list of int): The number of points to use in the regression.
                                        If an integer is provided, a single slope column is added.
                                        If a list is provided, one column is added for each window size.
            price_column (str): The column to use for the regression. Defaults to 'close'.
        
        Returns:
            pd.DataFrame: The dataframe with new rolling slope columns added.
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        import numpy as np
        from tqdm import tqdm
        import pandas as pd

        # Ensure windows is always a list.
        if isinstance(windows, int):
            windows = [windows]

        # Process each specified window.
        for window in windows:
            col_name = f'slope_{window}'
            # Create an empty Series to hold computed slope values, aligned with the original index.
            slope_series = pd.Series(index=self.dataframe.index, dtype=float)
            
            # Determine the number of unique days to track progress.
            n_groups = self.dataframe['date'].nunique()
            
            # Iterate over each day, using tqdm to show progress.
            for day, group in tqdm(self.dataframe.groupby('date'),
                                total=n_groups,
                                desc=f"Processing slopes for window {window}"):
                # Compute the rolling slope over the specified window.
                slopes = group[price_column].rolling(window, min_periods=window).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0], raw=True
                )
                # Assign the calculated slopes back into the corresponding positions.
                slope_series.loc[group.index] = slopes
            
            # Add the new column to the dataframe.
            self.dataframe[col_name] = slope_series

        return self.dataframe

    def plot_day_with_extrema(self, day, price_column='close',
                            start_time=None, end_time=None,
                            min_threshold=None, max_threshold=None,
                            save_path=None):
        """
        Plots a specific day's data in two subplots:
        - The upper subplot shows the close price along with any moving averages (columns starting with 'SMA_').
        - The lower subplot shows the slope values (columns starting with 'slope_').
        
        In addition, the function calculates local minima (for the close price) that fall below min_threshold
        and local maxima that exceed max_threshold. Vertical lines are drawn at these extrema on both subplots.
        
        Parameters:
            day (str): The date (as in the 'date' column) to plot.
            price_column (str): The main price column to plot (default 'close').
            start_time (str, optional): Start time (e.g., '08:30') to restrict the plot.
            end_time (str, optional): End time (e.g., '15:00') to restrict the plot.
            min_threshold (float, optional): Only consider local minima below this value.
            max_threshold (float, optional): Only consider local maxima above this value.
            save_path (str, optional): If provided, the plot is saved to this path; otherwise, it is displayed.
        
        Returns:
            None
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.signal import argrelextrema

        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Filter the dataframe for the specified day.
        day_df = self.dataframe[self.dataframe['date'] == day].copy()
        if day_df.empty:
            print(f"No data found for day: {day}")
            return
        
        day_df['time'] = pd.to_datetime(day_df['time'], format='%H:%M:%S').dt.time

        # Convert the input time strings to datetime.time objects.
        if start_time is not None:
            start_time_obj = pd.to_datetime(start_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] >= start_time_obj]
        if end_time is not None:
            end_time_obj = pd.to_datetime(end_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] <= end_time_obj]

        if day_df.empty:
            print(f"No data found for day: {day} within the time range {start_time} to {end_time}")
            return

        # Sort by epoch_time and reset the index for plotting.
        day_df.sort_values(by='epoch_time', inplace=True)
        day_df = day_df.reset_index(drop=True)
        x = day_df.index  # x-axis will be the sequential index

        # Create a figure with two subplots.
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))

        # --- Upper Subplot: Price and Moving Averages ---
        ax1.plot(x, day_df[price_column], label=price_column, color='black', linewidth=1.5)
        sma_columns = [col for col in day_df.columns if col.startswith('SMA_')]
        for col in sma_columns:
            ax1.plot(x, day_df[col], label=col, linestyle='--')
        ax1.set_ylabel("Price")
        ax1.set_title(f"{day} - Price and Moving Averages")
        ax1.grid(True)

        # --- Lower Subplot: Slope Values ---
        slope_columns = [col for col in day_df.columns if col.startswith('slope_')]
        if slope_columns:
            for col in slope_columns:
                ax2.plot(x, day_df[col], label=col)
        ax2.set_xlabel("Time (index; see tick labels for minute values)")
        ax2.set_ylabel("Slope")
        ax2.set_title(f"{day} - Rolling Slopes")
        ax2.grid(True)

        # # --- Calculate Local Extrema on the Close Price Series ---
        # price_vals = day_df[price_column].values
        # --- Calculate Local Extrema on the Slope Series ---
        if slope_columns:
            slope_vals = day_df[slope_columns[0]].values  # Use the first slope column for extrema calculations.
            local_min_idx = argrelextrema(slope_vals, np.less, order=1)[0]
            local_max_idx = argrelextrema(slope_vals, np.greater, order=1)[0]
        else:
            local_min_idx = np.array([])
            local_max_idx = np.array([])

        # Filter extrema based on provided thresholds.
        if min_threshold is not None:
            local_min_idx = local_min_idx[slope_vals[local_min_idx] < min_threshold]
        if max_threshold is not None:
            local_max_idx = local_max_idx[slope_vals[local_max_idx] > max_threshold]

        # --- Draw Vertical Lines for Extrema on Both Subplots ---
        min_label_added = False
        for idx in local_min_idx:
            label = 'Local Min' if not min_label_added else None
            ax1.axvline(x=idx, color='blue', linestyle=':', label=label)
            ax2.axvline(x=idx, color='blue', linestyle=':')
            min_label_added = True

        max_label_added = False
        for idx in local_max_idx:
            label = 'Local Max' if not max_label_added else None
            ax1.axvline(x=idx, color='red', linestyle=':', label=label)
            ax2.axvline(x=idx, color='red', linestyle=':')
            max_label_added = True

        # --- Set x-axis ticks using the 'time' column ---
        # Choose tick interval to avoid clutter (about 10 ticks across the plot).
        tick_interval = max(1, len(day_df) // 10)
        ax2.set_xticks(x[::tick_interval])
        ax2.set_xticklabels(day_df['time'].iloc[::tick_interval], rotation=45)

        # Add legends to both subplots.
        ax1.legend()
        ax2.legend()
        fig.tight_layout()

        # Save or display the plot.
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def add_future_metrics(self, future_window=40):
        """
        For each time point (row) in the dataframe, calculates two metrics:
        
        1. Finds the maximum 'high' value over the next two rows.
        2. Looks at a user-defined window (default 40 minutes) starting from row i+3 
            (i.e. after the next two points) and finds:
                - The maximum 'close' value within that window.
                - The number of minutes from the current row to the time when this maximum 'close'
                is first achieved (using the 'epoch_time' column for time differences).
        
        It then records:
        - The ratio of the future window's maximum close over the max high from the next two rows
            in a new column named 'future_ratio'.
        - The computed number of minutes to reach that maximum close in a new column named
            'minutes_to_max_close'.
        
        Parameters:
            future_window (int): The number of future minutes (rows) to consider for the max close.
                                Defaults to 40.
        
        Returns:
            pd.DataFrame: The dataframe with the new columns added.
        """
        import numpy as np
        import pandas as pd

        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        df = self.dataframe
        n = len(df)

        # --- 1. Compute max high over the next two points ---
        # Using shift(-1) and shift(-2) to get the next two 'high' values.
        df['max_high_2'] = pd.concat([df['high'].shift(-1), df['high'].shift(-2)], axis=1).max(axis=1)

        # --- 2. Prepare to compute max close and time-to-max in the next future_window minutes ---
        # The window now starts at index i+3 (after the two points used above) and spans `future_window` rows.
        close_array = df['close'].values
        epoch_array = df['epoch_time'].values  # assume epoch_time is in seconds
        future_max_close = np.full(n, np.nan, dtype=float)
        minutes_to_max = np.full(n, np.nan, dtype=float)

        # Loop over each row. For row i, ensure that there are at least future_window rows after i+2.
        for i in range(n):
            start_idx = i + 3
            end_idx = i + 3 + future_window  # window spans future_window rows
            if end_idx > n:
                # Not enough data points in the future; leave NaN.
                continue

            # Extract the close prices for the future window.
            window_close = close_array[start_idx:end_idx]
            # Compute the maximum close value in that window.
            max_close = window_close.max()
            future_max_close[i] = max_close

            # Identify the first occurrence (index within the window) of the maximum close.
            idx_in_window = int(np.argmax(window_close))
            target_idx = start_idx + idx_in_window

            # Compute the time difference in minutes between the current row and the row where the max close occurs.
            dt_minutes = (epoch_array[target_idx] - epoch_array[i]) / 60.0
            minutes_to_max[i] = dt_minutes

        # --- 3. Compute the ratio: max close in the future window over the max high from the next 2 points ---
        df['future_ratio'] = future_max_close / df['max_high_2']
        df['minutes_to_max_close'] = minutes_to_max

        # Optionally, if you do not need the temporary column, you can drop it:
        df.drop(columns=['max_high_2'], inplace=True)

        self.dataframe = df
        return df


    def plot_day_future_metrics(self, day, price_column='close',
                                start_time=None, end_time=None,
                                save_path=None):
        """
        Plots a specified day's metrics in three subplots:
        1. Top: The price (default 'close') and moving averages (columns starting with 'SMA_').
        2. Middle: The 'future_ratio' column.
        3. Bottom: The 'minutes_to_max_close' column.
        
        Parameters:
            day (str): The day (matching the 'date' column) to plot.
            price_column (str): The primary price column to plot (default 'close').
            start_time (str, optional): If provided, restricts data to times >= start_time.
            end_time (str, optional): If provided, restricts data to times <= end_time.
            save_path (str, optional): If provided, the plot is saved to this file; otherwise, it is displayed.
        
        Returns:
            None
        """
        import matplotlib.pyplot as plt

        # Ensure data is loaded.
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Filter the dataframe for the specified day.
        day_df = self.dataframe[self.dataframe['date'] == day].copy()
        if day_df.empty:
            print(f"No data found for day: {day}")
            return

        day_df['time'] = pd.to_datetime(day_df['time'], format='%H:%M:%S').dt.time

        # Convert the input time strings to datetime.time objects.
        if start_time is not None:
            start_time_obj = pd.to_datetime(start_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] >= start_time_obj]
        if end_time is not None:
            end_time_obj = pd.to_datetime(end_time, format='%H:%M').time()
            day_df = day_df[day_df['time'] <= end_time_obj]
        if day_df.empty:
            print(f"No data found for day: {day} within time range {start_time} to {end_time}")
            return

        # Sort the data and reset index for proper plotting.
        day_df.sort_values(by='epoch_time', inplace=True)
        day_df.reset_index(drop=True, inplace=True)
        x = day_df.index  # Use the sequential index as the x-axis

        # Create a figure with three subplots.
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

        # --- Subplot 1: Price and Moving Averages ---
        ax1.plot(x, day_df[price_column], label=price_column, color='black', linewidth=1.5)
        sma_columns = [col for col in day_df.columns if col.startswith('SMA_')]
        for col in sma_columns:
            ax1.plot(x, day_df[col], label=col, linestyle='--')
        ax1.set_title(f"{day} - {price_column.capitalize()} and Moving Averages")
        ax1.set_ylabel("Price")
        ax1.grid(True)
        ax1.legend()

        # --- Subplot 2: future_ratio ---
        if 'future_ratio' in day_df.columns:
            ax2.plot(x, day_df['future_ratio'], label='Future Ratio', color='purple')
            ax2.set_title("Future Ratio (40-min max close / Next 2 Points' Max High)")
            ax2.set_ylabel("Future Ratio")
            ax2.grid(True)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Column 'future_ratio' not found", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("Future Ratio")
        
        # --- Subplot 3: minutes_to_max_close ---
        if 'minutes_to_max_close' in day_df.columns:
            ax3.plot(x, day_df['minutes_to_max_close'], label='Minutes to Max Close', color='green')
            ax3.set_title("Minutes to Max Close")
            ax3.set_ylabel("Minutes")
            ax3.set_xlabel("Time (index; see tick labels for actual time)")
            ax3.grid(True)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Column 'minutes_to_max_close' not found", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Minutes to Max Close")
            ax3.set_xlabel("Index")

        # --- Set x-axis ticks using the 'time' column if available ---
        if 'time' in day_df.columns:
            # Display roughly 10 tick labels to avoid clutter.
            tick_interval = max(1, len(day_df) // 10)
            ax3.set_xticks(x[::tick_interval])
            ax3.set_xticklabels(day_df['time'].iloc[::tick_interval], rotation=45)
        else:
            ax3.set_xlabel("Index")

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def add_sma_condition_flag(self):
        """
        Adds a boolean column 'SMA_25_below_SMA_100_20min' to the dataframe that is True if,
        for the past 20 consecutive minutes, the 25-minute SMA has been below the 100-minute SMA.
        
        This function assumes that the dataframe already contains the columns 'SMA_25' and 'SMA_100'.
        
        Returns:
            pd.DataFrame: The dataframe with the new boolean column added.
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        df = self.dataframe.copy()
        
        # Ensure that the required SMA columns exist.
        if 'SMA_25' not in df.columns or 'SMA_100' not in df.columns:
            raise ValueError("Required SMA columns 'SMA_25' and/or 'SMA_100' not found in dataframe.")
        
        # Create a boolean series: True if SMA_25 is less than SMA_100 at that minute.
        condition = (df['SMA_25'] < df['SMA_100'])
        
        # For each row, check if the condition has held continuously for the last 20 rows.
        # The rolling window (size 20, requiring all 20 values) returns a sum equal to 20 if all are True.
        flag_series = condition.rolling(window=20, min_periods=20).sum() == 20
        
        # For the first 19 rows, where there isn't enough data to check, fill with False.
        df['downward_trend'] = flag_series.fillna(False)
        
        self.dataframe = df
        return df
    
    def save_processed_data(self, file_path):
        """
        Saves selected columns from the processed dataframe to a file that can be easily loaded later.
        
        The saved columns include:
        - Time information: 'date', 'time', 'epoch_time'
        - Price: 'close'
        - Moving averages (columns starting with 'SMA_')
        - Slope values (columns starting with 'slope_')
        - Future ratio: 'future_ratio'
        - Downward trend flag: 'downward_trend'
        
        Parameters:
            file_path (str): The file path (e.g. 'processed_data.csv') where the data will be saved.
        
        Returns:
            None
        """
        if self.dataframe is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        # Define the base columns we always want to include.
        base_cols = ['date', 'time', 'epoch_time', 'close', "volume"]
        
        # Find moving average and slope columns.
        sma_cols = [col for col in self.dataframe.columns if col.startswith('SMA_')]
        slope_cols = [col for col in self.dataframe.columns if col.startswith('slope_')]
        
        # Check for additional columns.
        extra_cols = []
        if 'future_ratio' in self.dataframe.columns:
            extra_cols.append('future_ratio')
        if 'downward_trend' in self.dataframe.columns:
            extra_cols.append('downward_trend')
        if 'minutes_to_max_close' in self.dataframe.columns:
            extra_cols.append('minutes_to_max_close')
            
        
        # Combine all columns.
        columns_to_save = base_cols + sma_cols + slope_cols + extra_cols
        
        # To preserve the original column ordering in the dataframe,
        # filter self.dataframe.columns using our desired list.
        columns_to_save = [col for col in self.dataframe.columns if col in columns_to_save]
        
        # Save the selected columns to a CSV file.
        self.dataframe[columns_to_save].to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}.")



# --- Example Usage ---
if __name__ == "__main__":
    loader = SPYDataLoader("stock_project\\SPY_data.db")
    
    # Load the data.
    df = loader.load_data()
    print("Initial data (first 5 rows):")
    print(df.head(), "\n")
    
    # Optionally, iterate over each day and print a summary.
    print("Daily summaries:")
    for day, day_df in loader.iterate_days():
        print(f"Date: {day}, Records: {len(day_df)}")
    
    # Add SMA columns (for example, 100, 25, and 15 minute SMAs).
    sma_windows = [100, 25, 15]
    df_with_sma = loader.add_sma(sma_windows)
    print("\nData with SMA columns added (first 5 rows):")
    loader.add_rolling_slope(10)
    loader.add_future_metrics()
    df_w_all = loader.add_sma_condition_flag()
    
    print(df_w_all)
    
    # Plot a specific day's data.
    # Replace '2020-01-02' with a date present in your database.
    day_to_plot = '2025-02-12'
    # To display the plot:
    # loader.plot_day(day_to_plot,start_time="8:30", end_time="15:00")
    loader.plot_day_with_extrema(day_to_plot, price_column='close',
                                start_time="8:30", end_time="15:00",
                                min_threshold=-0.1, max_threshold=0.1,
                                save_path=None)

    loader.plot_day_future_metrics(day_to_plot, price_column='close',
                                start_time="8:30", end_time="15:00",
                                save_path=None)
    loader.save_processed_data("dataset_w_features.csv")
    # Or to save the plot instead, provide a save path:
    # loader.plot_day(day_to_plot, save_path="plot_2020-01-02.png")
