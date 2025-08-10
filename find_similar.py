import pandas as pd
import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def cross_correlation(series1, series2):
    """
    Calculates the cross-correlation between two time series using FFT for efficiency.
    """
    n1 = len(series1)
    n2 = len(series2)
    n_overlap = min(n1, n2)

    if n_overlap == 0:
        return np.array([0])  # Handle empty series

    # Normalize series to zero mean and unit variance (avoiding division by zero)
    series1_norm = (series1 - np.mean(series1)) / (np.std(series1) + 1e-8)
    series2_norm = (series2 - np.mean(series2)) / (np.std(series2) + 1e-8)

    # Pad to avoid circular convolution issues
    f1 = fft(series1_norm, n=2 * n_overlap)
    f2 = fft(np.conjugate(series2_norm), n=2 * n_overlap)
    ccf = ifft(f1 * f2)
    ccf = ccf[:n_overlap].real

    return ccf

def calculate_series_similarity(df, selected_date, column_name, start_time, stop_time, similarity_measure='xcorr'):
    """
    Calculates similarity between a selected time series segment and the same segment
    on all other days in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with 'date', 'time', and at least one numerical column.
        selected_date (str): Date in 'YYYY-MM-DD' format for the reference series.
        column_name (str): Column to compare (e.g., 'slope_10').
        start_time (str): Start time in 'HH:MM:SS' for similarity measurement.
        stop_time (str): Stop time in 'HH:MM:SS' for similarity measurement.
        similarity_measure (str): 'xcorr' (default) or 'mse'.

    Returns:
        tuple: (top_similar_days, similarity_scores)
    """
    # Prepare the reference series for the selected day
    reference_day_df = df[df['date'] == selected_date].copy()
    if reference_day_df.empty:
        return "Selected date not found in dataframe.", None

    reference_day_df['datetime'] = pd.to_datetime(reference_day_df['date'] + ' ' + reference_day_df['time'])
    reference_series = reference_day_df[(reference_day_df['time'] >= start_time) & 
                                         (reference_day_df['time'] <= stop_time)][column_name].values

    if len(reference_series) == 0:
        return "No data for the selected time range on the selected date.", None

    similarity_scores = {}
    unique_dates = df['date'].unique()

    for other_date in unique_dates:
        if other_date == selected_date:
            continue

        other_day_df = df[df['date'] == other_date].copy()
        other_day_df['datetime'] = pd.to_datetime(other_day_df['date'] + ' ' + other_day_df['time'])
        comparison_series = other_day_df[(other_day_df['time'] >= start_time) & 
                                         (other_day_df['time'] <= stop_time)][column_name].values

        # If data is missing or lengths differ, assign an extreme value.
        if len(comparison_series) == 0 or len(comparison_series) != len(reference_series):
            similarity_scores[other_date] = float('inf') if similarity_measure.lower() == 'mse' else -float('inf')
            continue

        if similarity_measure.lower() == 'xcorr':
            ccf_values = cross_correlation(reference_series, comparison_series)
            # Higher maximum absolute correlation indicates a better match.
            metric = np.max(np.abs(ccf_values))
        elif similarity_measure.lower() == 'mse':
            # Lower mean squared error indicates a better match.
            metric = np.mean((reference_series - comparison_series) ** 2)
        else:
            raise ValueError("Unsupported similarity measure: " + similarity_measure)

        similarity_scores[other_date] = metric

    # Sort and select the top 5 days based on the chosen metric.
    if similarity_measure.lower() == 'xcorr':
        ranked_days = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    elif similarity_measure.lower() == 'mse':
        ranked_days = sorted(similarity_scores.items(), key=lambda item: item[1])
    
    top_similar_days = [day for day, score in ranked_days[:5]]
    top_similarity_values = [score for day, score in ranked_days[:5]]

    return top_similar_days, top_similarity_values

def plot_similar_days(df, selected_date, top_similar_days, plot_start_time="08:30:00", plot_stop_time="15:00:00"):
    """
    Creates a two-panel plot:
      - Top panel: 'close' values for the top similar days (with colors varying from red to blue)
                   and overlays the selected day in green.
      - Bottom panel: 'slope_10' values plotted similarly.
      
    The x-axis ticks are set at 15-minute intervals, and the plotted time range is controlled by 
    plot_start_time and plot_stop_time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax_close, ax_slope = axes

    # Create a red-to-blue color gradient for the similar days.
    n = len(top_similar_days)
    if n == 1:
        colors = [(1, 0, 0)]
    else:
        colors = [(1 - i/(n-1), 0, i/(n-1)) for i in range(n)]  # red to blue

    # Plot each similar day with its corresponding color.
    for idx, day in enumerate(top_similar_days):
        day_df = df[df['date'] == day].copy()
        day_df = day_df[(day_df['time'] >= plot_start_time) & (day_df['time'] <= plot_stop_time)]
        if day_df.empty:
            continue

        # Convert 'time' strings into datetime objects (dummy date is used)
        times = pd.to_datetime(day_df['time'], format='%H:%M:%S')
        ax_close.plot(times, day_df['close'], label=day, alpha=0.7, color=colors[idx])
        ax_slope.plot(times, day_df['slope_10'], label=day, alpha=0.7, color=colors[idx])

    # Overlay the selected (original) day in green with a thicker line.
    orig_df = df[df['date'] == selected_date].copy()
    orig_df = orig_df[(orig_df['time'] >= plot_start_time) & (orig_df['time'] <= plot_stop_time)]
    if not orig_df.empty:
        times_orig = pd.to_datetime(orig_df['time'], format='%H:%M:%S')
        ax_close.plot(times_orig, orig_df['close'],
                      label=f"{selected_date} (Original)",
                      color='green', linewidth=2.5)
        ax_slope.plot(times_orig, orig_df['slope_10'],
                      label=f"{selected_date} (Original)",
                      color='green', linewidth=2.5)

    # Format the x-axis with 15-minute intervals.
    locator = mdates.MinuteLocator(interval=15)
    formatter = mdates.DateFormatter("%H:%M")
    ax_slope.xaxis.set_major_locator(locator)
    ax_slope.xaxis.set_major_formatter(formatter)

    # Set titles, labels, and grid for clarity.
    ax_close.set_title("Close Values for Top Similar Days and Original Day")
    ax_close.set_ylabel("Close")
    ax_close.grid(True)
    ax_close.legend(loc="best")

    ax_slope.set_title("Slope 10 Values for Top Similar Days and Original Day")
    ax_slope.set_ylabel("Slope 10")
    ax_slope.set_xlabel("Time")
    ax_slope.grid(True)
    ax_slope.legend(loc="best")

    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Load dataset (update the path as needed)
    df = pd.read_csv("C:\\Users\\deade\\OneDrive\\Desktop\\data_science\\temp_normed_dataset.csv")

    # Ensure 'date' is a datetime object
    df['date'] = pd.to_datetime(df['date'])

    # --- Date Range Filtering ---
    start_date_filter = '2023-01-02'
    end_date_filter = '2025-06-02'
    start_date_filter = pd.to_datetime(start_date_filter)
    end_date_filter = pd.to_datetime(end_date_filter)
    df_filtered = df[(df['date'] >= start_date_filter) & (df['date'] <= end_date_filter)].copy()
    print(df_filtered)

    if df_filtered.empty:
        print(f"No data found within the date range: {start_date_filter.strftime('%Y-%m-%d')} to {end_date_filter.strftime('%Y-%m-%d')}")
    else:
        df = df_filtered
        # Convert 'date' back to string for easier comparisons
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Parameters for similarity measurement (e.g. 08:30 to 10:00)
        selected_date = '2025-06-04'  # Must be within the filtered range
        column_name = 'slope_10'      # Column used for similarity computation
        sim_start_time = '08:30:00'
        sim_stop_time = '09:00:00'

        # Choose the similarity measure: 'xcorr' or 'mse'
        similarity_measure = 'mse'  # or 'xcorr'

        top_days, similarity_values = calculate_series_similarity(
            df, selected_date, column_name, sim_start_time, sim_stop_time, similarity_measure=similarity_measure
        )

        if isinstance(top_days, str):  # An error message was returned
            print(top_days)
        elif top_days:
            if similarity_measure.lower() == 'xcorr':
                print(f"Top 5 days most similar to {selected_date} (using '{column_name}' from {sim_start_time} to {sim_stop_time} based on cross-correlation):")
            elif similarity_measure.lower() == 'mse':
                print(f"Top 5 days most similar to {selected_date} (using '{column_name}' from {sim_start_time} to {sim_stop_time} based on mean squared error):")
            for i, (day, score) in enumerate(zip(top_days, similarity_values), start=1):
                print(f"{i}. Day: {day}, Similarity Score: {score:.4f}")

            # Plot similar days using a fixed plot range from 08:30 to 15:00.
            plot_similar_days(df, selected_date, top_days, plot_start_time="08:30:00", plot_stop_time="15:00:00")
        else:
            print("No similar days found (or an error occurred) within the date range.")
