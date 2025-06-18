#!/usr/bin/env python3
import argparse
import os
import pickle
from datetime import datetime, timedelta, time
import concurrent.futures
import pytz
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)

# Global threshold constant for classification.
# RATIO_THRESHOLD = 1.002
# RATIO_THRESHOLD = 1.0015
RATIO_THRESHOLD = 0.9975
# SLOPE_THRESHOLD = 0.0005
SLOPE_THRESHOLD = 0.0000016536
SLOPE_THRESHOLD = 0.00002
if RATIO_THRESHOLD > 1:
    EVAL_RATIO = 1.0001
else:
    EVAL_RATIO = 0.999
# EVAL_RATIO = 0
# EVAL_RATIO = 0.0000016536


def load_and_normalize_data(norm_time):
    # Original LOCAL-SQLITE logic:
    import sqlite3
    import os
    norm_time = datetime.strptime(norm_time, '%H:%M').time()
    
    repo_root = "C:\\Users\\deade\\OneDrive\\Desktop\\data_science\\stock_project"
    db_path = os.path.join(repo_root, f"big_SPY_data.db")
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
    print(f"Dataframe {len(df)} after concat")
    df.sort_values(by='epoch_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

    # print(f"Dataframe {len(df)} after sort")
    # Normalize based on the open value
    norm_cols = ['close'] + [col for col in df.columns if col.startswith('SMA_')]
    def normalize_group(group):
        # Identify the row(s) with the normalization time
        base_rows = group[group['time'] == norm_time]
        if base_rows.empty:
            print(f"Warning: No row found at time '{norm_time}' for date {group.name}. Skipping normalization for this day.")
            return group

        # Get the close value at norm_time to use as the base
        base_value = base_rows.iloc[0]['close']
        if base_value == 0:
            print(f"Warning: Close value at {norm_time} is 0 for date {group.name}. Skipping normalization for this day.")
            return group

        # Normalize the designated columns in norm_cols
        group.loc[:, norm_cols] = group.loc[:, norm_cols] / base_value

        # Additionally, normalize 'high' and 'low' if they are present and not already in norm_cols
        for col in ['high', 'low']:
            if col in group.columns and col not in norm_cols:
                group[col] = group[col] / base_value

        return group


    df_normalized = df.groupby('date', group_keys=False).apply(normalize_group)
    # Calculate the count of data points for each day
    day_counts = df_normalized.groupby('date').size()
    
    # Compute statistics: mean, median, and mode
    mean_count = day_counts.mean()
    median_count = day_counts.median()
    mode_series = day_counts.mode()
    mode_count = mode_series.iloc[0] if not mode_series.empty else None

    print("Data points per day statistics:")
    print(f"  Mean: {mean_count:.2f}")
    print(f"  Median: {median_count:.2f}")
    print(f"  Mode: {mode_count}")

    # Filter out days with a data point count less than the mean
    valid_dates = day_counts[day_counts >= 600].index
    df_filtered = df_normalized[df_normalized['date'].isin(valid_dates)]

    cutoff_date = datetime.strptime("2008-10-01", "%Y-%m-%d").date()
    df_filtered = df_filtered[df_filtered['date'] >= cutoff_date]

    # Print out the first and last date in the filtered dataset
    first_date = df_filtered['date'].min()
    last_date = df_filtered['date'].max()
    print(f"First date in filtered dataset: {first_date}")
    print(f"Last date in filtered dataset: {last_date}")
    print(f"Dataframe {len(df_filtered)} after filter")
    unique_dates = df_filtered['date'].unique()
    # for idx, date in enumerate(unique_dates):
    #     print(f"{idx}: {date}")
    print(f"{len(unique_dates)} days")
    # exit()
    return df_filtered
    # print(f"Dataframe {len(df_normalized)} after normalization")
    # print(df_normalized)
    plot = False
    if plot:
        # Define time bounds for plotting
        start_time = time(8, 30)
        end_time = time(15, 0)

        # Get the first three unique dates (assuming 'date' is already sorted or sortable)
        unique_dates = sorted(df_normalized['date'].unique())
        first_three_dates = unique_dates[:3]

        # Set up a subplot for each day
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

        for i, day in enumerate(first_three_dates):
            # Filter the day's data for the specified time range
            day_df = df_normalized[(df_normalized['date'] == day) &
                        (df_normalized['time'] >= start_time) &
                        (df_normalized['time'] <= end_time)]
            if day_df.empty:
                print(f"No data for {day} between {start_time} and {end_time}")
                continue

            # Sort the data by time for a proper plot
            day_df = day_df.sort_values(by='time')

            day_df['time'] = day_df['time'].apply(lambda dt: dt.strftime('%H:%M'))

            # Plot high, low, and close
            axes[i].plot(day_df['time'], day_df['high'], label='High')
            axes[i].plot(day_df['time'], day_df['low'], label='Low')
            axes[i].plot(day_df['time'], day_df['close'], label='Close')
            axes[i].set_title(f"Data on {day}")
            axes[i].set_ylabel("Normalized Price")
            axes[i].legend()

        axes[-1].set_xlabel("Time")
        plt.tight_layout()
        plt.show()

    return df_normalized
    # """
    # Loads the CSV file and normalizes the 'close' and SMA columns based on the close value
    # at a specified time for each day.

    # Parameters:
    #     file_path (str): Path to the CSV file.
    #     norm_time (str): Normalization time in 'HH:MM' format.

    # Returns:
    #     pd.DataFrame: Normalized DataFrame.
    # """
    # df = pd.read_csv(file_path)
    # df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    # norm_time = pd.to_datetime(norm_time, format='%H:%M').time()

    # norm_cols = ['close'] + [col for col in df.columns if col.startswith('SMA_')]

    # def normalize_group(group):
    #     base_rows = group[group['time'] == norm_time]
    #     if base_rows.empty:
    #         print(f"Warning: No row found at time '{norm_time}' for date {group.name}. Skipping normalization for this day.")
    #         return group
    #     base_value = base_rows.iloc[0]['close']
    #     if base_value == 0:
    #         print(f"Warning: Close value at {norm_time} is 0 for date {group.name}. Skipping normalization for this day.")
    #         return group
    #     group.loc[:, norm_cols] = group.loc[:, norm_cols] / base_value
    #     return group

    # df_normalized = df.groupby('date', group_keys=False).apply(normalize_group)
    # return df_normalized


def filter_timeframe(df, start_time, end_time):
    """
    Filters the DataFrame to include only rows within the specified timeframe.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        start_time (str): Start time in 'HH:MM' format.
        end_time (str): End time in 'HH:MM' format.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    start_time = pd.to_datetime(start_time, format='%H:%M').time()
    end_time = pd.to_datetime(end_time, format='%H:%M').time()
    filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy().reset_index(drop=True)
    return filtered_df


def split_dataset(df, train_frac=0.85, val_frac=0.15, test_frac=None, random_state=42):
    """
    Shuffles and splits the DataFrame into training, validation, and test subsets.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        train_frac (float): Fraction of data for training.
        val_frac (float): Fraction of data for validation.
        test_frac (float): Fraction of data for testing.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if test_frac == None:
        total = train_frac + val_frac
    else:
        total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-5:
        raise ValueError("The train, validation, and test fractions must sum to 1.")
    
    # mask = df['epoch_time'] > 1735711200 # Jan 1. 2025
    # mask = df['epoch_time'] > 1727758800 # Oct 1. 2024
    # first_row = df[mask].iloc[0]
    # first_row = -1

    # print()
    # print(df.iloc[first_row.name])
    # exit()
    # df_shuffled = df.iloc[:first_row.name].sample(frac=1, random_state=random_state).reset_index(drop=True)
    # print(f"Pre split: {len(df)}")
    df_shuffled = df
    n = len(df_shuffled)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)
    # print(f"N: {n}")
    # print(f"Train end = {train_end}, Val end = {val_end}")

    # train_df = df_shuffled.iloc[:train_end].copy()
    train_df = df_shuffled.iloc[:train_end].sample(frac=1, random_state=random_state).reset_index(drop=True).copy()
    val_df = df_shuffled.iloc[train_end:val_end].copy()
    # print(f"Post split: T - {len(train_df)} + V - {len(val_df)} =  {len(train_df) + len(val_df)}")
    # print(val_df)
    # print(len(df), len(train_df), len(val_df))
    # exit()
    if test_frac == None:
        test_df = df.iloc[-800:].copy()
    else:
        test_df = df_shuffled.iloc[val_end:].copy()

    return train_df, val_df, test_df


def train_random_forest(train_df, random_state=42):
    """
    Trains a RandomForest classifier on the training DataFrame.

    Features used:
      - 'close'
      - All columns starting with 'SMA_'
      - All columns starting with 'slope_'
      - 'downward_trend' (if available)
      - Additional run-length features if available.

    The target is set to 1 if 'future_ratio' > RATIO_THRESHOLD, else 0.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Trained classifier.
    """
    # train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()

    features = ['close']
    features += [col for col in train_df.columns if col.startswith('SMA_')]
    features += [col for col in train_df.columns if col.startswith('slope_')]
    features += [col for col in train_df.columns if col.startswith('d2_')]

    if 'downward_trend' in train_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in train_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in train_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in train_df.columns:
        features.append('positive_slope_run_length')

    if RATIO_THRESHOLD > 1:
        train_df['target'] = (train_df['future_ratio'] > RATIO_THRESHOLD).astype(int)
    else:
        train_df['target'] = (train_df['future_ratio'] < RATIO_THRESHOLD).astype(int)
    # train_df['target'] = (train_df['future_slope'] > SLOPE_THRESHOLD).astype(int)
    
    X_train = train_df[features]
    y_train = train_df['target']

    clf = RandomForestClassifier(random_state=random_state,
                                 n_estimators=200,
                                 max_features=10,
                                #  class_weight={0:10, 1:1},
                                 class_weight=None,
                                 n_jobs=4)
    mean = 0.9983045337299954
    median = 0.9991301203341799
    sigma = 0.0029079728310040004
    
    # min - 0.9026200277163384
    # max - 1.070147309349634
    # most common bin was close tp 0.9998
    # So far, 0.25 around the mean has precision of 75% (8% Recall)
    # So far, 0.10 around the mean has precision of 72% (16% Recall)
    # So far, 0.10 around the median has precision of 72% (18% Recall)
    # lower_bound = median - 0.2 * sigma
    # upper_bound = median + 0.2 * sigma
    sample_weights = np.ones(len(train_df), dtype=float)
    sample_weights[train_df['future_ratio'] < 1.0] *= 2.0
    sample_weights[train_df['future_ratio'] < 0.998] *= 5.0
    sample_weights[train_df['future_ratio'] > 1.01] *= 2.0
    sample_weights[train_df['future_ratio'] > 1.02] *= 5.0
    # sample_weights[train_df['future_ratio'] > 1.0] *= 5.0
    # sample_weights[train_df['future_ratio'] > 1.0005] *= 7.0
    # sample_weights[train_df['future_ratio'] > 1.0010] *= 11.0
    # sample_weights[train_df['future_ratio'] > 1.0015] *= 13.0
    # sample_weights[train_df['future_ratio'] < RATIO_THRESHOLD - 0.0005] *= 5.0
    # sample_weights[train_df['future_ratio'] > mean + (0.5 * sigma)] *= 2.0
    # sample_weights[train_df['future_ratio'] > mean + sigma] *= 10.0
    # sample_weights[train_df['future_ratio'] > mean + (1.5 * sigma)] *= 10.0
    # sample_weights[train_df['future_ratio'] > mean + (2.0 * sigma)] *= 20.0
    # sample_weights[train_df['future_ratio'] < mean - (0.5 * sigma)] *= 2.0
    # sample_weights[train_df['future_ratio'] < mean - sigma] *= 2.0
    # sample_weights[train_df['future_ratio'] < mean - (1.5 * sigma)] *= 5.0
    # sample_weights[train_df['future_ratio'] < mean - (2.0 * sigma)] *= 10.0
    # print(f"sigmas: {0.5 * sigma + mean:.4f}; {sigma + mean:.4f}; {1.5 * sigma + mean:.4f}; {2 * sigma + mean:.4f}; ")
    # print(f"        {-0.5 * sigma + mean:.4f}; {-sigma + mean:.4f}; {-1.5 * sigma + mean:.4f}; {-2 * sigma + mean:.4f}; ")
    # mask_outside = (train_df['future_ratio'] > lower_bound) & (train_df['future_ratio'] < upper_bound)
    # sample_weights[mask_outside] = 0

    clf.fit(X_train, y_train, sample_weight=sample_weights)

    return clf


def train_random_forest_regressor(train_df, random_state=42):
    """
    Trains a RandomForest regressor on the training DataFrame.

    Features used:
      - 'close'
      - All columns starting with 'SMA_'
      - All columns starting with 'slope_'
      - 'downward_trend' (if available)
      - Additional run-length features if available.

    The target is the 'minutes_to_max_close' column.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestRegressor: Trained regressor.
    """
    # train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()

    features = ['close']
    features += [col for col in train_df.columns if col.startswith('SMA_')]
    features += [col for col in train_df.columns if col.startswith('slope_')]
    features += [col for col in train_df.columns if col.startswith('d2_')]

    if 'downward_trend' in train_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in train_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in train_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in train_df.columns:
        features.append('positive_slope_run_length')

    # Drop rows with missing target values
    train_df = train_df.dropna(subset=['minutes_to_max_close'])

    X_train = train_df[features]
    y_train = train_df['minutes_to_max_close']

    reg = RandomForestRegressor(random_state=random_state, n_estimators=100, min_samples_split=2, verbose=1)
    reg.fit(X_train, y_train)

    return reg

def add_slope_run_length(df, slope_column='slope_10'):
    """
    Adds run-length columns for negative and positive slope values computed separately for each day.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        slope_column (str): Name of the slope column for computing run lengths.

    Returns:
        pd.DataFrame: DataFrame with added run-length columns.
    """
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
    """
    Adds a column that counts the run length (in minutes) during which the 25-minute SMA
    is below the 100-minute SMA, computed separately for each day.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        sma25_col (str): Column name for the 25-minute SMA.
        sma100_col (str): Column name for the 100-minute SMA.
        new_col (str): Name of the new column to store run-length counts.

    Returns:
        pd.DataFrame: DataFrame with the added run-length column.
    """
    def compute_run_length(group):
        condition = group[sma25_col] < group[sma100_col]
        group[new_col] = condition.groupby((~condition).cumsum()).cumcount() + 1
        group.loc[~condition, new_col] = 0
        return group

    df_with_run = df.groupby('date', group_keys=False).apply(compute_run_length)
    return df_with_run

def evaluate_model(clf, df, start_time=None, end_time=None):
    """
    Evaluates the classifier using the provided DataFrame and prints performance metrics.
    Optionally, you can specify a timeframe by providing the 'start_time' and 'end_time'
    in 'HH:MM' format. Only rows within this timeframe will be evaluated.

    Parameters:
        clf: Trained classifier.
        df (pd.DataFrame): DataFrame for evaluation.
        start_time (str, optional): Start time in 'HH:MM' format.
        end_time (str, optional): End time in 'HH:MM' format.
    """
    # If a timeframe is provided, filter the DataFrame accordingly.
    if start_time is not None and end_time is not None:
        df = filter_timeframe(df, start_time, end_time)

    features = ['close']
    features += [col for col in df.columns if col.startswith('SMA_')]
    features += [col for col in df.columns if col.startswith('slope_')]
    features += [col for col in df.columns if col.startswith('d2_')]
    if 'downward_trend' in df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in df.columns:
        features.append('positive_slope_run_length')

    df = df.copy()
    
    if RATIO_THRESHOLD > 1:
        df['target'] = (df['future_ratio'] > EVAL_RATIO).astype(int)
    else:
        df['target'] = (df['future_ratio'] < EVAL_RATIO).astype(int)

    X = df[features]
    y_true = df['target']
    y_pred = clf.predict(X)

    print("Classification Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def evaluate_regression_model(reg, df):
    """
    Evaluates the regressor using the provided DataFrame and prints regression metrics.

    Parameters:
        reg: Trained regressor.
        df (pd.DataFrame): DataFrame for evaluation.
    """
    features = ['close']
    features += [col for col in df.columns if col.startswith('SMA_')]
    features += [col for col in df.columns if col.startswith('slope_')]
    features += [col for col in df.columns if col.startswith('d2_')]
    if 'downward_trend' in df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in df.columns:
        features.append('positive_slope_run_length')

    df = df.copy()
    df = df.dropna(subset=['minutes_to_max_close'])
    X = df[features]
    y_true = df['minutes_to_max_close']
    y_pred = reg.predict(X)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("Regression Metrics:")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"R2 Score: {r2:.3f}")


def evaluate_day_predictions(clf, df, day):
    """
    Evaluates classification predictions for a given day by plotting the normalized close values
    and comparing predicted vs. actual classifications.

    Parameters:
        clf: Trained classifier.
        df (pd.DataFrame): Complete DataFrame with features.
        day (str): Day in 'YYYY-MM-DD' format.
    """
    day = datetime.strptime(day, "%Y-%m-%d").date()

    day_df = df[df['date'] == day].copy()
    # print(day_df.iloc[0:60])
    # print(day_df[['close', 'shift_close', 'future_ratio']].iloc[0:60])
    day_df.to_csv(f"pred{day}.csv")
    if day_df.empty:
        print(f"No data available for day: {day}")
        return

    day_df.sort_values(by='epoch_time', inplace=True)

    features = ['close']
    features += [col for col in day_df.columns if col.startswith('SMA_')]
    features += [col for col in day_df.columns if col.startswith('slope_')]
    features += [col for col in day_df.columns if col.startswith('d2_')]
    if 'downward_trend' in day_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in day_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in day_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in day_df.columns:
        features.append('positive_slope_run_length')

    X_day = day_df[features]
    predictions = clf.predict(X_day)

    if RATIO_THRESHOLD > 1:
        actual = (day_df['future_ratio'] > EVAL_RATIO).astype(int)
    else:
        actual = (day_df['future_ratio'] < EVAL_RATIO).astype(int)
    # actual = (day_df['future_slope'] > EVAL_RATIO).astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    ax1.plot(day_df.index, day_df['close'], label='Normalized Close', color='blue')
    ax1.set_title(f"Close Values on {day}")
    ax1.set_ylabel("Normalized Close")
    ax1.legend()

    # Shade regions with contiguous predictions of 1 (at least 3 in a row)
    runs = []
    n = len(predictions)
    i = 0
    while i < n:
        if predictions[i] == 1:
            start = i
            while i < n and predictions[i] == 1:
                i += 1
            end = i - 1
            if (end - start + 1) >= 3:
                runs.append((start, end))
        else:
            i += 1

    for start, end in runs:
        ax1.axvspan(day_df.index[start], day_df.index[end], color='green', alpha=0.3)

    ax2.plot(day_df.index, predictions, label='Predicted', color='green')
    ax2.plot(day_df.index, actual, label='Actual', color='red', marker='x', linestyle='--')
    ax2.set_title(f"Predicted vs Actual Classification on {day}")
    ax2.set_ylabel("Classification (0 or 1)")
    ax2.set_xlabel("Data Index (Time progression)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def evaluate_day_regression(reg, df, day):
    """
    Evaluates regression predictions for a given day by plotting the actual vs. predicted
    'minutes_to_max_close' values.

    Parameters:
        reg: Trained regressor.
        df (pd.DataFrame): Complete DataFrame with features.
        day (str): Day in 'YYYY-MM-DD' format.
    """
    day_df = df[df['date'] == day].copy()
    if day_df.empty:
        print(f"No data available for day: {day}")
        return

    day_df.sort_values(by='epoch_time', inplace=True)

    features = ['close']
    features += [col for col in day_df.columns if col.startswith('SMA_')]
    features += [col for col in day_df.columns if col.startswith('slope_')]
    features += [col for col in day_df.columns if col.startswith('d2_')]
    if 'downward_trend' in day_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in day_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in day_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in day_df.columns:
        features.append('positive_slope_run_length')

    X_day = day_df[features]
    predictions = reg.predict(X_day)
    actual = day_df['minutes_to_max_close']

    plt.figure(figsize=(12, 6))
    plt.plot(day_df.index, actual, label="Actual minutes_to_max_close", color='blue')
    plt.plot(day_df.index, predictions, label="Predicted minutes_to_max_close", color='red', marker='x', linestyle='--')
    plt.xlabel("Data Index")
    plt.ylabel("Minutes to Max Close")
    plt.title(f"Regression: Actual vs Predicted minutes_to_max_close on {day}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def display_feature_importances(model, features):
    """
    Displays and plots the feature importances of the trained model.

    Parameters:
        model: Trained model (classifier or regressor).
        features (list): List of feature names.
    """
    importances = model.feature_importances_
    df_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_importances.sort_values(by='Importance', ascending=False, inplace=True)

    print("Feature Importances:")
    print(df_importances)

    plt.figure(figsize=(10, 6))
    plt.bar(df_importances['Feature'], df_importances['Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

def slope_of_best_fit(values):
    """
    Compute the slope of a best-fit line for a 1D array of values,
    assuming x-coordinates [0, 1, 2, ..., len(values) - 1].

    Parameters:
        values (array-like): The y-values for the best-fit line.
    
    Returns:
        float: The slope of the best-fit line.
    """
    x = np.arange(len(values))
    y = values
    slope, _ = np.polyfit(x, y, 1)
    return slope

def future_slope_of_best_fit(values):
    """
    Given an array of values (in reversed order),
    compute the slope of the best-fit line and then
    multiply by -1 to obtain the forward-looking slope.
    """
    return -slope_of_best_fit(values)

def add_future_slope(df, column='close', window=40, group_by_date=True, scale_constant=1.0):
    """
    Add a column 'future_slope' to the DataFrame that contains the slope
    of a best-fit line through the next `window` samples of the specified column.
    Optionally, divides the computed slope by a constant.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing a time-series.
        column (str): Name of the column to compute the future slope on.
        window (int): The number of future samples to include.
        group_by_date (bool): If True, compute the future slope separately for each day,
                              assuming the DataFrame has a 'date' column.
        scale_constant (float): Constant by which to divide the computed slope.
                                Defaults to 1.0 (i.e. no scaling).
    
    Returns:
        pd.DataFrame: The DataFrame with an added 'future_slope' column.
    """
    
    def compute_future_slope(group):
        # Ensure the group is sorted in ascending order by time.
        group = group.sort_values(by='time').copy()
        # Reverse the series, apply rolling window, then reverse back.
        group['future_slope'] = (
            group[column][::-1]
            .rolling(window=window, min_periods=window)
            .apply(future_slope_of_best_fit, raw=True)[::-1]
        )
        # Divide by the constant.
        group['future_slope'] = group['future_slope'] / scale_constant
        return group
    
    if group_by_date and 'date' in df.columns:
        # Compute the future slope for each day separately.
        df = df.groupby('date', group_keys=False).apply(compute_future_slope)
    else:
        df = compute_future_slope(df)
    
    return df


def compute_daily_slopes(group, slope_column, lookback):
    """
    For a given day group, compute the rolling slope of a chosen column 
    using a specified lookback window.

    Parameters:
        group (pd.DataFrame): A DataFrame representing a single day's data.
        slope_column (str): The name of the column on which to compute slopes.
        lookback (int): The window size for the rolling calculation.

    Returns:
        pd.DataFrame: The input group with an added 'chosen_slope' column.
    """
    # Make sure the group is sorted by time, if applicable.
    group = group.sort_values(by='time')
    
    # Compute the rolling slope using our custom function
    group[f'slope_{lookback}'] = group[slope_column].rolling(window=lookback).apply(slope_of_best_fit, raw=True)
    return group

def plot_day_data(day_str, df):
    """
    For a given day (as a string 'YYYY-MM-DD'), this function filters the DataFrame 
    to include only rows between 08:30 and 15:00, then plots:
      - In the top subplot: the 'close' values with SMA_100, SMA_25, and SMA_15.
      - In the middle subplot: slope_10 and slope_100 values.
      - In the bottom subplot: future_slope values.
    
    Parameters:
        day_str (str): The day to plot (format 'YYYY-MM-DD').
        df (pd.DataFrame): The DataFrame containing at least the following columns:
                           'date', 'time', 'close', 'SMA_100', 'SMA_25', 'SMA_15',
                           'slope_10', 'slope_100', and 'future_slope'.
    """
    # Convert the day string to a date object
    try:
        day_date = datetime.strptime(day_str, '%Y-%m-%d').date()
    except Exception as e:
        print(f"Error parsing day string '{day_str}': {e}")
        return

    # Define the time bounds for filtering
    start_time = time(8, 30)
    end_time = time(15, 0)
    
    # Filter the DataFrame for the specified day and time range
    day_df = df[(df['date'] == day_date) &
                (df['time'] >= start_time) &
                (df['time'] <= end_time)]
    
    if day_df.empty:
        print(f"No data available for {day_str} between {start_time} and {end_time}.")
        return
    
    # Create a new 'datetime' column by combining 'date' and 'time' for proper plotting.
    day_df = day_df.copy()
    day_df['datetime'] = day_df.apply(
        lambda row: datetime.combine(row['date'], row['time']),
        axis=1
    )
    
    # Set up three subplots (vertically stacked)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12), sharex=True)
    
    # --- Top Subplot: Close and SMA values ---
    ax1.plot(day_df['datetime'], day_df['close'], label='Close')
    ax1.plot(day_df['datetime'], day_df['SMA_100'], label='SMA_100')
    ax1.plot(day_df['datetime'], day_df['SMA_25'], label='SMA_25')
    ax1.plot(day_df['datetime'], day_df['SMA_15'], label='SMA_15')
    ax1.set_title(f"Close and SMA values for {day_str}")
    ax1.set_ylabel("Price")
    ax1.legend()
    
    # --- Middle Subplot: Slope values ---
    ax2.plot(day_df['datetime'], day_df['slope_10'], label='Slope 10')
    ax2.plot(day_df['datetime'], day_df['slope_100'], label='Slope 100')
    ax2.set_title(f"Slope values for {day_str}")
    ax2.set_ylabel("Slope")
    ax2.legend()
    
    # --- Bottom Subplot: Future Slope values ---
    ax3.plot(day_df['datetime'], day_df['d2_10'], label='d^2 10')
    ax3.plot(day_df['datetime'], day_df['d2_100'], label='d^2 100')
    ax3.set_title(f"d^2 for {day_str}")
    ax3.set_ylabel("d^2")
    ax3.set_xlabel("Time")
    ax3.legend()
    # ax3.plot(day_df['datetime'], day_df['future_slope'], label='Future Slope')
    # ax3.set_title(f"Future Slope for {day_str}")
    # ax3.set_ylabel("Future Slope")
    # ax3.set_xlabel("Time")
    # ax3.legend()
    
    # Format the x-axis to show only the time (HH:MM)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def add_future_metrics(df, future_window=40):
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

    if df is None:
        raise ValueError("Data not loaded. Please call load_data() first.")

    n = len(df)


    if RATIO_THRESHOLD > 1:
        # --- Max case: use next 2 points and a future window ---
        df['max_high_2'] = pd.concat([df['high'].shift(-1), df['high'].shift(-2)], axis=1).max(axis=1)

        close_array = df['close'].values
        epoch_array = df['epoch_time'].values  # assume epoch_time is in seconds
        future_max_close = np.full(n, np.nan, dtype=float)
        minutes_to_max = np.full(n, np.nan, dtype=float)

        # Loop over each row. For row i, use the window starting at i+3.
        for i in range(n):
            start_idx = i + 3
            end_idx = i + 3 + future_window  # window spans future_window rows
            if end_idx > n:
                continue

            window_close = close_array[start_idx:end_idx]
            max_close = window_close.max()
            future_max_close[i] = max_close

            # Get the first occurrence index of the maximum close within the window.
            idx_in_window = int(np.argmax(window_close))
            target_idx = start_idx + idx_in_window

            # Compute the time difference (in minutes) using epoch_time (which is in seconds).
            dt_minutes = (epoch_array[target_idx] - epoch_array[i]) / 60.0
            minutes_to_max[i] = dt_minutes

        df['future_ratio'] = future_max_close / df['max_high_2']
        df.drop(columns=['max_high_2'], inplace=True)
        df['minutes_to_max_close'] = minutes_to_max

    else:
        df['min_low_2'] = pd.concat([df['close'].shift(-1), df['close'].shift(-2)], axis=1).min(axis=1)
        # print(df[['close','min_low_2']])
        close_array = df['close'].values
        epoch_array = df['epoch_time'].values  # assume epoch_time is in seconds
        future_min_close = np.full(n, np.nan, dtype=float)
        minutes_to_min = np.full(n, np.nan, dtype=float)

        # Loop over each row. For row i, use the window starting at i+3.
        for i in range(n):
            start_idx = i + 3
            end_idx = i + 3 + future_window  # window spans future_window rows
            if end_idx > n:
                continue

            window_close = close_array[start_idx:end_idx]
            min_close = window_close.min()
            future_min_close[i] = min_close

            # Get the first occurrence index of the minimum close within the window.
            idx_in_window = int(np.argmin(window_close))
            target_idx = start_idx + idx_in_window

            # Compute the time difference (in minutes) using epoch_time (which is in seconds).
            dt_minutes = (epoch_array[target_idx] - epoch_array[i]) / 60.0
            minutes_to_min[i] = dt_minutes

        df['future_ratio'] = future_min_close / df['min_low_2']
        df.drop(columns=['min_low_2'], inplace=True)
        df['minutes_to_min_close'] = minutes_to_min
        # # # --- Min case: simply compare the current value to the value 30 minutes ahead ---
        # # # Assuming each row is 1 minute apart, the value 30 minutes from now is given by shift(-30).
        # # # Calculate the time difference in minutes using the epoch_time column.
        # # df['shift_close'] = df['close'].shift(-30)
        # # df['future_ratio'] = df['shift_close'] / df['close']
        # # df['minutes_to_max_close'] = (df['epoch_time'].shift(-30) - df['epoch_time'])

        # df['min_future_close'] = df['close'].iloc[::-1].rolling(window=40, min_periods=1).min().iloc[::-1]
        # df['future_ratio'] = df['min_future_close'] / df['close']

        # # df['shift_close'] = df['close'].shift(-30)
        # # df['future_ratio'] = df['shift_close'] / df['close']
        # df['minutes_to_max_close'] = (df['epoch_time'].shift(-30) - df['epoch_time'])



    return df
    # # --- 1. Compute max high over the next two points ---
    # # Using shift(-1) and shift(-2) to get the next two 'high' values.
    # if RATIO_THRESHOLD > 1:
    #     df['max_high_2'] = pd.concat([df['high'].shift(-1), df['high'].shift(-2)], axis=1).max(axis=1)
    # else:
    #     df['min_low_2'] = pd.concat([df['low'].shift(-1), df['low'].shift(-2)], axis=1).min(axis=1)

    # # --- 2. Prepare to compute max close and time-to-max in the next future_window minutes ---
    # # The window now starts at index i+3 (after the two points used above) and spans `future_window` rows.
    # close_array = df['close'].values
    # epoch_array = df['epoch_time'].values  # assume epoch_time is in seconds
    # if RATIO_THRESHOLD > 1:
    #     future_max_close = np.full(n, np.nan, dtype=float)
    # else:
    #     future_min_close = np.full(n, np.nan, dtype=float)
    # minutes_to_max = np.full(n, np.nan, dtype=float)

    # # Loop over each row. For row i, ensure that there are at least future_window rows after i+2.
    # for i in range(n):
    #     start_idx = i + 3
    #     end_idx = i + 3 + future_window  # window spans future_window rows
    #     if end_idx > n:
    #         # Not enough data points in the future; leave NaN.
    #         continue

    #     # Extract the close prices for the future window.
    #     window_close = close_array[start_idx:end_idx]
    #     if RATIO_THRESHOLD > 1:
    #         # Compute the maximum close value in that window.
    #         max_close = window_close.max()
    #         future_max_close[i] = max_close
    #         # Identify the first occurrence (index within the window) of the maximum close.
    #         idx_in_window = int(np.argmax(window_close))
    #     else:
    #         # Compute the minimum close value in that window.
    #         min_close = window_close.min()
    #         future_min_close[i] = min_close
    #         # Identify the first occurrence (index within the window) of the minimum close.
    #         idx_in_window = int(np.argmin(window_close))


    #     target_idx = start_idx + idx_in_window

    #     # Compute the time difference in minutes between the current row and the row where the max close occurs.
    #     dt_minutes = (epoch_array[target_idx] - epoch_array[i]) / 60.0
    #     minutes_to_max[i] = dt_minutes

    # # --- 3. Compute the ratio: max close in the future window over the max high from the next 2 points ---
    # if RATIO_THRESHOLD > 1:
    #     df['future_ratio'] = future_max_close / df['max_high_2']
    #     df.drop(columns=['max_high_2'], inplace=True)
    # else:
    #     # df['future_ratio'] = future_min_close / df['min_low_2']
    #     print(future_min_close)
    #     df['future_ratio'] = future_min_close / df['min_low_2']
    #     df.drop(columns=['min_low_2'], inplace=True)
    #     print(df['future_ratio'])
        
    # df['minutes_to_max_close'] = minutes_to_max

    # return df

def add_slope_difference(df, slope_lookback=10, group_by_date=True):
    """
    For a given DataFrame that already contains a column named 'slope_<slope_lookback>',
    add a new column 'd2_<slope_lookback>' that represents the difference between the current
    slope value and the previous slope value.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the slope column.
        slope_lookback (int): Lookback value used in the slope column name (e.g. 10 for 'slope_10').
        group_by_date (bool): If True, compute differences separately for each day (requires a 'date' column).
    
    Returns:
        pd.DataFrame: The DataFrame with an added 'd2_<slope_lookback>' column.
    """
    slope_col = f'slope_{slope_lookback}'
    diff_col = f'd2_{slope_lookback}'
    
    if group_by_date and 'date' in df.columns:
        df[diff_col] = df.groupby('date')[slope_col].diff()
    else:
        df[diff_col] = df[slope_col].diff()
    
    return df

# def compute_slopes_for_timeframe(timeframe, df):
#     """
#     Compute slope and difference columns for a given timeframe.
#     Returns a DataFrame with only the new columns.
#     """
#     print(f"Adding {timeframe} minute slopes (Takes a while)")
#     # Create a local copy to work on
#     df_local = df.copy()
#     # Compute rolling slopes for the chosen timeframe
#     df_local = df_local.groupby('date', group_keys=False).apply(
#         lambda group: compute_daily_slopes(group, 'close', timeframe)
#     )
#     # Add the difference column (e.g. d2_10 for timeframe=10)
#     df_local = add_slope_difference(df_local, slope_lookback=timeframe)
    
#     # Return only the new columns that were computed for this timeframe.
#     # For example, if timeframe is 10, we expect 'slope_10' and 'd2_10'.
#     new_cols = [f'slope_{timeframe}', f'd2_{timeframe}']
#     print(f"Done with {timeframe} minute slopes")
#     return df_local[new_cols]

def compute_slopes_for_timeframe(timeframe, df, load_saved=True):
    """
    Compute slope and difference columns for a given timeframe.
    Returns a DataFrame with only the standardized new columns.

    Additionally, the function fits a StandardScaler to these computed columns,
    optionally loads a previously saved scaler, and returns the standardized values.
    
    Parameters:
      timeframe (int): The timeframe (in minutes) for computing slopes.
      df (pd.DataFrame): Input DataFrame containing at least a 'date' column and a 'close' column.
      load_saved (bool): If True and a saved scaler exists, load it; otherwise, compute and save a new scaler.
    """
    # Create a local copy to work on
    df_local = df.copy()
    
    # Compute rolling slopes for the chosen timeframe (grouped by 'date')
    df_local = df_local.groupby('date', group_keys=False).apply(
        lambda group: compute_daily_slopes(group, 'close', timeframe)
    )
    
    # Add the difference column (e.g., d2_10 for timeframe=10)
    df_local = add_slope_difference(df_local, slope_lookback=timeframe)
    
    # Select only the new columns that were computed for this timeframe.
    new_cols = [f'slope_{timeframe}', f'd2_{timeframe}']
    new_cols_df = df_local[new_cols].copy()
    
    # Prepare the scaler file path
    scalers_dir = "scalers"
    os.makedirs(scalers_dir, exist_ok=True)
    scaler_filename = os.path.join(scalers_dir, f"scaler_timeframe_{timeframe}.pkl")
    
    # Load a previously saved scaler if requested and available
    if load_saved and os.path.exists(scaler_filename):
        scaler = joblib.load(scaler_filename)
        print(f"Loaded scaler for timeframe {timeframe} from {scaler_filename}")
        new_cols_standardized = scaler.transform(new_cols_df)
    else:
        # Fit a new scaler and save it
        scaler = StandardScaler()
        new_cols_standardized = scaler.fit_transform(new_cols_df)
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler for timeframe {timeframe} saved to {scaler_filename}")
    
    # Create a standardized DataFrame with the same index as the computed columns.
    new_cols_df_standardized = pd.DataFrame(
        new_cols_standardized, 
        columns=new_cols,
        index=new_cols_df.index
    )
    
    return new_cols_df_standardized


def get_dates_with_negative_mean_slope(df, slope_column='slope_10'):
    """
    Returns a list of dates where the mean value of the specified slope column is below zero.
    
    Parameters:
        df (pd.DataFrame): DataFrame that contains at least the columns 'date' and the slope column.
        slope_column (str): The name of the slope column to consider (default 'slope_10').
    
    Returns:
        list: List of dates (as datetime.date objects) where the mean slope is below zero.
    """
    # Group by date and compute the mean slope for each day
    mean_slopes = df.groupby('date')[slope_column].mean()
    # print(f"Mean of day aves: {mean_slopes.mean()}")
    # print(f"Median of day aves: {mean_slopes.median()}")
    # print(f"Std. dev. of day aves: {mean_slopes.std()}")
    # # Mean of day aves: 0.0019648871683763647
    # # Median of day aves: 0.007974153682770237
    # # Std. dev. of day aves: 0.16508090059369013
    
    # Select dates where the mean slope is below zero
    negative_dates = mean_slopes[mean_slopes < 0.00797].index.tolist()
    # negative_dates = mean_slopes[mean_slopes < -0.015].index.tolist()
    
    return negative_dates

def main(model_type="classifier", retrain=False, model_path="new_random_forest_model.pkl", save_trees_path=None):
    """
    Main function to load data, process features, train or load a model (classifier or regressor),
    and evaluate its performance.

    Parameters:
        model_type (str): "classifier" or "regressor".
        retrain (bool): If True, retrain the model; otherwise, load the model if it exists.
        model_path (str): Path to the model file.
        save_trees_path (str or None): If provided, path to save decision trees in a human-readable format.
    """
    # file_path = "dataset_w_features.csv"
    norm_time = "08:30"
    filter_start_time = "08:30"
    filter_end_time = "15:00"
    
    central = pytz.timezone('US/Central')
    now = datetime.now(central)
    # print(f"Staring at {now.strftime('%H:%M:%S')}")
    # df = pd.read_csv("normed_dataset.csv")
    # df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    load_from_csv = True
    if load_from_csv:
        df = pd.read_csv("C:\\Users\\deade\\OneDrive\\Desktop\\data_science\\normed_dataset.csv")
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
    else:
        print("Loading and normalizing data...")
        df_normalized = load_and_normalize_data(norm_time)
        print(f"Dataframe {len(df_normalized)} after load_and_normalize_data")
        # df = df_normalized
        # df['shift_close'] = df['close'].shift(-30)
        # df['future_ratio'] = df['close'] / df['shift_close']
        # print(df.iloc[20002:20062])
        # exit()

        # 0.993953
        # 0.976458

        if 'downward_trend' in df_normalized.columns:
            df_normalized.drop(columns=['downward_trend'], inplace=True)            

        df_with_slopes = df_normalized

        timeframes = [10, 15, 25, 40, 90, 100, 120]
        # Use a ProcessPoolExecutor for parallel processing (or ThreadPoolExecutor if more appropriate).
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Launch each timeframe computation in parallel
            future_to_tf = {executor.submit(compute_slopes_for_timeframe, tf, df_with_slopes): tf for tf in timeframes}
            
            # Collect results as they complete
            results = {}
            for future in concurrent.futures.as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    result_df = future.result()
                    results[tf] = result_df
                except Exception as exc:
                    print(f"Timeframe {tf} generated an exception: {exc}")

        # Merge all the new columns back into the original DataFrame.
        for tf, new_cols_df in results.items():
            df_with_slopes = df_with_slopes.join(new_cols_df)
            
        print(f"Dataframe {len(df_with_slopes)} after adding slopes")

        timeframes = [10, 25, 40, 90, 100, 120]
        for timeframe in timeframes:
            df_with_slopes[f'SMA_{timeframe}'] = df_with_slopes.groupby('date')['close'].transform(lambda x: x.rolling(window=timeframe).mean())

        # print(df_with_slopes.iloc[150])
        df = add_future_slope(df_with_slopes, column='close', window=30, group_by_date=True, scale_constant=9.1e-5)
        # print(df['future_slope'].min())
        # print(df['future_slope'].max())
        print(f"Median value is {df['future_slope'].median()}")
        print(f"Std. dev is {df['future_slope'].std()}")
        
        # plot_day_data("2025-02-14", df)
        # plot_day_data("2025-02-13", df)
        # plot_day_data("2025-02-12", df)
        # plot_day_data("2025-02-11", df)
        print("Adding slope run-length features...")
        df_with_runs = add_slope_run_length(df, slope_column='slope_10')
        print(f"Dataframe {len(df_with_runs)} after add_slope_run_length")

        print("Filtering data by timeframe...")
        df_filtered = filter_timeframe(df_with_runs, filter_start_time, filter_end_time)
        print(f"Dataframe {len(df_filtered)} after filter_timeframe")

        print("Adding SMA run-length features...")
        df = add_sma_run_length(
            df_filtered, sma25_col='SMA_25', sma100_col='SMA_100', new_col='sma_25_below_100_run_length'
        )
        print(f"Dataframe {len(df)} after add_sma_run_length")

        df = add_future_metrics(df)
        df.to_csv("C:\\Users\\deade\\OneDrive\\Desktop\\data_science\\normed_dataset.csv")
        print(f"Dataframe {len(df)} after add_sma_run_length")
    # exit()
    # print(df.iloc[150])
    # plot_day_data("2025-02-14", df)
    # plot_day_data("2025-02-13", df)
    # plot_day_data("2025-02-12", df)
    # plot_day_data("2025-02-11", df)
    
    # exit()
    # if 'positive_slope_run_length' in df.columns:
    #     df.drop(columns=['positive_slope_run_length'], inplace=True)
    # if 'negative_slope_run_length' in df.columns:
    #     df.drop(columns=['negative_slope_run_length'], inplace=True)
    # print(df.describe())
    # for col in df.columns:
    #     print(f"Statistics for column '{col}':")
    #     # If you want to only process numeric columns:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         print(df[col].describe())
    #     else:
    #         print("Non-numeric column, skipping statistics.")
    #     print("=" * 40)

    # exit()

    # print(df['future_ratio'].min())
    # print(df['future_ratio'].max())
    # mean = df['future_ratio'].mean()
    # median = df['future_ratio'].median()
    # mode = df['future_ratio'].median()
    # std = df['future_ratio'].std()
    # print(mean, median)
    # print(mode, std)
    # plt.hist(df['future_ratio'], bins=1200, color='skyblue', alpha=0.75)
    # plt.axvline(mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.4f}')

    # # Add a vertical line for the median
    # plt.axvline(median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median:.4f}')
    # plt.axvline(mode, color='green', linestyle='dashed', linewidth=2, label=f'Mode: {mode:.4f}')

    # plt.xlabel('Data Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of data_column')
    # plt.legend()
    # plt.show()
    # exit()

    negative_dates = get_dates_with_negative_mean_slope(df)
    # print("Dates with mean slope_10 below zero:", negative_dates)
    negative_slope_df = df[df['date'].isin(negative_dates)]
    other_df = df[~df['date'].isin(negative_dates)]

    # print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(negative_slope_df, train_frac=0.9, val_frac=0.10, test_frac=None, random_state=42)
    if test_df is None:
        test_df_len = 0
    else:
        test_df_len = len(test_df)
    # print(f"Train set: {len(train_df)} rows, Validation set: {len(val_df)} rows, Test set: {test_df_len} rows.\n")
    # exit()
    # Load or train the model based on model_type.
    if model_type == "classifier":
        if not retrain and os.path.exists(model_path):
            print(f"Loading classifier from {model_path}...")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            # print("Training new Random Forest classifier...")
            # print(f"Training Data Stats:")
            # print(f"    Mean = {test_df["future_ratio"].mean()}")
            # print(f"    Median = {test_df["future_ratio"].median()}")
            # print(f"    Std. Dev. = {test_df["future_ratio"].std()}")
            model = train_random_forest(train_df, random_state=42)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Classifier saved to {model_path}.")
    else:  # model_type == "regressor"
        if not retrain and os.path.exists(model_path):
            print(f"Loading regressor from {model_path}...")
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            print("Training new Random Forest regressor...")
            model = train_random_forest_regressor(train_df, random_state=42)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Regressor saved to {model_path}.")

    # If requested, export the decision trees in a human-readable format.
    if save_trees_path:
        from sklearn.tree import export_text

        # Reconstruct the feature list used for training.
        feature_list = ['close']
        feature_list += [col for col in train_df.columns if col.startswith('SMA_')]
        feature_list += [col for col in train_df.columns if col.startswith('slope_')]
        feature_list += [col for col in train_df.columns if col.startswith('d2_')]
        if 'downward_trend' in train_df.columns:
            feature_list.append('downward_trend')
        if 'sma_25_below_100_run_length' in train_df.columns:
            feature_list.append('sma_25_below_100_run_length')
        if 'negative_slope_run_length' in train_df.columns:
            feature_list.append('negative_slope_run_length')
        if 'positive_slope_run_length' in train_df.columns:
            feature_list.append('positive_slope_run_length')

        print(f"Exporting decision trees to {save_trees_path} in a human-readable format...")
        with open(save_trees_path, "w") as f:
            for i, tree in enumerate(model.estimators_):
                f.write(f"Decision Tree {i}\n")
                tree_text = export_text(tree, feature_names=feature_list)
                f.write(tree_text)
                f.write("\n" + "=" * 80 + "\n")

    now = datetime.now(central)
    print(f"Done training at {now.strftime('%H:%M:%S')}")
    # Evaluate the model.
    if model_type == "classifier":
        print("Evaluating classifier on Validation Set:")
        
        start_dt = datetime(2020, 1, 1, 8, 30)
        end_dt   = datetime(2020, 1, 1, 9, 0)

        
        # for idx in range(0, 241, 5):
        #     period_start = start_dt + timedelta(minutes=idx)
        #     period_end = end_dt + timedelta(minutes=idx)
        #     print(f"Start: {period_start.time().strftime("%H:%M")}; End {period_end.time().strftime("%H:%M")}")
        #     evaluate_model(model, val_df, period_start.time().strftime("%H:%M"), period_end.time().strftime("%H:%M"))
        #     # input("Press Any Key...")
            
        # evaluate_model(model, val_df, "08:30", "09:00")
        evaluate_model(model, val_df)
        # Example evaluation for a specific day.
        # print(df['date'].unique)
        evaluate_day_predictions(model, df, '2025-01-02')
        evaluate_day_predictions(model, df, '2025-01-07')
        evaluate_day_predictions(model, df, '2025-01-10')
        evaluate_day_predictions(model, df, '2025-02-10')
        evaluate_day_predictions(model, df, '2025-02-11')
        evaluate_day_predictions(model, df, '2025-02-12')
        evaluate_day_predictions(model, df, '2025-02-13')
        evaluate_day_predictions(model, df, '2025-02-14')
        evaluate_day_predictions(model, df, '2025-02-20')
        evaluate_day_predictions(model, df, '2025-02-21')
        evaluate_day_predictions(model, df, '2025-02-25')
        evaluate_day_predictions(model, df, '2025-02-26')
        evaluate_day_predictions(model, df, '2025-02-27')
    else:
        print("Evaluating regressor on Validation Set:")
        evaluate_regression_model(model, val_df)
        # Example evaluation for a specific day.
        evaluate_day_regression(model, df, '2025-01-02')

    # Prepare feature list for displaying feature importances.
    features = ['close']
    features += [col for col in train_df.columns if col.startswith('SMA_')]
    features += [col for col in train_df.columns if col.startswith('slope_')]
    features += [col for col in train_df.columns if col.startswith('d2_')]
    if 'downward_trend' in train_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in train_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in train_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in train_df.columns:
        features.append('positive_slope_run_length')

    # display_feature_importances(model, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest prediction script for SPY data.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model instead of loading from file.")
    parser.add_argument("--model-path", type=str, default="low_close_random_forest_model.pkl", help="Path to save/load the model.")
    parser.add_argument("--save-trees", type=str, default=None, help="Path to save the decision trees in a human-readable format.")
    parser.add_argument("--model-type", type=str, choices=["classifier", "regressor"], default="classifier",
                        help="Type of model to train: 'classifier' (default) or 'regressor'.")
    args = parser.parse_args()

    main(model_type=args.model_type, retrain=True, model_path=args.model_path, save_trees_path=args.save_trees)
