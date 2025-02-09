#!/usr/bin/env python3
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Global threshold constant for classification.
RATIO_THRESHOLD = 1.001


def load_and_normalize_data(file_path, norm_time):
    """
    Loads the CSV file and normalizes the 'close' and SMA columns based on the close value
    at a specified time for each day.

    Parameters:
        file_path (str): Path to the CSV file.
        norm_time (str): Normalization time in 'HH:MM' format.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    norm_time = pd.to_datetime(norm_time, format='%H:%M').time()

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
        return group

    df_normalized = df.groupby('date', group_keys=False).apply(normalize_group)
    return df_normalized


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


def split_dataset(df, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
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
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-5:
        raise ValueError("The train, validation, and test fractions must sum to 1.")

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    train_df = df_shuffled.iloc[:train_end].copy()
    val_df = df_shuffled.iloc[train_end:val_end].copy()
    test_df = df_shuffled.iloc[val_end:].copy()

    return train_df, val_df, test_df


def train_random_forest(train_df, random_state=42):
    """
    Trains a RandomForest classifier on the training DataFrame.

    Features used:
      - 'close'
      - All columns starting with 'SMA_'
      - All columns starting with 'slope_'
      - 'downward_trend'
      - Additional run-length features if available.

    The target is set to 1 if 'future_ratio' > RATIO_THRESHOLD, else 0.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame.
        random_state (int): Random state for reproducibility.

    Returns:
        RandomForestClassifier: Trained classifier.
    """
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True).copy()

    features = ['close']
    features += [col for col in train_df.columns if col.startswith('SMA_')]
    features += [col for col in train_df.columns if col.startswith('slope_')]

    if 'downward_trend' in train_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in train_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in train_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in train_df.columns:
        features.append('positive_slope_run_length')

    train_df['target'] = (train_df['future_ratio'] > RATIO_THRESHOLD).astype(int)

    X_train = train_df[features]
    y_train = train_df['target']

    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    return clf


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


def evaluate_model(clf, df):
    """
    Evaluates the classifier using the provided DataFrame and prints performance metrics.

    Parameters:
        clf: Trained classifier.
        df (pd.DataFrame): DataFrame for evaluation.
    """
    features = ['close']
    features += [col for col in df.columns if col.startswith('SMA_')]
    features += [col for col in df.columns if col.startswith('slope_')]
    if 'downward_trend' in df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in df.columns:
        features.append('positive_slope_run_length')

    df = df.copy()
    df['target'] = (df['future_ratio'] > RATIO_THRESHOLD).astype(int)

    X = df[features]
    y_true = df['target']
    y_pred = clf.predict(X)

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


def display_feature_importances(clf, features):
    """
    Displays and plots the feature importances of the trained classifier.

    Parameters:
        clf: Trained classifier.
        features (list): List of feature names.
    """
    importances = clf.feature_importances_
    df_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
    df_importances.sort_values(by='Importance', ascending=False, inplace=True)

    print("Feature Importances:")
    print(df_importances)

    plt.figure(figsize=(10, 6))
    plt.bar(df_importances['Feature'], df_importances['Importance'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importances from the Random Forest Model")
    plt.tight_layout()
    plt.show()


def evaluate_day_predictions(clf, df, day):
    """
    Evaluates predictions for a given day by plotting the normalized close values,
    and comparing predicted vs. actual classifications.

    Parameters:
        clf: Trained classifier.
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
    actual = (day_df['future_ratio'] > 1).astype(int)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    ax1.plot(day_df.index, day_df['close'], label='Normalized Close', color='blue', marker='o')
    ax1.set_title(f"Close Values on {day}")
    ax1.set_ylabel("Normalized Close")
    ax1.legend()

    # Identify contiguous runs of predictions equal to 1.
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

    ax2.plot(day_df.index, predictions, label='Predicted', color='green', marker='o')
    ax2.plot(day_df.index, actual, label='Actual', color='red', marker='x', linestyle='--')
    ax2.set_title(f"Predicted vs Actual Classification on {day}")
    ax2.set_ylabel("Classification (0 or 1)")
    ax2.set_xlabel("Data Index (Time progression)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main(retrain=False, model_path="random_forest_model.pkl", save_trees_path=None):
    """
    Main function to load data, process features, train or load a model, and evaluate performance.

    Parameters:
        retrain (bool): If True, retrain the model. Otherwise, load the model if it exists.
        model_path (str): Path to the model file.
        save_trees_path (str or None): If provided, the file path to save decision trees in a human-readable format.
    """
    file_path = "dataset_w_features.csv"
    norm_time = "08:30"
    filter_start_time = "09:30"
    filter_end_time = "14:00"

    print("Loading and normalizing data...")
    df_normalized = load_and_normalize_data(file_path, norm_time)
    if 'downward_trend' in df_normalized.columns:
        df_normalized.drop(columns=['downward_trend'], inplace=True)

    print("Adding slope run-length features...")
    df_with_runs = add_slope_run_length(df_normalized, slope_column='slope_10')

    print("Filtering data by timeframe...")
    df_filtered = filter_timeframe(df_with_runs, filter_start_time, filter_end_time)

    print("Adding SMA run-length features...")
    df_with_sma_run = add_sma_run_length(
        df_filtered, sma25_col='SMA_25', sma100_col='SMA_100', new_col='sma_25_below_100_run_length'
    )

    print("Splitting dataset...")
    train_df, val_df, test_df = split_dataset(df_with_sma_run, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42)
    print(f"Train set: {len(train_df)} rows, Validation set: {len(val_df)} rows, Test set: {len(test_df)} rows.\n")

    # Either load the model or train a new one.
    if not retrain and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
    else:
        print("Training new Random Forest classifier...")
        clf = train_random_forest(train_df, random_state=42)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"Model saved to {model_path}.")

    # If requested, export the decision trees in a human-readable format.
    if save_trees_path:
        from sklearn.tree import export_text

        # Reconstruct the feature list used for training.
        feature_list = ['close']
        feature_list += [col for col in train_df.columns if col.startswith('SMA_')]
        feature_list += [col for col in train_df.columns if col.startswith('slope_')]
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
            for i, tree in enumerate(clf.estimators_):
                f.write(f"Decision Tree {i}\n")
                tree_text = export_text(tree, feature_names=feature_list)
                f.write(tree_text)
                f.write("\n" + "=" * 80 + "\n")

    print("Evaluating model on Validation Set:")
    evaluate_model(clf, val_df)

    # Evaluate predictions on a specific day (example day: 2025-02-07)
    evaluate_day_predictions(clf, df_with_sma_run, '2025-02-07')

    # Prepare feature list for displaying feature importances.
    features = ['close']
    features += [col for col in train_df.columns if col.startswith('SMA_')]
    features += [col for col in train_df.columns if col.startswith('slope_')]
    if 'downward_trend' in train_df.columns:
        features.append('downward_trend')
    if 'sma_25_below_100_run_length' in train_df.columns:
        features.append('sma_25_below_100_run_length')
    if 'negative_slope_run_length' in train_df.columns:
        features.append('negative_slope_run_length')
    if 'positive_slope_run_length' in train_df.columns:
        features.append('positive_slope_run_length')

    display_feature_importances(clf, features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest prediction script for SPY data.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model instead of loading from file.")
    parser.add_argument("--model-path", type=str, default="random_forest_model.pkl", help="Path to save/load the model.")
    parser.add_argument("--save-trees", type=str, default="trees.txt", help="Path to save the decision trees in a human-readable format.")
    args = parser.parse_args()

    main(retrain=args.retrain, model_path=args.model_path, save_trees_path=args.save_trees)
