import sqlite3
import pandas as pd

class SPYDataLoader:
    def __init__(self, db_path):
        """
        Initialize the loader with the path to the SQLite database.
        """
        self.db_path = db_path
        self.dataframe = None

    def load_data(self):
        """
        Loads data from all tables matching the pattern 'SPY_prices_%', 
        concatenates them into a single DataFrame, orders by 'epoch_time',
        and resets the DataFrame's index.
        
        Returns:
            pd.DataFrame: The concatenated and sorted DataFrame.
        """
        # Establish a connection to the SQLite database.
        conn = sqlite3.connect(self.db_path)
        try:
            # Retrieve table names matching the pattern 'SPY_prices_%'
            query = """
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' AND name LIKE 'SPY_prices_%';
            """
            table_names = pd.read_sql_query(query, conn)['name'].tolist()

            # Collect data from each table in a list of DataFrames.
            df_list = []
            for table in table_names:
                # Fetch all records from the current table.
                df = pd.read_sql_query(f"SELECT * FROM {table};", conn)
                df_list.append(df)

            # Concatenate all the DataFrames.
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
            else:
                combined_df = pd.DataFrame()

            # Sort the DataFrame by 'epoch_time' and reset the index.
            combined_df.sort_values(by='epoch_time', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)

            # Save the processed DataFrame to the instance.
            self.dataframe = combined_df
            return self.dataframe

        finally:
            # Always close the database connection.
            conn.close()


if __name__ == "__main__":
    loader = SPYDataLoader("stock_project\SPY_data.db")
    df = loader.load_data()
    print(df.head())
