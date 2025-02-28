import datetime
import subprocess
import holidays
import pandas as pd

def next_trading_day(input_date):
    if isinstance(input_date, str):
        next_trading_day = datetime.datetime.strptime(input_date, '%Y-%m-%d')
    else:
        next_trading_day = input_date
    
    while True:
        next_trading_day = next_trading_day + datetime.timedelta(days=1)
        if trading_hours_today(next_trading_day) is not None:
            return next_trading_day


def trading_hours_today(input_date):
    """
    Determine if a date is a trading day. If it is a holiday, return None.
    If it is an early closure day, return adjusted trading hours.
    Otherwise, return the regular trading hours of the NYSE in Central Time (CT).

    :param input_date: The date to check (as a datetime.date object).
    :return: None if it's a holiday, adjusted trading hours if it's an early closure day, 
             or regular trading hours in CT if it's a regular trading day.
    """
    if isinstance(input_date, str):
        input_date = datetime.datetime.strptime(input_date, '%Y-%m-%d')

    nyse_holidays = holidays.NYSE(years=input_date.year)

    # Check if the date is a holiday
    if input_date in nyse_holidays:
        return None

    # Juneteenth not in holidays...
    if input_date.month == 6 and input_date.day == 19 and input_date.year > 2020:
        return None
    
    # Check if the date is a weekend
    if input_date.weekday() in [5, 6]:  # 5: Saturday, 6: Sunday
        return None
    
    # NYSE regular trading hours in Central Time (CT)
    opening_time = "8:30"
    closing_time = "15:00"

    # Check for early closure
    if is_early_closure(input_date):
        closing_time = "12:00"

    return [opening_time, closing_time]

def is_early_closure(date):
    # Day before Independence Day
    if date.month == 7 and date.day == 3 and date.weekday() < 5:
        return True

    # Black Friday (the Friday after the fourth Thursday of November)
    if date.month == 11:
        fourth_thursday = 22 + (3 - datetime.date(date.year, 11, 1).weekday()) % 7
        if date.day == fourth_thursday + 1:
            return True

    # Christmas Eve
    if date.month == 12 and date.day == 24 and date.weekday() < 5:
        return True

    return False

def epoch_to_datetime(epoch_time):
    """
    Converts an epoch time to a list containing a date and a time.

    :param epoch_time: The epoch time in seconds.
    :return: A list containing the date and time.
    """
    dt = datetime.datetime.fromtimestamp(epoch_time)
    date_str = dt.strftime('%Y-%m-%d')
    time_str = dt.strftime('%H:%M:%S')
    return [date_str, time_str]

# Function to parse time in HH:MM or HH:MM:SS format
def parse_time(time_str):
    try:
        return datetime.datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return datetime.datetime.strptime(time_str, "%H:%M").time()

def parse_date(date_str):
    date_formats = [
        "%Y-%m-%d",  # Formats like 2024-02-01
        "%d-%m-%Y",  # Formats like 01-02-2024
        "%m/%d/%Y",  # Formats like 02/01/2024
        "%Y/%m/%d",  # Formats like 2024/02/01
        "%d/%m/%Y",  # Formats like 01/02/2024
        "%Y.%m.%d",  # Formats like 2024.02.01
        "%d.%m.%Y",  # Formats like 01.02.2024
        "%m-%d-%Y",  # Formats like 02-01-2024
        "%m.%d.%Y",  # Formats like 02.01.2024
        "%y%m%d"     # Formats like 240201 for February 1, 2024
    ]

    for fmt in date_formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Date string '{date_str}' does not match any known date format.")


def is_time_between(target_str, start_str, end_str):

    target = parse_time(target_str)
    start = parse_time(start_str)
    end = parse_time(end_str)

    return start <= target < end

def get_git_repo_root():
    try:
        # Run the git command
        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
        # Decode the output from byte string to normal string and strip newlines
        return root.decode('utf-8').strip()
    except subprocess.CalledProcessError:
        # Handle the case where the current directory is not a Git repository
        return None

def is_time_in_range(time_list, target_time_str):
    start_time_str, end_time_str = time_list
    start_time = datetime.datetime.strptime(start_time_str, "%H:%M").time()
    end_time = datetime.datetime.strptime(end_time_str, "%H:%M").time()
    target_time = datetime.datetime.strptime(target_time_str, "%H:%M:%S").time()

    return start_time <= target_time <= end_time

def minutes_until_end_of_trading_day(epoch_times):
    def calc_minutes(epoch_time):
        date_str, time_str = epoch_to_datetime(epoch_time)
        trading_hours = trading_hours_today(date_str)

        if trading_hours:
            closing_time = parse_time(trading_hours[1])
            current_time = parse_time(time_str)

            # Calculate the difference in minutes
            delta = datetime.datetime.combine(datetime.date.min, closing_time) - datetime.datetime.combine(datetime.date.min, current_time)
            return delta.total_seconds() / 60
        else:
            # Non-trading day
            return None

    # Check if a single value or a list/Series is passed
    if isinstance(epoch_times, (list, pd.Series)):
        return [calc_minutes(time) for time in epoch_times]
    else:
        # Single value
        return calc_minutes(epoch_times)
