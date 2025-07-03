from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import pytz
import numpy as np
import io

app = Flask(__name__)

# Set file upload size limit (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths using environment variables for Render compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
excel_file_path = os.getenv("RENDER_EXCEL_PATH", os.path.join(DATA_DIR, "DDR_Predictor.xlsx"))
csv_path = os.getenv("RENDER_CSV_PATH", os.path.join(DATA_DIR, "CLhistorical5m.csv"))
wed_odr_path = os.getenv("RENDER_WED_ODR_PATH", os.path.join(DATA_DIR, "Week Wed ODR.csv"))
output_path = os.getenv("RENDER_OUTPUT_PATH", os.path.join(DATA_DIR, "day_model_probabilities.txt"))
model_weekly_data_path = os.path.join(DATA_DIR, "model_weekly_data.csv")
processed_weekly_data_path = os.path.join(DATA_DIR, "processed_weekly_data.csv")
model_weekly_data_with_partial_path = os.path.join(DATA_DIR, "model_weekly_data_with_partial.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Level hierarchy for reference
LEVEL_HIERARCHY = {
    "Min": ["Min"],
    "Min-Med": ["Min-Med", "Min"],
    "Med-Max": ["Med-Max", "Min-Med", "Min"],
    "Max Extreme": ["Max Extreme", "Med-Max", "Min-Med", "Min"],
    "Unknown": ["Unknown"]
}

# Initialize global variables for scenario comparison
df = None
num_columns = 0
total_rows = 0
odr_starts = []
start_colors = []
odr_models = []
odr_true_false = []
locations_low = []
colors = []
locations_high = []
high_level_hits = []
colors_high = []
low_level_hits = []

# Initialize global variables for day model analysis
df_day_model = None
df_wed_odr = None
df_model_weekly = None
df_processed_weekly = None
df_model_weekly_with_partial = None
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
model_options = ['Any', 'Upside', 'Downside', 'Inside', 'Outside']
partial_model_options = ['Any', 'Upside', 'Downside', 'Inside', 'Outside', 'Undefined']
week_role_options = ['Any', 'High of Week (HOW)', 'Low of Week (LOW)']
week_wed_odr_options = ['Any']

# Load and process the Excel file for scenario comparison
def load_excel_file():
    global df, num_columns, total_rows, odr_starts, start_colors, odr_models, odr_true_false, locations_low, colors, locations_high, high_level_hits, colors_high, low_level_hits
    if not os.path.exists(excel_file_path):
        logger.error(f"The file {excel_file_path} does not exist.")
        # Initialize with empty data instead of failing
        df = pd.DataFrame()
        num_columns = 0
        total_rows = 0
        odr_starts = ['Any']
        start_colors = ['Any']
        odr_models = ['Any']
        odr_true_false = ['Any']
        locations_low = ['Any']
        colors = ['Any']
        locations_high = ['Any']
        high_level_hits = ['Any']
        colors_high = ['Any']
        low_level_hits = ['Any']
        return False
    
    try:
        xl = pd.ExcelFile(excel_file_path)
        sheet_names = xl.sheet_names
        df_raw = None
        for sheet in sheet_names:
            temp_df = pd.read_excel(excel_file_path, sheet_name=sheet, header=0)
            num_columns = len(temp_df.columns)
            if num_columns in [9, 11]:
                df_raw = temp_df
                break
        
        if df_raw is None:
            logger.error("No sheet found with 9 or 11 columns.")
            return False
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        return False

    # Define column names
    if num_columns == 11:
        column_names = [
            "Date", "Odr start", "Start color", "Location of Low", "Low Level Hit", "Low color",
            "Location of High", "High Level Hit", "High color", "ODR Model", "ODR True/False"
        ]
    elif num_columns == 9:
        column_names = [
            "Date", "Location of Low", "Low Level Hit", "Low color",
            "Location of High", "High Level Hit", "High color", "ODR Model", "ODR True/False"
        ]
    df_raw.columns = column_names
    df = df_raw.drop(columns=["Date"]).reset_index(drop=True)

    # Handle NaN and convert to strings
    for col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)

    total_rows = len(df)

    # Get unique values for dropdowns
    odr_starts = sorted(df["Odr start"].unique()) if num_columns == 11 else []
    start_colors = sorted(df["Start color"].unique()) if num_columns == 11 else []
    odr_models = sorted(df["ODR Model"].unique())
    odr_true_false = sorted(df["ODR True/False"].unique())
    locations_low = sorted(df["Location of Low"].unique())
    colors = sorted(df["Low color"].unique())
    locations_high = sorted(df["Location of High"].unique())
    high_level_hits = sorted(df["High Level Hit"].unique())
    colors_high = sorted(df["High color"].unique())
    low_level_hits = sorted(df["Low Level Hit"].unique())

    # Ensure 'Any' is included in all dropdowns for scenario comparison
    if 'Any' not in odr_starts:
        odr_starts = ['Any'] + odr_starts
    if 'Any' not in start_colors:
        start_colors = ['Any'] + start_colors
    if 'Any' not in odr_models:
        odr_models = ['Any'] + odr_models
    if 'Any' not in odr_true_false:
        odr_true_false = ['Any'] + odr_true_false
    if 'Any' not in locations_low:
        locations_low = ['Any'] + locations_low
    if 'Any' not in colors:
        colors = ['Any'] + colors
    if 'Any' not in locations_high:
        locations_high = ['Any'] + locations_high
    if 'Any' not in high_level_hits:
        high_level_hits = ['Any'] + high_level_hits
    if 'Any' not in colors_high:
        colors_high = ['Any'] + colors_high
    if 'Any' not in low_level_hits:
        low_level_hits = ['Any'] + low_level_hits

    logger.debug(f"Loaded dropdowns: locations_low={locations_low}, locations_high={locations_high}, total_rows={total_rows}")
    return True

# Load and process the Week Wed ODR CSV file
def load_wed_odr_file():
    global df_wed_odr, week_wed_odr_options
    if not os.path.exists(wed_odr_path):
        logger.error(f"Week Wed ODR file {wed_odr_path} not found")
        # Initialize with empty data instead of failing
        df_wed_odr = None
        week_wed_odr_options = ['Any']
        return False
    try:
        df_wed_odr = pd.read_csv(wed_odr_path)
        df_wed_odr['Week Start'] = pd.to_datetime(df_wed_odr['Week Start']).dt.date  # Parse as date only
        logger.debug(f"Week Wed ODR CSV loaded. Rows: {len(df_wed_odr)}, Columns: {df_wed_odr.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading Week Wed ODR CSV: {e}")
        df_wed_odr = None
        week_wed_odr_options = ['Any']
        return False

    required_columns = ['Week Start', 'Week Wed ODR']
    if not all(col in df_wed_odr.columns for col in required_columns):
        if 'Week Wed ODR ' in df_wed_odr.columns:
            df_wed_odr.rename(columns={'Week Wed ODR ': 'Week Wed ODR'}, inplace=True)
            logger.debug("Renamed 'Week Wed ODR ' to 'Week Wed ODR'")
        else:
            logger.error(f"Week Wed ODR CSV must contain columns: {required_columns}")
            df_wed_odr = None
            week_wed_odr_options = ['Any']
            return False

    df_wed_odr['Week Wed ODR'] = df_wed_odr['Week Wed ODR'].astype(str).str.strip()
    print('Unique Week Wed ODR values:', sorted(set(df_wed_odr['Week Wed ODR'])))
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            output_content = f.read()
        if '2025' not in output_content:
            df_wed_odr = df_wed_odr[pd.to_datetime(df_wed_odr['Week Start']).dt.year < 2025]
            logger.debug("Excluding 2025 data from Week Wed ODR as it's not in day_model_probabilities.txt")
    week_wed_odr_options = ['Any'] + sorted(set(df_wed_odr['Week Wed ODR'].astype(str)))
    logger.debug(f"Week Wed ODR options: {week_wed_odr_options}")
    return True

# Load and process the CSV file for day model analysis
def load_csv_file():
    global df_day_model
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        df_day_model = None
        return False
    try:
        df_day_model = pd.read_csv(csv_path, parse_dates=['time'])
        logger.debug(f"CSV loaded. Rows: {len(df_day_model)}, Columns: {df_day_model.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        df_day_model = None
        return False

    required_columns = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df_day_model.columns for col in required_columns):
        logger.error(f"CSV must contain columns: {required_columns}")
        df_day_model = None
        return False
    return True

# Load the pre-calculated model weekly data
def load_model_weekly_data():
    global df_model_weekly
    if not os.path.exists(model_weekly_data_path):
        logger.error(f"Model weekly data file {model_weekly_data_path} not found")
        df_model_weekly = None
        return False
    try:
        df_model_weekly = pd.read_csv(model_weekly_data_path)
        # Ensure DataFrame type
        if not isinstance(df_model_weekly, pd.DataFrame):
            df_model_weekly = pd.DataFrame(df_model_weekly)
        df_model_weekly['week_start'] = pd.to_datetime(df_model_weekly['week_start'])
        logger.debug(f"Model weekly data loaded. Rows: {len(df_model_weekly)}, Columns: {df_model_weekly.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading model weekly data: {e}")
        df_model_weekly = None
        return False
    return True

# Load the pre-calculated processed weekly data
def load_processed_weekly_data():
    global df_processed_weekly
    if not os.path.exists(processed_weekly_data_path):
        logger.error(f"Processed weekly data file {processed_weekly_data_path} not found")
        df_processed_weekly = None
        return False
    try:
        df_processed_weekly = pd.read_csv(processed_weekly_data_path)
        df_processed_weekly['week_start'] = pd.to_datetime(df_processed_weekly['week_start'])
        logger.debug(f"Processed weekly data loaded. Rows: {len(df_processed_weekly)}, Columns: {df_processed_weekly.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading processed weekly data: {e}")
        df_processed_weekly = None
        return False
    return True

# Load the weekly data with partial models
def load_model_weekly_data_with_partial():
    global df_model_weekly_with_partial
    if not os.path.exists(model_weekly_data_with_partial_path):
        logger.error(f"Model weekly data with partial file {model_weekly_data_with_partial_path} not found")
        df_model_weekly_with_partial = None
        return False
    try:
        df_model_weekly_with_partial = pd.read_csv(model_weekly_data_with_partial_path)
        df_model_weekly_with_partial['week_start'] = pd.to_datetime(df_model_weekly_with_partial['week_start'])
        logger.debug(f"Model weekly data with partial loaded. Rows: {len(df_model_weekly_with_partial)}, Columns: {df_model_weekly_with_partial.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading model weekly data with partial: {e}")
        df_model_weekly_with_partial = None
        return False
    return True

# Helper function for day model identification
def identify_day_type(day1_high, day1_low, day2_high, day2_low):
    """
    Determine the model for a trading day based on price action:
    - Upside: High > Previous High and Low >= Previous Low
    - Downside: Low < Previous Low and High <= Previous High
    - Inside: High < Previous High and Low >= Previous Low
    - Outside: High > Previous High and Low < Previous Low
    - Undefined: Doesn't fit any of the above patterns
    """
    # Upside: High > Previous High and Low >= Previous Low
    if day2_high > day1_high and day2_low >= day1_low:
        return "Upside"
    # Downside: Low < Previous Low and High <= Previous High
    elif day2_low < day1_low and day2_high <= day1_high:
        return "Downside"
    # Inside: High < Previous High and Low >= Previous Low
    elif day2_high < day1_high and day2_low >= day1_low:
        return "Inside"
    # Outside: High > Previous High and Low < Previous Low
    elif day2_high > day1_high and day2_low < day1_low:
        return "Outside"
    return "Undefined"

# Process weekly data for day model analysis
def get_week_data(df):
    try:
        # Convert timezone to NY time if not already
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        elif df['time'].dt.tz != pytz.timezone('America/New_York'):
            df['time'] = df['time'].dt.tz_convert('America/New_York')
        
        df['date'] = df['time'].dt.date
        df['day_of_week'] = df['time'].dt.day_name()
        
        # Create week start (Tuesday) for each date
        def get_week_start(date):
            days_since_tuesday = (date.weekday() - 1) % 7  # Tuesday = 1
            return date - timedelta(days=days_since_tuesday)
        df['week_start'] = df['date'].apply(get_week_start)
        
        weekly_data = []
        for week_start, week_group in df.groupby('week_start'):
            # Group by date to get daily OHLC
            daily_data = week_group.groupby('date').agg({
                'high': 'max',
                'low': 'min',
                'open': 'first',
                'close': 'last',
                'day_of_week': 'first'
            }).reset_index()
            # Only keep days that are Tue, Wed, Thu, Fri, Mon
            day_order = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Monday']
            daily_data = daily_data[daily_data['day_of_week'].isin(day_order)]
            # Sort by day_order, not by date
            daily_data['day_order'] = daily_data['day_of_week'].apply(lambda d: day_order.index(d) if d in day_order else -1)
            daily_data = daily_data.sort_values('day_order').reset_index(drop=True)
            # Assign day_index 1-5 in this order
            daily_data['day_index'] = daily_data['day_order'] + 1
            # Only include weeks with at least 2 trading days
            if len(daily_data) >= 2:
                # Find high and low days (by price)
                high_idx = daily_data['high'].idxmax()
                low_idx = daily_data['low'].idxmin()
                high_day_index = int(daily_data.loc[high_idx, 'day_index'])
                low_day_index = int(daily_data.loc[low_idx, 'day_index'])
                high_day_name = daily_data.loc[high_idx, 'day_of_week']
                low_day_name = daily_data.loc[low_idx, 'day_of_week']
                # Calculate models for each day (comparing to previous day)
                daily_data['model'] = 'Undefined'
                for i in range(1, len(daily_data)):
                    prev_day = daily_data.iloc[i-1]
                    curr_day = daily_data.iloc[i]
                    daily_data.loc[i, 'model'] = identify_day_type(
                        prev_day['high'], prev_day['low'],
                        curr_day['high'], curr_day['low']
                    )
                # Get Week Wed ODR value
                week_wed_odr_value = "Unknown"
                if df_wed_odr is not None:
                    match = df_wed_odr[df_wed_odr['Week Start'] == week_start]
                    if isinstance(match, pd.DataFrame) and not match.empty:
                        week_wed_odr_value = match['Week Wed ODR'].iloc[0]
                weekly_data.append({
                    'week_start': week_start,
                    'days': daily_data,
                    'high_day': high_day_name,
                    'low_day': low_day_name,
                    'high_day_index': high_day_index,
                    'low_day_index': low_day_index,
                    'week_wed_odr': week_wed_odr_value
                })
        logger.debug(f"Total valid weeks: {len(weekly_data)}")
        return weekly_data
    except Exception as e:
        logger.error(f"Error processing weekly data: {e}")
        return []

# Calculate day model probabilities
def compute_day_model_probabilities(conditions):
    """Compute day model probabilities using the new processed data files"""
    global df_model_weekly, df_processed_weekly, df_wed_odr, df_model_weekly_with_partial
    import pandas as pd
    
    logger.debug(f"Received conditions: {conditions}")
    if df_model_weekly is None:
        return f"Error: Model weekly data file not loaded."
    if df_processed_weekly is None:
        return f"Error: Processed weekly data file not loaded."
    
    # Use the data with partial models for filtering
    if df_model_weekly_with_partial is not None:
        filtered_data = df_model_weekly_with_partial.copy()
        logger.debug(f"Using data with partial models. Initial rows: {len(filtered_data)}")
    else:
        # Fallback to processed weekly data if partial data not available
        filtered_data = df_processed_weekly.copy()
        logger.debug(f"Using processed weekly data (no partial models). Initial rows: {len(filtered_data)}")
    
    if not isinstance(filtered_data, pd.DataFrame):
        filtered_data = pd.DataFrame(filtered_data)
    
    # Merge with Week Wed ODR data if available
    if df_wed_odr is not None:
        filtered_data['week_start_dt'] = pd.to_datetime(filtered_data['week_start'])
        df_wed_odr['Week Start'] = pd.to_datetime(df_wed_odr['Week Start'])
        right_df = df_wed_odr[['Week Start', 'Week Wed ODR']]
        if not isinstance(right_df, pd.DataFrame):
            right_df = pd.DataFrame(right_df)
        filtered_data = pd.merge(
            filtered_data,
            right_df,
            left_on='week_start_dt',
            right_on='Week Start',
            how='left'
        )
        filtered_data = filtered_data.drop('week_start_dt', axis=1)
    for day, cond in conditions.items():
        day_num = int(day.split()[1])
        logger.debug(f"Filtering for {day}: {cond}")
        if cond['role'] == 'High of Week (HOW)':
            if not isinstance(filtered_data, pd.DataFrame):
                filtered_data = pd.DataFrame(filtered_data)
            filtered_data = filtered_data[filtered_data['high_day'] == day_num].copy()
            logger.debug(f"After HOW filter for {day}: {len(filtered_data)} rows")
        elif cond['role'] == 'Low of Week (LOW)':
            if not isinstance(filtered_data, pd.DataFrame):
                filtered_data = pd.DataFrame(filtered_data)
            filtered_data = filtered_data[filtered_data['low_day'] == day_num].copy()
            logger.debug(f"After LOW filter for {day}: {len(filtered_data)} rows")
        if 'model' in cond and cond['model'] != 'Any':
            model_col = f'day_{day_num}_model'
            if isinstance(filtered_data, pd.DataFrame) and model_col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[model_col] == cond['model']].copy()
                logger.debug(f"After model filter for {day} ({cond['model']}): {len(filtered_data)} rows")
        if day == 'Day 1' and 'week_wed_odr' in cond and cond['week_wed_odr'] != 'Any':
            if isinstance(filtered_data, pd.DataFrame) and 'Week Wed ODR' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['Week Wed ODR'] == cond['week_wed_odr']].copy()
                logger.debug(f"After Week Wed ODR filter for {day}: {len(filtered_data)} rows")
        if 'partial_model' in cond and cond['partial_model'] != 'Any':
            partial_model_col = f'day_{day_num}_partial_model'
            if isinstance(filtered_data, pd.DataFrame) and partial_model_col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[partial_model_col] == cond['partial_model']].copy()
                logger.debug(f"After partial model filter for {day} ({cond['partial_model']}): {len(filtered_data)} rows")
    if not isinstance(filtered_data, pd.DataFrame):
        filtered_data = pd.DataFrame(filtered_data)
    logger.debug(f"Final filtered rows: {len(filtered_data)}")
    if len(filtered_data) == 0:
        return f"No historical weeks match the selected conditions: {conditions}"
    total_weeks = len(filtered_data)
    high_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    low_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for _, row in filtered_data.iterrows():
        high_day = int(row['high_day'])
        low_day = int(row['low_day'])
        high_counts[high_day] += 1
        low_counts[low_day] += 1
    high_probabilities = {day: (count / total_weeks) * 100 for day, count in high_counts.items()}
    low_probabilities = {day: (count / total_weeks) * 100 for day, count in low_counts.items()}
    week_starts = list(filtered_data['week_start'])
    # Always convert to DataFrame to guarantee .isin() is available
    df_model_weekly = pd.DataFrame(df_model_weekly)
    model_data_filtered = df_model_weekly[df_model_weekly['week_start'].isin(week_starts)]
    model_probs_by_day = {}
    for day_num in range(1, 6):
        model_col = f'day_{day_num}_model'
        if isinstance(model_data_filtered, pd.DataFrame) and model_col in model_data_filtered.columns:
            model_counts = model_data_filtered[model_col].value_counts()
            total_models = len(model_data_filtered)
            model_probs_by_day[f'Day {day_num}'] = {}
            for model in ['Upside', 'Downside', 'Inside', 'Outside', 'Undefined']:
                count = model_counts.get(model, 0)
                percentage = (count / total_models) * 100 if total_models > 0 else 0
                model_probs_by_day[f'Day {day_num}'][model] = percentage
        else:
            model_probs_by_day[f'Day {day_num}'] = {
                'Upside': 0.0, 'Downside': 0.0, 'Inside': 0.0, 'Outside': 0.0, 'Undefined': 0.0
            }
    
    # Calculate partial model probabilities
    partial_model_probs_by_day = {}
    if df_model_weekly_with_partial is not None:
        partial_data_filtered = df_model_weekly_with_partial[df_model_weekly_with_partial['week_start'].isin(week_starts)]
        for day_num in range(1, 6):
            partial_model_col = f'day_{day_num}_partial_model'
            if isinstance(partial_data_filtered, pd.DataFrame) and partial_model_col in partial_data_filtered.columns:
                partial_model_counts = partial_data_filtered[partial_model_col].value_counts()
                total_partial_models = len(partial_data_filtered)
                partial_model_probs_by_day[f'Day {day_num}'] = {}
                for model in ['Upside', 'Downside', 'Inside', 'Outside', 'Undefined']:
                    count = partial_model_counts.get(model, 0)
                    percentage = (count / total_partial_models) * 100 if total_partial_models > 0 else 0
                    partial_model_probs_by_day[f'Day {day_num}'][model] = percentage
            else:
                partial_model_probs_by_day[f'Day {day_num}'] = {
                    'Upside': 0.0, 'Downside': 0.0, 'Inside': 0.0, 'Outside': 0.0, 'Undefined': 0.0
                }
    else:
        # If no partial data available, set all to 0
        for day_num in range(1, 6):
            partial_model_probs_by_day[f'Day {day_num}'] = {
                'Upside': 0.0, 'Downside': 0.0, 'Inside': 0.0, 'Outside': 0.0, 'Undefined': 0.0
            }
    
    result = f"📊 **{total_weeks} matching datasets** found based on your criteria\n\n"
    result += "High of Week Probabilities:\n"
    for day in range(1, 6):
        result += f"Day {day}: {high_probabilities[day]:.1f}%\n"
    result += "\nLow of Week Probabilities:\n"
    for day in range(1, 6):
        result += f"Day {day}: {low_probabilities[day]:.1f}%\n"
    result += "\nModel Probabilities:\n"
    for day, models in model_probs_by_day.items():
        result += f"\n{day}:\n"
        for model, prob in models.items():
            result += f"  {model}: {prob:.1f}%\n"
        # Add partial model statistics right after regular model statistics for each day
        if day in partial_model_probs_by_day:
            result += f"  Partial Model (4:00-9:25):\n"
            for model, prob in partial_model_probs_by_day[day].items():
                result += f"    {model}: {prob:.1f}%\n"
    return result

# Startup function to load all data files
def load_all_data_files():
    """Load all data files at startup with error handling"""
    logger.debug("Loading data files at startup...")
    
    # Load Excel file for scenario comparison
    excel_loaded = load_excel_file()
    if excel_loaded:
        logger.debug("Excel file loaded successfully")
    else:
        logger.warning("Excel file could not be loaded - scenario comparison features will be limited")
    
    # Load CSV file for day model analysis
    csv_loaded = load_csv_file()
    if csv_loaded:
        logger.debug("CSV file loaded successfully")
    else:
        logger.warning("CSV file could not be loaded - day model analysis features will be limited")
    
    # Load Week Wed ODR file
    wed_odr_loaded = load_wed_odr_file()
    if wed_odr_loaded:
        logger.debug("Week Wed ODR CSV file loaded successfully")
    else:
        logger.warning("Week Wed ODR file could not be loaded - Week Wed ODR filtering will be limited")
    
    # Load model weekly data
    model_weekly_loaded = load_model_weekly_data()
    if model_weekly_loaded:
        logger.debug("Model weekly data file loaded successfully")
    else:
        logger.warning("Model weekly data file could not be loaded - day model analysis will be limited")
    
    # Load processed weekly data
    processed_weekly_loaded = load_processed_weekly_data()
    if processed_weekly_loaded:
        logger.debug("Processed weekly data file loaded successfully")
    else:
        logger.warning("Processed weekly data file could not be loaded - day model analysis will be limited")
    
    # Load model weekly data with partial models
    model_weekly_with_partial_loaded = load_model_weekly_data_with_partial()
    if model_weekly_with_partial_loaded:
        logger.debug("Model weekly data with partial file loaded successfully")
    else:
        logger.warning("Model weekly data with partial file could not be loaded - partial model analysis will be limited")
    
    logger.debug("Startup data loading complete")
    return excel_loaded, csv_loaded, wed_odr_loaded, model_weekly_loaded, processed_weekly_loaded, model_weekly_with_partial_loaded

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, num_columns, total_rows
    result = ""
    error = ""
    selected_odr_start = 'Any'
    selected_start_color = 'Any'
    selected_odr_model = 'Any'
    selected_odr_true_false = 'Any'
    selected_location_low1 = 'Any'
    selected_low_level_hit1 = 'Any'
    selected_color1 = 'Any'
    selected_location_high1 = 'Any'
    selected_high_level_hit1 = 'Any'
    selected_color_high1 = 'Any'
    selected_location_low2 = 'Any'
    selected_low_level_hit2 = 'Any'
    selected_color2 = 'Any'
    selected_location_high2 = 'Any'
    selected_high_level_hit2 = 'Any'
    selected_color_high2 = 'Any'
    
    if df is None:
        if not load_excel_file():
            error = f"Failed to load Excel file: {excel_file_path}. Please ensure the data file is available."
            # Return a basic template with error message
            return render_template(
                'index.html',
                error=error,
                odr_starts=['Any'],
                start_colors=['Any'],
                odr_models=['Any'],
                odr_true_false=['Any'],
                locations_low=['Any'],
                colors=['Any'],
                locations_high=['Any'],
                high_level_hits=['Any'],
                colors_high=['Any'],
                low_level_hits=['Any'],
                selected_odr_start=selected_odr_start,
                selected_start_color=selected_start_color,
                selected_odr_model=selected_odr_model,
                selected_odr_true_false=selected_odr_true_false,
                selected_location_low1=selected_location_low1,
                selected_low_level_hit1=selected_low_level_hit1,
                selected_color1=selected_color1,
                selected_location_high1=selected_location_high1,
                selected_high_level_hit1=selected_high_level_hit1,
                selected_color_high1=selected_color_high1,
                selected_location_low2=selected_location_low2,
                selected_low_level_hit2=selected_low_level_hit2,
                selected_color2=selected_color2,
                selected_location_high2=selected_location_high2,
                selected_high_level_hit2=selected_high_level_hit2,
                selected_color_high2=selected_color_high2,
                days=days,
                model_options=model_options,
                week_role_options=week_role_options,
                week_wed_odr_options=week_wed_odr_options,
                selected_day_models={day: 'Any' for day in days},
                selected_day_roles={day: 'Any' for day in days},
                selected_day_week_wed_odrs={day: 'Any' for day in days},
                selected_day_partial_models={day: 'Any' for day in days},
                day_model_result=""
            )
    
    # After loading df and total_rows, ensure total_rows is always an int and not None
    if total_rows is None:
        total_rows = 0
    else:
        try:
            total_rows = int(total_rows)
        except Exception:
            total_rows = 0

    if request.method == 'POST' and 'odr_start1' in request.form:
        try:
            # Shared ODR start and start color
            shared_odr_start = request.form.get('odr_start1', 'Unknown')
            shared_start_color = request.form.get('start_color1', 'Unknown')
            shared_odr_model = request.form.get('odr_model1', 'Unknown')
            shared_odr_true_false = request.form.get('odr_true_false1', 'Unknown')

            scenario1_conditions = {
                'Odr start': shared_odr_start,
                'Start color': shared_start_color,
                'ODR Model': shared_odr_model,
                'ODR True/False': shared_odr_true_false,
                'Location of Low': request.form.get('location_low1', 'Unknown'),
                'Low Level Hit': request.form.get('low_level_hit1', 'Unknown'),
                'Low color': request.form.get('color1', 'Unknown'),
                'Location of High': request.form.get('location_high1', 'Unknown'),
                'High Level Hit': request.form.get('high_level_hit1', 'Unknown'),
                'High color': request.form.get('color_high1', 'Unknown')
            }
            scenario2_conditions = {
                'Odr start': shared_odr_start,
                'Start color': shared_start_color,
                'ODR Model': shared_odr_model,
                'ODR True/False': shared_odr_true_false,
                'Location of Low': request.form.get('location_low2', 'Unknown'),
                'Low Level Hit': request.form.get('low_level_hit2', 'Unknown'),
                'Low color': request.form.get('color2', 'Unknown'),
                'Location of High': request.form.get('location_high2', 'Unknown'),
                'High Level Hit': request.form.get('high_level_hit2', 'Unknown'),
                'High color': request.form.get('color_high2', 'Unknown')
            }

            selected_odr_start = scenario1_conditions['Odr start']
            selected_start_color = scenario1_conditions['Start color']
            selected_odr_model = scenario1_conditions['ODR Model']
            selected_odr_true_false = scenario1_conditions['ODR True/False']
            selected_location_low1 = scenario1_conditions['Location of Low']
            selected_low_level_hit1 = scenario1_conditions['Low Level Hit']
            selected_color1 = scenario1_conditions['Low color']
            selected_location_high1 = scenario1_conditions['Location of High']
            selected_high_level_hit1 = scenario1_conditions['High Level Hit']
            selected_color_high1 = scenario1_conditions['High color']
            selected_location_low2 = scenario2_conditions['Location of Low']
            selected_low_level_hit2 = scenario2_conditions['Low Level Hit']
            selected_color2 = scenario2_conditions['Low color']
            selected_location_high2 = scenario2_conditions['Location of High']
            selected_high_level_hit2 = scenario2_conditions['High Level Hit']
            selected_color_high2 = scenario2_conditions['High color']
            
            if df is None:
                result = "Error: Data not loaded."
            else:
                def filter_scenario(df, cond):
                    df_filtered = df.copy()
                    for key, value in cond.items():
                        if value != "Any":
                            if key == "Low Level Hit" or key == "High Level Hit":
                                df_filtered = df_filtered[df_filtered[key].isin(LEVEL_HIERARCHY.get(value, [value]))]
                            else:
                                df_filtered = df_filtered[df_filtered[key] == value]
                    return df_filtered

                # Filter both scenarios with shared ODR start/start color
                df_filtered1 = filter_scenario(df, scenario1_conditions)
                df_filtered2 = filter_scenario(df, scenario2_conditions)

                matching_rows1 = len(df_filtered1)
                matching_rows2 = len(df_filtered2)

                # Ensure all are valid integers
                try:
                    matching_rows1 = int(matching_rows1)
                except Exception:
                    matching_rows1 = 0
                try:
                    matching_rows2 = int(matching_rows2)
                except Exception:
                    matching_rows2 = 0
                try:
                    total_rows_int = int(total_rows)
                except Exception:
                    total_rows_int = 0
                if total_rows_int > 0:
                    scenario1_percentage = matching_rows1 / total_rows_int * 100
                    scenario2_percentage = matching_rows2 / total_rows_int * 100
                else:
                    scenario1_percentage = 0
                    scenario2_percentage = 0
                
                # Determine which scenario is more likely and create trading recommendation
                def get_most_likely_locations(df_filtered):
                    if len(df_filtered) == 0:
                        return "Unknown", "Unknown"
                    
                    # Get the most common high and low locations from the data
                    high_locations = df_filtered['Location of High'].value_counts()
                    low_locations = df_filtered['Location of Low'].value_counts()
                    
                    most_likely_high = high_locations.index[0] if len(high_locations) > 0 else "Unknown"
                    most_likely_low = low_locations.index[0] if len(low_locations) > 0 else "Unknown"
                    
                    return most_likely_high, most_likely_low
                
                if scenario1_percentage > scenario2_percentage:
                    more_likely_scenario = "Scenario 1"
                    more_likely_pct = scenario1_percentage
                    less_likely_pct = scenario2_percentage
                    most_likely_high, most_likely_low = get_most_likely_locations(df_filtered1)
                    trading_recommendation = f"EXPECT: High in {most_likely_high}, Low in {most_likely_low}"
                elif scenario2_percentage > scenario1_percentage:
                    more_likely_scenario = "Scenario 2"
                    more_likely_pct = scenario2_percentage
                    less_likely_pct = scenario1_percentage
                    most_likely_high, most_likely_low = get_most_likely_locations(df_filtered2)
                    trading_recommendation = f"EXPECT: High in {most_likely_high}, Low in {most_likely_low}"
                else:
                    more_likely_scenario = "Equal"
                    more_likely_pct = scenario1_percentage
                    less_likely_pct = scenario2_percentage
                    trading_recommendation = "EQUAL PROBABILITY - No clear direction"
                
                # Calculate direction analysis - only consider fields that have actual values (not 'Any')
                def get_direction_percentage(df_filtered, high_loc, low_loc):
                    if len(df_filtered) == 0:
                        return "0%"
                    
                    # Only filter on fields that have actual values (not 'Any')
                    filtered_df = df_filtered.copy()
                    
                    if high_loc != 'Any':
                        filtered_df = filtered_df[filtered_df['Location of High'] == high_loc]
                    
                    if low_loc != 'Any':
                        filtered_df = filtered_df[filtered_df['Location of Low'] == low_loc]
                    
                    # Calculate percentage of this specific combination within the filtered data
                    if len(filtered_df) == 0:
                        return "0%"
                    
                    # Count how many rows have this specific high-low combination
                    direction_count = len(filtered_df[
                        (filtered_df['Location of High'] == high_loc if high_loc != 'Any' else True) & 
                        (filtered_df['Location of Low'] == low_loc if low_loc != 'Any' else True)
                    ])
                    direction_pct = (direction_count / len(filtered_df)) * 100
                    return f"{direction_pct:.0f}%"
                
                # Get the most likely locations for each scenario from the data
                scenario1_high, scenario1_low = get_most_likely_locations(df_filtered1)
                scenario2_high, scenario2_low = get_most_likely_locations(df_filtered2)
                
                scenario1_direction = f"{get_direction_percentage(df_filtered1, scenario1_high, scenario1_low)} High {scenario1_high}-Low {scenario1_low}"
                scenario2_direction = f"{get_direction_percentage(df_filtered2, scenario2_high, scenario2_low)} High {scenario2_high}-Low {scenario2_low}"
                
                # Calculate level probabilities for each scenario - show only the top 3 most likely levels
                def get_level_probabilities(df_filtered, scenario_name):
                    if len(df_filtered) == 0:
                        return "No data"
                    
                    levels = []
                    # High levels
                    high_levels = df_filtered['High Level Hit'].value_counts(normalize=True) * 100
                    for level, pct in high_levels.head(3).items():
                        if pct > 0:
                            levels.append(f"High {level}: {pct:.1f}%")
                    
                    # Low levels  
                    low_levels = df_filtered['Low Level Hit'].value_counts(normalize=True) * 100
                    for level, pct in low_levels.head(3).items():
                        if pct > 0:
                            levels.append(f"Low {level}: {pct:.1f}%")
                    
                    return levels[:6]  # Limit to 6 most common, return as list
                
                scenario1_levels = get_level_probabilities(df_filtered1, "Scenario 1")
                scenario2_levels = get_level_probabilities(df_filtered2, "Scenario 2")
                
                # Format the result with trading-focused output
                result = f"""TRADING ANALYSIS - {datetime.now().strftime('%B %d, %Y, %I:%M %p %Z')}
====================================================================================================
MORE LIKELY: {more_likely_scenario} ({more_likely_pct:.1f}% vs {less_likely_pct:.1f}%)
TRADING RECOMMENDATION: {trading_recommendation}
====================================================================================================
Scenario 1: {scenario1_percentage:.1f}% chance                                                     Scenario 2: {scenario2_percentage:.1f}% chance
Dataset: Scenario 1 ({matching_rows1}/{total_rows} rows)         Dataset: Scenario 2 ({matching_rows2}/{total_rows} rows)
====================================================================================================
{scenario1_direction}                                                         {scenario2_direction}"""
                
                # Add level probabilities with proper formatting
                max_levels = max(len(scenario1_levels), len(scenario2_levels))
                for i in range(max_levels):
                    level1 = scenario1_levels[i] if i < len(scenario1_levels) else ""
                    level2 = scenario2_levels[i] if i < len(scenario2_levels) else ""
                    result += f"\n{level1:<65} {level2}"
        except Exception as e:
            error = f"Error processing scenarios: {e}"
    
    return render_template(
        'index.html',
        result=result,
        error=error,
        odr_starts=odr_starts,
        start_colors=start_colors,
        odr_models=odr_models,
        odr_true_false=odr_true_false,
        locations_low=locations_low,
        colors=colors,
        locations_high=locations_high,
        high_level_hits=high_level_hits,
        colors_high=colors_high,
        low_level_hits=low_level_hits,
        selected_odr_start=selected_odr_start,
        selected_start_color=selected_start_color,
        selected_odr_model=selected_odr_model,
        selected_odr_true_false=selected_odr_true_false,
        selected_location_low1=selected_location_low1,
        selected_low_level_hit1=selected_low_level_hit1,
        selected_color1=selected_color1,
        selected_location_high1=selected_location_high1,
        selected_high_level_hit1=selected_high_level_hit1,
        selected_color_high1=selected_color_high1,
        selected_location_low2=selected_location_low2,
        selected_low_level_hit2=selected_low_level_hit2,
        selected_color2=selected_color2,
        selected_location_high2=selected_location_high2,
        selected_high_level_hit2=selected_high_level_hit2,
        selected_color_high2=selected_color_high2,
        days=days,
        model_options=model_options,
        partial_model_options=partial_model_options,
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models={day: 'Any' for day in days},
        selected_day_roles={day: 'Any' for day in days},
        selected_day_week_wed_odrs={day: 'Any' for day in days},
        selected_day_partial_models={day: 'Any' for day in days},
        day_model_result=""
    )

@app.route('/upload', methods=['POST'])
def upload_file():
    # Define default values for selected variables
    selected_odr_start = 'Any'
    selected_start_color = 'Any'
    selected_odr_model = 'Any'
    selected_odr_true_false = 'Any'
    selected_location_low1 = 'Any'
    selected_low_level_hit1 = 'Any'
    selected_color1 = 'Any'
    selected_location_high1 = 'Any'
    selected_high_level_hit1 = 'Any'
    selected_color_high1 = 'Any'
    selected_location_low2 = 'Any'
    selected_low_level_hit2 = 'Any'
    selected_color2 = 'Any'
    selected_location_high2 = 'Any'
    selected_high_level_hit2 = 'Any'
    selected_color_high2 = 'Any'
    
    if 'file' not in request.files:
        return render_template(
            'index.html',
            error="No file part in the request",
            odr_starts=odr_starts,
            start_colors=start_colors,
            odr_models=odr_models,
            odr_true_false=odr_true_false,
            locations_low=locations_low,
            colors=colors,
            locations_high=locations_high,
            high_level_hits=high_level_hits,
            colors_high=colors_high,
            low_level_hits=low_level_hits,
            selected_odr_start=selected_odr_start,
            selected_start_color=selected_start_color,
            selected_odr_model=selected_odr_model,
            selected_odr_true_false=selected_odr_true_false,
            selected_location_low1=selected_location_low1,
            selected_low_level_hit1=selected_low_level_hit1,
            selected_color1=selected_color1,
            selected_location_high1=selected_location_high1,
            selected_high_level_hit1=selected_high_level_hit1,
            selected_color_high1=selected_color_high1,
            selected_location_low2=selected_location_low2,
            selected_low_level_hit2=selected_low_level_hit2,
            selected_color2=selected_color2,
            selected_location_high2=selected_location_high2,
            selected_high_level_hit2=selected_high_level_hit2,
            selected_color_high2=selected_color_high2,
            days=days,
            model_options=model_options,
            partial_model_options=partial_model_options,
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
            selected_day_partial_models={day: 'Any' for day in days},
            day_model_result=""
        )
    
    file = request.files['file']
    if not file or not getattr(file, 'filename', None):
        return render_template(
            'index.html',
            error="No file selected",
            odr_starts=odr_starts,
            start_colors=start_colors,
            odr_models=odr_models,
            odr_true_false=odr_true_false,
            locations_low=locations_low,
            colors=colors,
            locations_high=locations_high,
            high_level_hits=high_level_hits,
            colors_high=colors_high,
            low_level_hits=low_level_hits,
            selected_odr_start=selected_odr_start,
            selected_start_color=selected_start_color,
            selected_odr_model=selected_odr_model,
            selected_odr_true_false=selected_odr_true_false,
            selected_location_low1=selected_location_low1,
            selected_low_level_hit1=selected_low_level_hit1,
            selected_color1=selected_color1,
            selected_location_high1=selected_location_high1,
            selected_high_level_hit1=selected_high_level_hit1,
            selected_color_high1=selected_color_high1,
            selected_location_low2=selected_location_low2,
            selected_low_level_hit2=selected_low_level_hit2,
            selected_color2=selected_color2,
            selected_location_high2=selected_location_high2,
            selected_high_level_hit2=selected_high_level_hit2,
            selected_color_high2=selected_color_high2,
            days=days,
            model_options=model_options,
            partial_model_options=partial_model_options,
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
            selected_day_partial_models={day: 'Any' for day in days},
            day_model_result=""
        )
    if isinstance(file.filename, str) and file.filename.endswith('.xlsx'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DATA_DIR, filename)
        try:
            file.save(file_path)
            global excel_file_path
            excel_file_path = file_path
            if load_excel_file():
                return render_template(
                    'index.html',
                    success="File uploaded and processed successfully",
                    odr_starts=odr_starts,
                    start_colors=start_colors,
                    odr_models=odr_models,
                    odr_true_false=odr_true_false,
                    locations_low=locations_low,
                    colors=colors,
                    locations_high=locations_high,
                    high_level_hits=high_level_hits,
                    colors_high=colors_high,
                    low_level_hits=low_level_hits,
                    selected_odr_day="Unknown",
                    selected_odr_start=selected_odr_start,
                    selected_start_color=selected_start_color,
                    selected_odr_model=selected_odr_model,
                    selected_odr_true_false=selected_odr_true_false,
                    selected_location_low1=selected_location_low1,
                    selected_low_level_hit1=selected_low_level_hit1,
                    selected_color1=selected_color1,
                    selected_location_high1=selected_location_high1,
                    selected_high_level_hit1=selected_high_level_hit1,
                    selected_color_high1=selected_color_high1,
                    selected_location_low2=selected_location_low2,
                    selected_low_level_hit2=selected_low_level_hit2,
                    selected_color2=selected_color2,
                    selected_location_high2=selected_location_high2,
                    selected_high_level_hit2=selected_high_level_hit2,
                    selected_color_high2=selected_color_high2,
                    days=days,
                    model_options=model_options,
                    partial_model_options=partial_model_options,
                    week_role_options=week_role_options,
                    week_wed_odr_options=week_wed_odr_options,
                    selected_day_models={day: 'Any' for day in days},
                    selected_day_roles={day: 'Any' for day in days},
                    selected_day_week_wed_odrs={day: 'Any' for day in days},
                    selected_day_partial_models={day: 'Any' for day in days},
                    day_model_result=""
                )
            else:
                return render_template(
                    'index.html',
                    error="Failed to process uploaded file",
                    odr_starts=odr_starts,
                    start_colors=start_colors,
                    odr_models=odr_models,
                    odr_true_false=odr_true_false,
                    locations_low=locations_low,
                    colors=colors,
                    locations_high=locations_high,
                    high_level_hits=high_level_hits,
                    colors_high=colors_high,
                    low_level_hits=low_level_hits,
                    selected_odr_start=selected_odr_start,
                    selected_start_color=selected_start_color,
                    selected_odr_model=selected_odr_model,
                    selected_odr_true_false=selected_odr_true_false,
                    selected_location_low1=selected_location_low1,
                    selected_low_level_hit1=selected_low_level_hit1,
                    selected_color1=selected_color1,
                    selected_location_high1=selected_location_high1,
                    selected_high_level_hit1=selected_high_level_hit1,
                    selected_color_high1=selected_color_high1,
                    selected_location_low2=selected_location_low2,
                    selected_low_level_hit2=selected_low_level_hit2,
                    selected_color2=selected_color2,
                    selected_location_high2=selected_location_high2,
                    selected_high_level_hit2=selected_high_level_hit2,
                    selected_color_high2=selected_color_high2,
                    days=days,
                    model_options=model_options,
                    partial_model_options=partial_model_options,
                    week_role_options=week_role_options,
                    week_wed_odr_options=week_wed_odr_options,
                    selected_day_models={day: 'Any' for day in days},
                    selected_day_roles={day: 'Any' for day in days},
                    selected_day_week_wed_odrs={day: 'Any' for day in days},
                    selected_day_partial_models={day: 'Any' for day in days},
                    day_model_result=""
                )
        except Exception as e:
            return render_template(
                'index.html',
                error=f"Error saving file: {e}",
                odr_starts=odr_starts,
                start_colors=start_colors,
                odr_models=odr_models,
                odr_true_false=odr_true_false,
                locations_low=locations_low,
                colors=colors,
                locations_high=locations_high,
                high_level_hits=high_level_hits,
                colors_high=colors_high,
                low_level_hits=low_level_hits,
                selected_odr_start=selected_odr_start,
                selected_start_color=selected_start_color,
                selected_odr_model=selected_odr_model,
                selected_odr_true_false=selected_odr_true_false,
                selected_location_low1=selected_location_low1,
                selected_low_level_hit1=selected_low_level_hit1,
                selected_color1=selected_color1,
                selected_location_high1=selected_location_high1,
                selected_high_level_hit1=selected_high_level_hit1,
                selected_color_high1=selected_color_high1,
                selected_location_low2=selected_location_low2,
                selected_low_level_hit2=selected_low_level_hit2,
                selected_color2=selected_color2,
                selected_location_high2=selected_location_high2,
                selected_high_level_hit2=selected_high_level_hit2,
                selected_color_high2=selected_color_high2,
                days=days,
                model_options=model_options,
                partial_model_options=partial_model_options,
                week_role_options=week_role_options,
                week_wed_odr_options=week_wed_odr_options,
                selected_day_models={day: 'Any' for day in days},
                selected_day_roles={day: 'Any' for day in days},
                selected_day_week_wed_odrs={day: 'Any' for day in days},
                selected_day_partial_models={day: 'Any' for day in days},
                day_model_result=""
            )
    else:
        return render_template(
            'index.html',
            error="Invalid file format. Please upload an .xlsx file",
            odr_starts=odr_starts,
            start_colors=start_colors,
            odr_models=odr_models,
            odr_true_false=odr_true_false,
            locations_low=locations_low,
            colors=colors,
            locations_high=locations_high,
            high_level_hits=high_level_hits,
            colors_high=colors_high,
            low_level_hits=low_level_hits,
            selected_odr_start=selected_odr_start,
            selected_start_color=selected_start_color,
            selected_odr_model=selected_odr_model,
            selected_odr_true_false=selected_odr_true_false,
            selected_location_low1=selected_location_low1,
            selected_low_level_hit1=selected_low_level_hit1,
            selected_color1=selected_color1,
            selected_location_high1=selected_location_high1,
            selected_high_level_hit1=selected_high_level_hit1,
            selected_color_high1=selected_color_high1,
            selected_location_low2=selected_location_low2,
            selected_low_level_hit2=selected_low_level_hit2,
            selected_color2=selected_color2,
            selected_location_high2=selected_location_high2,
            selected_high_level_hit2=selected_high_level_hit2,
            selected_color_high2=selected_color_high2,
            days=days,
            model_options=model_options,
            partial_model_options=partial_model_options,
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
            selected_day_partial_models={day: 'Any' for day in days},
            day_model_result=""
        )

@app.route('/day_model', methods=['POST'])
def day_model():
    global df_day_model, df_wed_odr
    result = ""
    error = ""
    selected_day_models = {day: 'Any' for day in days}
    selected_day_roles = {day: 'Any' for day in days}
    selected_day_week_wed_odrs = {day: 'Any' for day in days}
    selected_day_partial_models = {day: 'Any' for day in days}

    # Debug: print the raw POST form data
    logger.debug(f"POST form data: {dict(request.form)}")
    
    # Preserve scenario comparison results from session or request
    scenario_result = request.form.get('scenario_result', "")
    
    if df_day_model is None:
        if not load_csv_file():
            error = f"Failed to load CSV file: {csv_path}"
    
    if df_wed_odr is None:
        if not load_wed_odr_file():
            error = f"Failed to load Week Wed ODR CSV file: {wed_odr_path}"
    
    if request.method == 'POST':
        try:
            conditions = {}
            for day in days:
                day_key = day.lower().replace(' ', '_')
                model = request.form.get(f'{day_key}_model', 'Any')
                role = request.form.get(f'{day_key}_role', 'Any')
                week_wed_odr = request.form.get('day_1_week_wed_odr', 'Any') if day == 'Day 1' else 'Any'
                partial_model = request.form.get(f'{day_key}_partial_model', 'Any') if day != 'Day 1' else 'Any'
                selected_day_models[day] = model
                selected_day_roles[day] = role
                selected_day_week_wed_odrs[day] = week_wed_odr
                selected_day_partial_models[day] = partial_model
                # Add to conditions if ANY of model, role, week_wed_odr, or partial_model is not 'Any'
                if model != 'Any' or role != 'Any' or (day == 'Day 1' and week_wed_odr != 'Any') or (day != 'Day 1' and partial_model != 'Any'):
                    conditions[day] = {'model': model, 'role': role}
                    if day == 'Day 1' and week_wed_odr != 'Any':
                        conditions[day]['week_wed_odr'] = week_wed_odr
                    if day != 'Day 1' and partial_model != 'Any':
                        conditions[day]['partial_model'] = partial_model
                    logger.debug(f"Added {day} to conditions: {conditions[day]}")
                else:
                    logger.debug(f"Skipped {day}: model='{model}', role='{role}', week_wed_odr='{week_wed_odr}', partial_model='{partial_model}'")
            
            logger.debug(f"Final conditions: {conditions}")
            result = compute_day_model_probabilities(conditions)
        except Exception as e:
            error = f"Error processing day model analysis: {e}"
    
    return render_template(
        'index.html',
        result=scenario_result,  # Preserve scenario comparison results
        error=error,
        odr_starts=odr_starts,
        start_colors=start_colors,
        odr_models=odr_models,
        odr_true_false=odr_true_false,
        locations_low=locations_low,
        colors=colors,
        locations_high=locations_high,
        high_level_hits=high_level_hits,
        colors_high=colors_high,
        low_level_hits=low_level_hits,
        selected_odr_start="Any",
        selected_start_color="Any",
        selected_odr_model="Any",
        selected_odr_true_false="Any",
        selected_location_low1="Any",
        selected_low_level_hit1="Any",
        selected_color1="Any",
        selected_location_high1="Any",
        selected_high_level_hit1="Any",
        selected_color_high1="Any",
        selected_location_low2="Any",
        selected_low_level_hit2="Any",
        selected_color2="Any",
        selected_location_high2="Any",
        selected_high_level_hit2="Any",
        selected_color_high2="Any",
        days=days,
        model_options=model_options,
        partial_model_options=partial_model_options,
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models=selected_day_models,
        selected_day_roles=selected_day_roles,
        selected_day_week_wed_odrs=selected_day_week_wed_odrs,
        selected_day_partial_models=selected_day_partial_models,
        day_model_result=result
    )

@app.route('/clear', methods=['POST'])
def clear_all():
    """Clear all form selections and results"""
    return render_template(
        'index.html',
        result="",
        error="",
        odr_starts=odr_starts,
        start_colors=start_colors,
        odr_models=odr_models,
        odr_true_false=odr_true_false,
        locations_low=locations_low,
        colors=colors,
        locations_high=locations_high,
        high_level_hits=high_level_hits,
        colors_high=colors_high,
        low_level_hits=low_level_hits,
        selected_odr_start="Any",
        selected_start_color="Any",
        selected_odr_model="Any",
        selected_odr_true_false="Any",
        selected_location_low1="Any",
        selected_low_level_hit1="Any",
        selected_color1="Any",
        selected_location_high1="Any",
        selected_high_level_hit1="Any",
        selected_color_high1="Any",
        selected_location_low2="Any",
        selected_low_level_hit2="Any",
        selected_color2="Any",
        selected_location_high2="Any",
        selected_high_level_hit2="Any",
        selected_color_high2="Any",
        days=days,
        model_options=model_options,
        partial_model_options=partial_model_options,
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models={day: 'Any' for day in days},
        selected_day_roles={day: 'Any' for day in days},
        selected_day_week_wed_odrs={day: 'Any' for day in days},
        selected_day_partial_models={day: 'Any' for day in days},
        day_model_result=""
    )

# Place this before the main execution block
logger.debug("Loading data files at startup...")
if load_csv_file():
    logger.debug("CSV file loaded successfully")
else:
    logger.error("Failed to load CSV file at startup")

if load_wed_odr_file():
    logger.debug("Week Wed ODR CSV file loaded successfully")
else:
    logger.error("Failed to load Week Wed ODR CSV file at startup")

# Load the new model and processed weekly data files
if load_model_weekly_data():
    logger.debug("Model weekly data file loaded successfully")
else:
    logger.error("Failed to load model weekly data file at startup")

if load_processed_weekly_data():
    logger.debug("Processed weekly data file loaded successfully")
else:
    logger.error("Failed to load processed weekly data file at startup")

logger.debug("Startup data loading complete")

# Load all data files at startup with better error handling
load_all_data_files()

# --- Scenario Comparison Logic (from user-provided script) ---
SCENARIO_EXCEL_PATH = os.path.join('data', 'DDR_Predictor.xlsx')

def load_scenario_comparison_df():
    if not os.path.exists(SCENARIO_EXCEL_PATH):
        raise FileNotFoundError(f"The file {SCENARIO_EXCEL_PATH} does not exist.")
    xl = pd.ExcelFile(SCENARIO_EXCEL_PATH)
    sheet_names = xl.sheet_names
    df_raw = None
    for sheet in sheet_names:
        temp_df = pd.read_excel(SCENARIO_EXCEL_PATH, sheet_name=sheet, header=0)
        num_columns = len(temp_df.columns)
        if num_columns == 11:
            df_raw = temp_df
            break
        elif num_columns == 9:
            df_raw = temp_df
    if df_raw is None:
        raise ValueError(f"No sheet found with 9 or 11 columns. Found sheets: {sheet_names}.")
    num_columns = len(df_raw.columns)
    if num_columns == 11:
        column_names = [
            "Date", "Odr start", "Start color",
            "Location of Low", "Low Level Hit", "Low Color",
            "Location of High", "High Level Hit", "High color",
            "ODR Model", "ODR True/False"
        ]
    elif num_columns == 9:
        column_names = [
            "Date",
            "Location of Low", "Low Level Hit", "Low Color",
            "Location of High", "High Level Hit", "High color",
            "ODR Model", "ODR True/False"
        ]
    else:
        raise ValueError(f"Unexpected number of columns: {num_columns}.")
    df_raw.columns = column_names
    df = df_raw.drop(columns=["Date"]).reset_index(drop=True)
    for col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str)
    return df, num_columns

SCENARIO_TARGETS = {
    "Med-Max": ["Med-Max", "Max Extreme"],
    "Max Extreme": ["Max Extreme"],
    "Min": ["Min"]
}

def calculate_scenario_probability(df, num_columns, odr_start, start_color, odr_model, odr_true_false, location_high, high_level_hit, high_color, location_low, low_level_hit, low_color, debug_label=None):
    filtered_df = df.copy()
    debug_msgs = []
    if num_columns == 11:
        if odr_start != "Any":
            filtered_df = filtered_df[filtered_df["Odr start"] == odr_start]
            debug_msgs.append(f"[{debug_label}] Filter Odr start={odr_start}: {len(filtered_df)} rows")
        if start_color != "Any":
            filtered_df = filtered_df[filtered_df["Start color"] == start_color]
            debug_msgs.append(f"[{debug_label}] Filter Start color={start_color}: {len(filtered_df)} rows")
    if odr_model != "Any":
        filtered_df = filtered_df[filtered_df["ODR Model"] == odr_model]
        debug_msgs.append(f"[{debug_label}] Filter ODR Model={odr_model}: {len(filtered_df)} rows")
    if odr_true_false != "Any":
        filtered_df = filtered_df[filtered_df["ODR True/False"] == odr_true_false]
        debug_msgs.append(f"[{debug_label}] Filter ODR True/False={odr_true_false}: {len(filtered_df)} rows")
    if location_high != "Any":
        filtered_df = filtered_df[filtered_df["Location of High"] == location_high]
        debug_msgs.append(f"[{debug_label}] Filter Location of High={location_high}: {len(filtered_df)} rows")
    if high_color != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == high_color]
        debug_msgs.append(f"[{debug_label}] Filter High color={high_color}: {len(filtered_df)} rows")
    if high_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
        debug_msgs.append(f"[{debug_label}] Filter High Level Hit={high_level_hit}: {len(filtered_df)} rows")
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
        debug_msgs.append(f"[{debug_label}] Filter Location of Low={location_low}: {len(filtered_df)} rows")
    if low_color != "Any":
        filtered_df = filtered_df[filtered_df["Low Color"] == low_color]
        debug_msgs.append(f"[{debug_label}] Filter Low Color={low_color}: {len(filtered_df)} rows")
    if low_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
        debug_msgs.append(f"[{debug_label}] Filter Low Level Hit={low_level_hit}: {len(filtered_df)} rows")
    try:
        total_rows = int(len(df)) if df is not None else 0
    except Exception:
        total_rows = 0
    try:
        matching_rows = int(len(filtered_df)) if filtered_df is not None else 0
    except Exception:
        matching_rows = 0
    if not isinstance(total_rows, int):
        total_rows = 0
    if not isinstance(matching_rows, int):
        matching_rows = 0
    if total_rows > 0:
        probability = (matching_rows / total_rows) * 100
    else:
        probability = 0.0
    # Print debug info
    if debug_label:
        print(f"\n--- Debug for {debug_label} ---")
        for msg in debug_msgs:
            print(msg)
        print(f"[{debug_label}] Final filtered rows: {matching_rows} / {total_rows}")
        print(f"[{debug_label}] Unique values in each column:")
        for col in filtered_df.columns:
            print(f"  {col}: {sorted(filtered_df[col].unique())}")
    return matching_rows, probability, filtered_df

def calculate_target_probabilities(df, num_columns, odr_start, start_color, odr_model, odr_true_false, location_high, high_level_hit, high_color, location_low, low_level_hit, low_color):
    filtered_df = df.copy()
    if num_columns == 11:
        if odr_start != "Any":
            filtered_df = filtered_df[filtered_df["Odr start"] == odr_start]
        if start_color != "Any":
            filtered_df = filtered_df[filtered_df["Start color"] == start_color]
    if odr_model != "Any":
        filtered_df = filtered_df[filtered_df["ODR Model"] == odr_model]
    if odr_true_false != "Any":
        filtered_df = filtered_df[filtered_df["ODR True/False"] == odr_true_false]
    if location_high != "Any":
        filtered_df = filtered_df[filtered_df["Location of High"] == location_high]
    if high_color != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == high_color]
    if high_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_color != "Any":
        filtered_df = filtered_df[filtered_df["Low Color"] == low_color]
    if low_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
    matching_rows = len(filtered_df)
    if matching_rows == 0:
        return ["No matching data found for this scenario."]
    output = []
    for target, target_levels in SCENARIO_TARGETS.items():
        high_target_df = filtered_df[filtered_df["High Level Hit"].isin(target_levels)]
        high_target_count = len(high_target_df)
        low_target_df = filtered_df[filtered_df["Low Level Hit"].isin(target_levels)]
        low_target_count = len(low_target_df)
        target_count = high_target_count + low_target_count - len(high_target_df[high_target_df.index.isin(low_target_df.index)])
        target_prob = (target_count / matching_rows) * 100 if matching_rows > 0 else 0.0
        location_probs = []
        if target_count > 0:
            high_combinations = high_target_df.groupby(["Location of High", "Location of Low"]).size().reset_index(name="count")
            high_combinations["probability"] = (high_combinations["count"] / target_count) * 100
            low_combinations = low_target_df.groupby(["Location of High", "Location of Low"]).size().reset_index(name="count")
            low_combinations["probability"] = (low_combinations["count"] / target_count) * 100
            all_combinations = pd.concat([high_combinations, low_combinations])
            all_combinations = all_combinations.groupby(["Location of High", "Location of Low"])["probability"].sum().reset_index()
            all_combinations = all_combinations.sort_values(by="probability", ascending=False)
            for _, row in all_combinations.iterrows():
                loc_high = row["Location of High"]
                loc_low = row["Location of Low"]
                prob = row["probability"]
                if prob > 0:
                    location_probs.append(f"    Details: High {loc_high}-Low {loc_low} ({prob:.1f}%)")
        output.append(f"Target: {target} ({target_prob:.1f}%)")
        if location_probs:
            output.extend(location_probs)
        else:
            output.append("    No specific location combinations found.")
    return output

@app.route('/scenario_comparison', methods=['GET', 'POST'])
def scenario_comparison():
    df, num_columns = load_scenario_comparison_df()
    # Get dropdown options
    if num_columns == 11:
        odr_starts = sorted(df["Odr start"].unique())
        start_colors = sorted(df["Start color"].unique())
    else:
        odr_starts = []
        start_colors = []
    odr_models = sorted(df["ODR Model"].unique())
    odr_true_false = sorted(df["ODR True/False"].unique())
    locations_low = sorted(df["Location of Low"].unique())
    low_level_hits = sorted(df["Low Level Hit"].unique())
    colors = sorted(df["Low Color"].unique())
    locations_high = sorted(df["Location of High"].unique())
    high_level_hits = sorted(df["High Level Hit"].unique())
    colors_high = sorted(df["High color"].unique())
    result = None
    filtered_df1 = None
    filtered_df2 = None
    if request.method == 'POST':
        # Scenario 1
        odr_start1 = request.form.get('odr_start1', 'Any')
        start_color1 = request.form.get('start_color1', 'Any')
        odr_model1 = request.form.get('odr_model1', 'Any')
        odr_true_false1 = request.form.get('odr_true_false1', 'Any')
        location_high1 = request.form.get('location_high1', 'Any')
        high_level_hit1 = request.form.get('high_level_hit1', 'Any')
        high_color1 = request.form.get('high_color1', 'Any')
        location_low1 = request.form.get('location_low1', 'Any')
        low_level_hit1 = request.form.get('low_level_hit1', 'Any')
        low_color1 = request.form.get('low_color1', 'Any')
        # Scenario 2
        odr_start2 = request.form.get('odr_start2', 'Any')
        start_color2 = request.form.get('start_color2', 'Any')
        odr_model2 = request.form.get('odr_model2', 'Any')
        odr_true_false2 = request.form.get('odr_true_false2', 'Any')
        location_high2 = request.form.get('location_high2', 'Any')
        high_level_hit2 = request.form.get('high_level_hit2', 'Any')
        high_color2 = request.form.get('high_color2', 'Any')
        location_low2 = request.form.get('location_low2', 'Any')
        low_level_hit2 = request.form.get('low_level_hit2', 'Any')
        low_color2 = request.form.get('low_color2', 'Any')
        # Probabilities
        matching_rows1, prob1, filtered_df1 = calculate_scenario_probability(
            df, num_columns, odr_start1, start_color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, high_color1, location_low1, low_level_hit1, low_color1, debug_label="Scenario 1")
        matching_rows2, prob2, filtered_df2 = calculate_scenario_probability(
            df, num_columns, odr_start2, start_color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, high_color2, location_low2, low_level_hit2, low_color2, debug_label="Scenario 2")
        total_prob = prob1 + prob2
        if total_prob > 0:
            normalized_prob1 = (prob1 / total_prob) * 100
            normalized_prob2 = (prob2 / total_prob) * 100
        else:
            normalized_prob1 = 0.0
            normalized_prob2 = 0.0
        scenario1_lines = calculate_target_probabilities(
            df, num_columns, odr_start1, start_color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, high_color1, location_low1, low_level_hit1, low_color1)
        scenario2_lines = calculate_target_probabilities(
            df, num_columns, odr_start2, start_color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, high_color2, location_low2, low_level_hit2, low_color2)
        max_lines = max(len(scenario1_lines), len(scenario2_lines))
        scenario1_lines.extend([""] * (max_lines - len(scenario1_lines)))
        scenario2_lines.extend([""] * (max_lines - len(scenario2_lines)))
        # Calculate probabilities (normalized to sum to 100%)
        safe_rows1 = int(matching_rows1) if matching_rows1 is not None else 0
        safe_rows2 = int(matching_rows2) if matching_rows2 is not None else 0
        total_matches = safe_rows1 + safe_rows2
        if total_matches > 0:
            normalized_prob1 = (safe_rows1 / total_matches) * 100
            normalized_prob2 = (safe_rows2 / total_matches) * 100
        elif safe_rows1 > 0 and safe_rows2 == 0:
            normalized_prob1 = 100.0
            normalized_prob2 = 0.0
        elif safe_rows2 > 0 and safe_rows1 == 0:
            normalized_prob1 = 0.0
            normalized_prob2 = 100.0
        else:
            normalized_prob1 = 0.0
            normalized_prob2 = 0.0
        result = {
            'prob_comparison': f"Scenario 1: {normalized_prob1:.1f}% chance     Scenario 2: {normalized_prob2:.1f}% chance",
            'scenario1_lines': scenario1_lines,
            'scenario2_lines': scenario2_lines,
            'matching_rows1': safe_rows1,
            'matching_rows2': safe_rows2,
            'total_rows': int(len(df)) if df is not None else 0
        }
    return render_template(
        'scenario_comparison.html',
        odr_starts=odr_starts,
        start_colors=start_colors,
        odr_models=odr_models,
        odr_true_false=odr_true_false,
        locations_low=locations_low,
        low_level_hits=low_level_hits,
        colors=colors,
        locations_high=locations_high,
        high_level_hits=high_level_hits,
        colors_high=colors_high,
        num_columns=num_columns,
        result=result,
        filtered_df1=filtered_df1,
        filtered_df2=filtered_df2
    )

@app.route('/scenario_comparison_export', methods=['POST'])
def scenario_comparison_export():
    import io
    from flask import send_file
    df, num_columns = load_scenario_comparison_df()
    scenario = int(request.form.get('scenario', '1'))
    if scenario == 1:
        odr_start = request.form.get('odr_start1', 'Any')
        start_color = request.form.get('start_color1', 'Any')
        odr_model = request.form.get('odr_model1', 'Any')
        odr_true_false = request.form.get('odr_true_false1', 'Any')
        location_high = request.form.get('location_high1', 'Any')
        high_level_hit = request.form.get('high_level_hit1', 'Any')
        high_color = request.form.get('high_color1', 'Any')
        location_low = request.form.get('location_low1', 'Any')
        low_level_hit = request.form.get('low_level_hit1', 'Any')
        low_color = request.form.get('low_color1', 'Any')
    else:
        odr_start = request.form.get('odr_start2', 'Any')
        start_color = request.form.get('start_color2', 'Any')
        odr_model = request.form.get('odr_model2', 'Any')
        odr_true_false = request.form.get('odr_true_false2', 'Any')
        location_high = request.form.get('location_high2', 'Any')
        high_level_hit = request.form.get('high_level_hit2', 'Any')
        high_color = request.form.get('high_color2', 'Any')
        location_low = request.form.get('location_low2', 'Any')
        low_level_hit = request.form.get('low_level_hit2', 'Any')
        low_color = request.form.get('low_color2', 'Any')
    _, _, filtered_df = calculate_scenario_probability(
        df, num_columns, odr_start, start_color, odr_model, odr_true_false,
        location_high, high_level_hit, high_color, location_low, low_level_hit, low_color, debug_label=f"Export Scenario {scenario}")
    output = io.StringIO()
    filtered_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name=f'scenario_{scenario}_filtered.csv')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)