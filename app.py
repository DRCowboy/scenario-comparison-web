from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

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
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
model_options = ['Any', 'Upside', 'Downside', 'Inside', 'Outside']
week_role_options = ['Any', 'High of Week (HOW)', 'Low of Week (LOW)']
week_wed_odr_options = ['Any']

# Load and process the Excel file for scenario comparison
def load_excel_file():
    global df, num_columns, total_rows, odr_starts, start_colors, odr_models, odr_true_false, locations_low, colors, locations_high, high_level_hits, colors_high, low_level_hits
    if not os.path.exists(excel_file_path):
        logger.error(f"The file {excel_file_path} does not exist.")
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
        return False
    try:
        df_wed_odr = pd.read_csv(wed_odr_path)
        df_wed_odr['Week Start'] = pd.to_datetime(df_wed_odr['Week Start']).dt.date  # Parse as date only
        logger.debug(f"Week Wed ODR CSV loaded. Rows: {len(df_wed_odr)}, Columns: {df_wed_odr.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading Week Wed ODR CSV: {e}")
        return False

    required_columns = ['Week Start', 'Week Wed ODR']
    if not all(col in df_wed_odr.columns for col in required_columns):
        if 'Week Wed ODR ' in df_wed_odr.columns:
            df_wed_odr.rename(columns={'Week Wed ODR ': 'Week Wed ODR'}, inplace=True)
            logger.debug("Renamed 'Week Wed ODR ' to 'Week Wed ODR'")
        else:
            logger.error(f"Week Wed ODR CSV must contain columns: {required_columns}")
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
        return False
    try:
        df_day_model = pd.read_csv(csv_path, parse_dates=['time'])
        logger.debug(f"CSV loaded. Rows: {len(df_day_model)}, Columns: {df_day_model.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return False

    required_columns = ['time', 'open', 'high', 'low', 'close']
    if not all(col in df_day_model.columns for col in required_columns):
        logger.error(f"CSV must contain columns: {required_columns}")
        return False
    return True

# Helper function for day model identification
def identify_day_type(day1_high, day1_low, day2_high, day2_low):
    if day2_high > day1_high and day2_low > day1_low:
        return "Upside"
    elif day2_high < day1_high and day2_low < day1_low:
        return "Downside"
    elif day2_high < day1_high and day2_low > day1_low:
        return "Inside"
    elif day2_high > day1_high and day2_low < day1_low:
        return "Outside"
    return "Undefined"

# Process weekly data for day model analysis
def get_week_data(df):
    try:
        df['date'] = df['time'].dt.date
        # Align week_start to Tuesday (dayofweek=1)
        df['week_start'] = df['time'] - pd.to_timedelta((df['time'].dt.dayofweek - 1) % 7, unit='D')
        df['week_start'] = df['week_start'].dt.date
        weekly_data = []
        for week_start, week_group in df.groupby('week_start'):
            week_days = week_group.groupby('date').agg({
                'high': 'max',
                'low': 'min',
                'time': 'first'
            }).reset_index()
            logger.debug(f"Week {week_start}: {len(week_days)} days, Days={week_days['time'].dt.day_name().tolist()}")
            if len(week_days) >= 2:
                week_days['day_of_week'] = pd.to_datetime(week_days['time']).dt.day_name()
                high_day_idx = week_days['high'].idxmax()
                high_day = week_days.loc[high_day_idx, 'day_of_week']
                low_day_idx = week_days['low'].idxmin()
                low_day = week_days.loc[low_day_idx, 'day_of_week']
                logger.debug(f"Week {week_start}: High day={high_day}, Low day={low_day}, Days={week_days['time'].tolist()}")
                week_days['model'] = 'Undefined'
                for i in range(1, len(week_days)):
                    day1 = week_days.iloc[i-1]
                    day2 = week_days.iloc[i]
                    week_days.loc[i, 'model'] = identify_day_type(
                        day1['high'], day1['low'], day2['high'], day2['low']
                    )
                # Attach Week Wed ODR value
                week_wed_odr_value = "Unknown"
                if df_wed_odr is not None:
                    match = df_wed_odr[df_wed_odr['Week Start'] == week_start]
                    if isinstance(match, pd.DataFrame) and not match.empty:
                        week_wed_odr_value = match['Week Wed ODR'].values[0]
                print(f"week_start: {week_start}, Week Wed ODR in df_wed_odr: {df_wed_odr[df_wed_odr['Week Start'] == week_start]['Week Wed ODR'].tolist() if df_wed_odr is not None else 'No data'}")
                weekly_data.append({
                    'week_start': week_start,
                    'days': week_days,
                    'high_day': high_day,
                    'low_day': low_day,
                    'week_wed_odr': week_wed_odr_value
                })
        # Log low day distribution
        low_day_counts = {}
        for week in weekly_data:
            low_day = week['low_day']
            low_day_counts[low_day] = low_day_counts.get(low_day, 0) + 1
        logger.debug(f"Low day distribution: {low_day_counts}")
        logger.debug(f"Total valid weeks: {len(weekly_data)}")
        return weekly_data
    except Exception as e:
        logger.error(f"Error processing weekly data: {e}")
        return []

# Calculate day model probabilities
def compute_day_model_probabilities(conditions):
    day_indices = {'Day 1': 0, 'Day 2': 1, 'Day 3': 2, 'Day 4': 3, 'Day 5': 4}
    if df_day_model is None:
        return f"Error: CSV file {csv_path} not loaded."
    if df_wed_odr is None:
        return f"Error: Week Wed ODR CSV file {wed_odr_path} not loaded."
    
    weekly_data = get_week_data(df_day_model)
    if not weekly_data:
        return "Error: No valid weekly data found."
    
    matching_weeks = []
    for week in weekly_data:
        days_df = week['days']
        match = True
        logger.debug(f"Processing week {week['week_start']}: {len(days_df)} days, Low day={week['low_day']}")
        
        for day, cond in conditions.items():
            day_idx = day_indices[day]
            if day_idx >= len(days_df):
                logger.debug(f"Week {week['week_start']} skipped: not enough days ({len(days_df)} < {day_idx + 1})")
                match = False
                break
            if cond['model'] != 'Any' and days_df.iloc[day_idx]['model'] != cond['model']:
                logger.debug(f"Week {week['week_start']} skipped: {day} model {days_df.iloc[day_idx]['model']} != {cond['model']}")
                match = False
                break
            if cond['role'] == 'High of Week (HOW)' and days_df.iloc[day_idx]['day_of_week'] != week['high_day']:
                logger.debug(f"Week {week['week_start']} skipped: {day} not High of Week, day_of_week={days_df.iloc[day_idx]['day_of_week']}, high_day={week['high_day']}")
                match = False
                break
            if cond['role'] == 'Low of Week (LOW)' and days_df.iloc[day_idx]['day_of_week'] != week['low_day']:
                logger.debug(f"Week {week['week_start']} skipped: {day} not Low of Week, day_of_week={days_df.iloc[day_idx]['day_of_week']}, low_day={week['low_day']}")
                match = False
                break
            if day == 'Day 1' and 'week_wed_odr' in cond and cond['week_wed_odr'] != 'Any':
                week_wed_odr_value = week.get('week_wed_odr', 'Unknown')
                print(f"Filtering: week_wed_odr_value='{week_wed_odr_value}', filter='{cond['week_wed_odr']}'")
                if week_wed_odr_value != cond['week_wed_odr']:
                    logger.debug(f"Week {week['week_start']} skipped: Day 1 Week Wed ODR {week_wed_odr_value} != {cond['week_wed_odr']}")
                    match = False
                    break
        if match:
            matching_weeks.append(week)
            logger.debug(f"Week {week['week_start']} matched conditions")
        else:
            logger.debug(f"Week {week['week_start']} failed conditions: {conditions}")
    
    logger.debug(f"Matching weeks: {len(matching_weeks)}, Conditions: {conditions}")
    if not matching_weeks:
        return f"No historical weeks match the selected conditions: {conditions}"
    
    # After filtering, always show model probabilities for Day 2-5
    model_probs_by_day = {}
    for day_num in range(1, 5):  # Day 2 to Day 5
        model_counts = {'Upside': 0, 'Downside': 0, 'Inside': 0, 'Outside': 0}
        total_valid = 0
        for week in matching_weeks:
            days_df = week['days']
            if day_num < len(days_df):
                model = days_df.iloc[day_num]['model']
                if model in model_counts:
                    model_counts[model] += 1
                    total_valid += 1
        model_probs = {model: (count / total_valid if total_valid > 0 else 0) for model, count in model_counts.items()}
        model_probs_by_day[f"Day {day_num+1}"] = model_probs

    result = f"Conditions (as of {datetime.now().strftime('%B %d, %Y, %I:%M %p %Z')}):\n"
    for day, cond in conditions.items():
        week_wed_odr_str = f", Week Wed ODR={cond['week_wed_odr']}" if day == 'Day 1' and 'week_wed_odr' in cond else ""
        result += f"{day}: Model={cond['model']}, Role={cond['role']}{week_wed_odr_str}\n"
    result += f"\nNumber of historical weeks matching conditions: {len(matching_weeks)}\n"
    result += f"\nProbabilities for Day 2-5:\n"
    for day in ["Day 2", "Day 3", "Day 4", "Day 5"]:
        result += f"{day}:\n"
        for model, prob in model_probs_by_day[day].items():
            result += f"  {model}: {prob:.2%}\n"
    
    try:
        with open(output_path, 'a') as f:
            f.write(result + "\n\n")
    except Exception as e:
        logger.error(f"Error writing to {output_path}: {e}")
    
    return result

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
            error = f"Failed to load Excel file: {excel_file_path}"
    
    if request.method == 'POST' and 'odr_start1' in request.form:
        try:
            scenario1_conditions = {
                'Odr start': request.form.get('odr_start1', 'Unknown'),
                'Start color': request.form.get('start_color1', 'Unknown'),
                'ODR Model': request.form.get('odr_model1', 'Unknown'),
                'ODR True/False': request.form.get('odr_true_false1', 'Unknown'),
                'Location of Low': request.form.get('location_low1', 'Unknown'),
                'Low Level Hit': request.form.get('low_level_hit1', 'Unknown'),
                'Low color': request.form.get('color1', 'Unknown'),
                'Location of High': request.form.get('location_high1', 'Unknown'),
                'High Level Hit': request.form.get('high_level_hit1', 'Unknown'),
                'High color': request.form.get('color_high1', 'Unknown')
            }
            scenario2_conditions = {
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
                df_filtered1 = df.copy()
                df_filtered2 = df.copy()
                for key, value in scenario1_conditions.items():
                    if value != "Any":
                        if key == "Low Level Hit" or key == "High Level Hit":
                            df_filtered1 = df_filtered1[df_filtered1[key].isin(LEVEL_HIERARCHY.get(value, [value]))]
                        else:
                            df_filtered1 = df_filtered1[df_filtered1[key] == value]
                for key, value in scenario2_conditions.items():
                    if value != "Any":
                        if key == "Low Level Hit" or key == "High Level Hit":
                            df_filtered2 = df_filtered2[df_filtered2[key].isin(LEVEL_HIERARCHY.get(value, [value]))]
                        else:
                            df_filtered2 = df_filtered2[df_filtered2[key] == value]
                
                matching_rows1 = len(df_filtered1)
                matching_rows2 = len(df_filtered2)
                
                # Analyze which scenario is more likely
                scenario1_percentage = matching_rows1 / total_rows * 100
                scenario2_percentage = matching_rows2 / total_rows * 100
                
                # Calculate total percentage and normalize to 100%
                total_pct = scenario1_percentage + scenario2_percentage
                if total_pct > 0:
                    scenario1_pct_normalized = (scenario1_percentage / total_pct) * 100
                    scenario2_pct_normalized = (scenario2_percentage / total_pct) * 100
                else:
                    scenario1_pct_normalized = 0
                    scenario2_pct_normalized = 0
                
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
                
                if scenario1_pct_normalized > scenario2_pct_normalized:
                    more_likely_scenario = "Scenario 1"
                    more_likely_pct = scenario1_pct_normalized
                    less_likely_pct = scenario2_pct_normalized
                    most_likely_high, most_likely_low = get_most_likely_locations(df_filtered1)
                    trading_recommendation = f"EXPECT: High in {most_likely_high}, Low in {most_likely_low}"
                elif scenario2_pct_normalized > scenario1_pct_normalized:
                    more_likely_scenario = "Scenario 2"
                    more_likely_pct = scenario2_pct_normalized
                    less_likely_pct = scenario1_pct_normalized
                    most_likely_high, most_likely_low = get_most_likely_locations(df_filtered2)
                    trading_recommendation = f"EXPECT: High in {most_likely_high}, Low in {most_likely_low}"
                else:
                    more_likely_scenario = "Equal"
                    more_likely_pct = scenario1_pct_normalized
                    less_likely_pct = scenario2_pct_normalized
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
Scenario 1: {scenario1_pct_normalized:.1f}% chance                                                     Scenario 2: {scenario2_pct_normalized:.1f}% chance
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
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models={day: 'Any' for day in days},
        selected_day_roles={day: 'Any' for day in days},
        selected_day_week_wed_odrs={day: 'Any' for day in days},
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
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
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
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
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
                    week_role_options=week_role_options,
                    week_wed_odr_options=week_wed_odr_options,
                    selected_day_models={day: 'Any' for day in days},
                    selected_day_roles={day: 'Any' for day in days},
                    selected_day_week_wed_odrs={day: 'Any' for day in days},
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
                    week_role_options=week_role_options,
                    week_wed_odr_options=week_wed_odr_options,
                    selected_day_models={day: 'Any' for day in days},
                    selected_day_roles={day: 'Any' for day in days},
                    selected_day_week_wed_odrs={day: 'Any' for day in days},
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
                week_role_options=week_role_options,
                week_wed_odr_options=week_wed_odr_options,
                selected_day_models={day: 'Any' for day in days},
                selected_day_roles={day: 'Any' for day in days},
                selected_day_week_wed_odrs={day: 'Any' for day in days},
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
            week_role_options=week_role_options,
            week_wed_odr_options=week_wed_odr_options,
            selected_day_models={day: 'Any' for day in days},
            selected_day_roles={day: 'Any' for day in days},
            selected_day_week_wed_odrs={day: 'Any' for day in days},
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
                selected_day_models[day] = model
                selected_day_roles[day] = role
                selected_day_week_wed_odrs[day] = week_wed_odr
                if model != 'Any' or role != 'Any' or (day == 'Day 1' and week_wed_odr != 'Any'):
                    conditions[day] = {'model': model, 'role': role}
                    if day == 'Day 1' and week_wed_odr != 'Any':
                        conditions[day]['week_wed_odr'] = week_wed_odr
            
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
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models=selected_day_models,
        selected_day_roles=selected_day_roles,
        selected_day_week_wed_odrs=selected_day_week_wed_odrs,
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
        week_role_options=week_role_options,
        week_wed_odr_options=week_wed_odr_options,
        selected_day_models={day: 'Any' for day in days},
        selected_day_roles={day: 'Any' for day in days},
        selected_day_week_wed_odrs={day: 'Any' for day in days},
        day_model_result=""
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)