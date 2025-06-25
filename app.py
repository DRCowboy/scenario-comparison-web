from flask import Flask, render_template, request
import pandas as pd
import os
import logging
from datetime import datetime, timedeltaapp = Flask(__name__)# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)# Define paths using relative paths
BASE_DIR = os.path.dirname(__file__)
excel_file_path = os.path.join(BASE_DIR, "data", "DDR_Predictor.xlsx")
csv_path = os.path.join(BASE_DIR, "data", "CLhistorical5m.csv")
output_path = os.path.join(BASE_DIR, "data", "day_model_probabilities.txt")# Ensure data directory exists
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)# Level hierarchy for reference (used only in target probabilities)
LEVEL_HIERARCHY = {
    "Min": ["Min"],
    "Min-Med": ["Min-Med", "Min"],
    "Med-Max": ["Med-Max", "Min-Med", "Min"],
    "Max Extreme": ["Max Extreme", "Med-Max", "Min-Med", "Min"],
    "Unknown": ["Unknown"]
}# Initialize global variables for scenario comparison
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
low_level_hits = []# Initialize global variables for day model analysis
df_day_model = None
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
model_options = ['Any', 'Upside', 'Downside', 'Inside', 'Outside']
week_role_options = ['Any', 'High of Week (HOW)', 'Low of Week (LOW)']# Load and process the Excel file for scenario comparison
def load_excel_file():
    global df, num_columns, total_rows, odr_starts, start_colors, odr_models, odr_true_false, locations_low, colors, locations_high, high_level_hits, colors_high, low_level_hits
    if not os.path.exists(excel_file_path):
        logger.error(f"The file {excel_file_path} does not exist.")
        return Falsetry:
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
logger.debug(f"Loaded dropdowns: locations_low={locations_low}, locations_high={locations_high}, total_rows={total_rows}")
return True# Load and process the CSV file for day model analysis
def load_csv_file():
    global df_day_model
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found")
        return False
    try:
        df_day_model = pd.read_csv(csv_path, parse_dates=['time'])
        logger.debug(f"CSV loaded. Rows: {len(df_day_model)}, Columns: {df_day_model.columns.tolist()}")
        logger.debug(f"Sample data:\n{df_day_model.head().to_string()}")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return Falserequired_columns = ['time', 'open', 'high', 'low', 'close']
if not all(col in df_day_model.columns for col in required_columns):
    logger.error(f"CSV must contain columns: {required_columns}")
    return False
return True# Helper function for day model identification
def identify_day_type(day1_high, day1_low, day2_high, day2_low):
    if day2_high > day1_high and day2_low > day1_low:
        return "Upside"
    elif day2_high < day1_high and day2_low < day1_low:
        return "Downside"
    elif day2_high < day1_high and day2_low > day1_low:
        return "Inside"
    elif day2_high > day1_high and day2_low < day1_low:
        return "Outside"
    return "Undefined"# Process weekly data for day model analysis
def get_week_data(df):
    try:
        df['date'] = df['time'].dt.date
        df['week_start'] = df['time'] - pd.to_timedelta(df['time'].dt.dayofweek, unit='D')
        df['week_start'] = df['week_start'].dt.date
        weekly_data = []
        for week_start, week_group in df.groupby('week_start'):
            week_days = week_group.groupby('date').agg({
                'high': 'max',
                'low': 'min',
                'time': 'first'
            }).reset_index()
            logger.debug(f"Week {week_start}: {len(week_days)} days, Days={week_days['time'].dt.day_name().tolist()}")
            if len(week_days) >= 2:  # Need at least 2 days for models
                week_days['day_of_week'] = pd.to_datetime(week_days['time']).dt.day_name()
                high_day_idx = week_days['high'].idxmax()
                high_day = week_days.loc[high_day_idx, 'day_of_week']
                low_day_idx = week_days['low'].idxmin()
                low_day = week_days.loc[low_day_idx, 'day_of_week']
                logger.debug(f"Week {week_start}: High day={high_day}, Low day={low_day}")
                week_days['model'] = 'Undefined'
                for i in range(1, len(week_days)):
                    day1 = week_days.iloc[i-1]
                    day2 = week_days.iloc[i]
                    week_days.loc[i, 'model'] = identify_day_type(
                        day1['high'], day1['low'], day2['high'], day2['low']
                    )
                weekly_data.append({
                    'week_start': week_start,
                    'days': week_days,
                    'high_day': high_day,
                    'low_day': low_day
                })
        logger.debug(f"Total valid weeks: {len(weekly_data)}")
        return weekly_data
    except Exception as e:
        logger.error(f"Error processing weekly data: {e}")
        return []# Calculate day model probabilities
def compute_day_model_probabilities(conditions):
    if df_day_model is None:
        return f"Error: CSV file {csv_path} not loaded."weekly_data = get_week_data(df_day_model)
if not weekly_data:
    return "Error: No valid weekly data found."

day_indices = {'Day 1': 0, 'Day 2': 1, 'Day 3': 2, 'Day 4': 3, 'Day 5': 4}
last_day_idx = max(day_indices[day] for day in conditions)
target_day = f"Day {last_day_idx + 2}" if last_day_idx < 4 else None

matching_weeks = []
for week in weekly_data:
    days_df = week['days']
    match = True
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
            logger.debug(f"Week {week['week_start']} skipped: {day} not High of Week")
            match = False
            break
        if cond['role'] == 'Low of Week (LOW)' and days_df.iloc[day_idx]['day_of_week'] != week['low_day']:
            logger.debug(f"Week {week['week_start']} skipped: {day} not Low of Week")
            match = False
            break
    if match:
        matching_weeks.append(week)

logger.debug(f"Matching weeks: {len(matching_weeks)}, Conditions: {conditions}")
if not matching_weeks:
    return f"No historical weeks match the selected conditions: {conditions}"

model_probs = {}
if target_day and last_day_idx < 4:
    target_day_idx = last_day_idx + 1
    model_counts = {'Upside': 0, 'Downside': 0, 'Inside': 0, 'Outside': 0}
    total_valid = 0
    for week in matching_weeks:
        days_df = week['days']
        if target_day_idx < len(days_df):
            model = days_df.iloc[target_day_idx]['model']
            if model in model_counts:
                model_counts[model] += 1
                total_valid += 1
    model_probs = {model: count / total_valid if total_valid > 0 else 0 for model, count in model_counts.items()}

# Find the first day with HOW or LOW role
role_day = None
role_type = None
for day, cond in conditions.items():
    if cond['role'] in ['High of Week (HOW)', 'Low of Week (LOW)']:
        role_day = day
        role_type = cond['role']
        break

target_probs = {}
if role_day and role_type:
    role_day_idx = day_indices[role_day]
    remaining_days = [f"Day {i+1}" for i in range(role_day_idx + 1, 5)]
    for day in remaining_days:
        day_idx = day_indices[day]
        if role_type == 'High of Week (HOW)':
            count = sum(1 for week in matching_weeks if day_idx < len(week['days']) and week['days'].iloc[day_idx]['day_of_week'] == week['low_day'])
            target_probs[day] = count / len(matching_weeks) if matching_weeks else 0
        elif role_type == 'Low of Week (LOW)':
            count = sum(1 for week in matching_weeks if day_idx < len(week['days']) and week['days'].iloc[day_idx]['day_of_week'] == week['high_day'])
            target_probs[day] = count / len(matching_weeks) if matching_weeks else 0

result = f"Conditions (as of {datetime.now().strftime('%B %d, %Y, %I:%M %p %Z')}):\n"
for day, cond in conditions.items():
    result += f"{day}: Model={cond['model']}, Role={cond['role']}\n"
if target_day and last_day_idx < 4:
    result += f"\nModel Probabilities for {target_day}:\n"
    for model, prob in model_probs.items():
        result += f"{model}: {prob:.2%}\n"
if target_probs:
    result += f"\n{'Low' if role_type == 'High of Week (HOW)' else 'High'} Day Probabilities:\n"
    for day, prob in target_probs.items():
        result += f"{day}: {prob:.2%}\n"
else:
    result += "\nNo High or Low of Week role selected for any day.\n"

try:
    with open(output_path, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(result)
except Exception as e:
    logger.error(f"Error writing to output file: {e}")

return result# Initial load of files
if not load_excel_file():
    logger.error("Failed to load Excel file for scenario comparison")
if not load_csv_file():
    logger.error("Failed to load CSV file for day model analysis")def calculate_scenario_probability(odr_start, start_color, color, odr_model, odr_true_false, location_high, high_level_hit, color_high, location_low, low_level_hit):
    if df is None or total_rows == 0:
        logger.warning("No data available for probability calculation")
        return 0, 0.0filtered_df = df.copy()

if num_columns == 11 and odr_start != "Any":
    filtered_df = filtered_df[filtered_df["Odr start"] == odr_start]
if num_columns == 11 and start_color != "Any":
    filtered_df = filtered_df[filtered_df["Start color"] == start_color]
if odr_model != "Any":
    filtered_df = filtered_df[filtered_df["ODR Model"] == odr_model]
if odr_true_false != "Any":
    filtered_df = filtered_df[filtered_df["ODR True/False"] == odr_true_false]
if location_high != "Any":
    filtered_df = filtered_df[filtered_df["Location of High"] == location_high]
if high_level_hit != "Any":
    # Use exact matching (no hierarchy) for input filtering
    filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
if color_high != "Any":
    filtered_df = filtered_df[filtered_df["High color"] == color_high]
if location_low != "Any":
    filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
if low_level_hit != "Any":
    # Use exact matching (no hierarchy) for input filtering
    filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
if color != "Any":
    filtered_df = filtered_df[filtered_df["Low color"] == color]

matching_rows = len(filtered_df)
probability = matching_rows / total_rows if total_rows > 0 else 0
logger.debug(f"Scenario probability: matching_rows={matching_rows}, probability={probability}, filters={{odr_start={odr_start}, color={color}, high_level_hit={high_level_hit}}}")
return matching_rows, probabilitydef calculate_target_probabilities(odr_start, start_color, color, odr_model, odr_true_false, location_high, high_level_hit, color_high, location_low, low_level_hit):
    if df is None or total_rows == 0:
        logger.warning("No data available for target probabilities")
        return ["No data available"]filtered_df = df.copy()

if num_columns == 11 and odr_start != "Any":
    filtered_df = filtered_df[filtered_df["Odr start"] == odr_start]
if num_columns == 11 and start_color != "Any":
    filtered_df = filtered_df[filtered_df["Start color"] == start_color]
if odr_model != "Any":
    filtered_df = filtered_df[filtered_df["ODR Model"] == odr_model]
if odr_true_false != "Any":
    filtered_df = filtered_df[filtered_df["ODR True/False"] == odr_true_false]
if location_high != "Any":
    filtered_df = filtered_df[filtered_df["Location of High"] == location_high]
if high_level_hit != "Any":
    # Use exact matching (no hierarchy) for input filtering
    filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
if color_high != "Any":
    filtered_df = filtered_df[filtered_df["High color"] == color_high]
if location_low != "Any":
    filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
if low_level_hit != "Any":
    # Use exact matching (no hierarchy) for input filtering
    filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
if color != "Any":
    filtered_df = filtered_df[filtered_df["Low color"] == color]

logger.debug(f"Filtered rows: {len(filtered_df)}, filters={{odr_start={odr_start}, color={color}, high_level_hit={high_level_hit}}}")

# Calculate most probable Location of High and Location of Low combination
if len(filtered_df) > 0:
    loc_pairs = filtered_df.groupby(['Location of High', 'Location of Low']).size().reset_index(name='counts')
    if not loc_pairs.empty:
        max_pair = loc_pairs.loc[loc_pairs['counts'].idxmax()]
        most_common_high = max_pair['Location of High']
        most_common_low = max_pair['Location of Low']
        total_pairs = len(filtered_df)
        prob_high_low = (max_pair['counts'] / total_pairs) * 100 if total_pairs > 0 else 0
        location_summary = f"{prob_high_low:.1f}% High {most_common_high}-Low {most_common_low}"
    else:
        location_summary = "0.0% High Unknown-Low Unknown"
else:
    location_summary = "0.0% High Unknown-Low Unknown"
logger.debug(f"Location summary: {location_summary}")

matching_rows = len(filtered_df)
if matching_rows == 0:
    return [location_summary, "No matching data found for this scenario."]

# Calculate target probabilities using hierarchy
output = [location_summary]
# Calculate for High Level Hit
for level in high_level_hits:  # Iterate through possible levels
    matching_levels = LEVEL_HIERARCHY.get(level, [level])  # Apply hierarchy
    high_level_count = filtered_df[filtered_df['High Level Hit'].isin(matching_levels)].shape[0]
    total_high = len(filtered_df)
    percentage = (high_level_count / total_high) * 100 if total_high > 0 else 0
    output.append(f"High {level}: {percentage:.1f}%")

# Calculate for Low Level Hit
for level in low_level_hits:  # Iterate through possible levels
    matching_levels = LEVEL_HIERARCHY.get(level, [level])  # Apply hierarchy
    low_level_count = filtered_df[filtered_df['Low Level Hit'].isin(matching_levels)].shape[0]
    total_low = len(filtered_df)
    percentage = (low_level_count / total_low) * 100 if total_low > 0 else 0
    output.append(f"Low {level}: {percentage:.1f}%")

return output@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return render_template(
            "index.html",
            error="No file part in the request",
            odr_starts=["Any"] + odr_starts,
            start_colors=["Any"] + start_colors,
            odr_models=["Any"] + odr_models,
            odr_true_false=["Any"] + odr_true_false,
            locations_high=["Any"] + locations_high,
            high_level_hits=["Any"] + high_level_hits,
            colors_high=["Any"] + colors_high,
            locations_low=["Any"] + locations_low,
            low_level_hits=["Any"] + low_level_hits,
            colors=["Any"] + colors,
            result="Comparison will appear here",
            days=days,
            model_options=model_options,
            week_role_options=week_role_options,
            selected_day_models={day: "Any" for day in days},
            selected_day_roles={day: "Any" for day in days},
            day_model_result="Day model results will appear here"
        )file = request.files['file']
if file.filename == '':
    return render_template(
        "index.html",
        error="No file selected",
        odr_starts=["Any"] + odr_starts,
        start_colors=["Any"] + start_colors,
        odr_models=["Any"] + odr_models,
        odr_true_false=["Any"] + odr_true_false,
        locations_high=["Any"] + locations_high,
        high_level_hits=["Any"] + high_level_hits,
        colors_high=["Any"] + colors_high,
        locations_low=["Any"] + locations_low,
        low_level_hits=["Any"] + low_level_hits,
        colors=["Any"] + colors,
        result="Comparison will appear here",
        days=days,
        model_options=model_options,
        week_role_options=week_role_options,
        selected_day_models={day: "Any" for day in days},
        selected_day_roles={day: "Any" for day in days},
        day_model_result="Day model results will appear here"
    )

if file and file.filename.endswith('.xlsx'):
    try:
        file.save(excel_file_path)
        if load_excel_file():
            return render_template(
                "index.html",
                success="File uploaded successfully",
                odr_starts=["Any"] + odr_starts,
                start_colors=["Any"] + start_colors,
                odr_models=["Any"] + odr_models,
                odr_true_false=["Any"] + odr_true_false,
                locations_high=["Any"] + locations_high,
                high_level_hits=["Any"] + high_level_hits,
                colors_high=["Any"] + colors_high,
                locations_low=["Any"] + locations_low,
                low_level_hits=["Any"] + low_level_hits,
                colors=["Any"] + colors,
                result="Comparison will appear here",
                days=days,
                model_options=model_options,
                week_role_options=week_role_options,
                selected_day_models={day: "Any" for day in days},
                selected_day_roles={day: "Any" for day in days},
                day_model_result="Day model results will appear here"
            )
        else:
            return render_template(
                "index.html",
                error="Failed to load uploaded Excel file",
                odr_starts=["Any"] + odr_starts,
                start_colors=["Any"] + start_colors,
                odr_models=["Any"] + odr_models,
                odr_true_false=["Any"] + odr_true_false,
                locations_high=["Any"] + locations_high,
                high_level_hits=["Any"] + high_level_hits,
                colors_high=["Any"] + colors_high,
                locations_low=["Any"] + locations_low,
                low_level_hits=["Any"] + low_level_hits,
                colors=["Any"] + colors,
                result="Comparison will appear here",
                days=days,
                model_options=model_options,
                week_role_options=week_role_options,
                selected_day_models={day: "Any" for day in days},
                selected_day_roles={day: "Any" for day in days},
                day_model_result="Day model results will appear here"
            )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return render_template(
            "index.html",
            error=f"Error processing file: {str(e)}",
            odr_starts=["Any"] + odr_starts,
            start_colors=["Any"] + start_colors,
            odr_models=["Any"] + odr_models,
            odr_true_false=["Any"] + odr_true_false,
            locations_high=["Any"] + locations_high,
            high_level_hits=["Any"] + high_level_hits,
            colors_high=["Any"] + colors_high,
            locations_low=["Any"] + locations_low,
            low_level_hits=["Any"] + low_level_hits,
            colors=["Any"] + colors,
            result="Comparison will appear here",
            days=days,
            model_options=model_options,
            week_role_options=week_role_options,
            selected_day_models={day: "Any" for day in days},
            selected_day_roles={day: "Any" for day in days},
            day_model_result="Day model results will appear here"
        )
else:
    return render_template(
        "index.html",
        error="Invalid file format. Please upload an .xlsx file.",
        odr_starts=["Any"] + odr_starts,
        start_colors=["Any"] + start_colors,
        odr_models=["Any"] + odr_models,
        odr_true_false=["Any"] + odr_true_false,
        locations_high=["Any"] + locations_high,
        high_level_hits=["Any"] + high_level_hits,
        colors_high=["Any"] + colors_high,
        locations_low=["Any"] + locations_low,
        low_level_hits=["Any"] + low_level_hits,
        colors=["Any"] + colors,
        result="Comparison will appear here",
        days=days,
        model_options=model_options,
        week_role_options=week_role_options,
        selected_day_models={day: "Any" for day in days},
        selected_day_roles={day: "Any" for day in days},
        day_model_result="Day model results will appear here"
    )@app.route("/", methods=["GET", "POST"])
def index():
    logger.debug(f"Request method: {request.method}, Form data: {request.form.to_dict()}")if request.method == "POST" and "file" not in request.form:
    # Get shared inputs from Scenario 1 with default "Any" if not present
    selected_odr_start = request.form.get("odr_start1", "Any")
    selected_start_color = request.form.get("start_color1", "Any")
    selected_odr_model = request.form.get("odr_model1", "Any")
    selected_odr_true_false = request.form.get("odr_true_false1", "Any")
    
    # Log received form data for debugging
    logger.debug(f"Received form data: odr_start1={request.form.get('odr_start1')}, odr_true_false1={request.form.get('odr_true_false1')}")

    # Get Scenario 1 specific inputs
    location_high1 = request.form.get("location_high1", "Any")
    high_level_hit1 = request.form.get("high_level_hit1", "Any")
    color_high1 = request.form.get("color_high1", "Any")
    location_low1 = request.form.get("location_low1", "Any")
    low_level_hit1 = request.form.get("low_level_hit1", "Any")
    color1 = request.form.get("color1", "Any")

    # Get Scenario 2 specific inputs
    location_high2 = request.form.get("location_high2", "Any")
    high_level_hit2 = request.form.get("high_level_hit2", "Any")
    color_high2 = request.form.get("color_high2", "Any")
    location_low2 = request.form.get("location_low2", "Any")
    low_level_hit2 = request.form.get("low_level_hit2", "Any")
    color2 = request.form.get("color2", "Any")

    logger.debug(f"Shared inputs: selected_odr_start={selected_odr_start}, selected_odr_model={selected_odr_model}, selected_odr_true_false={selected_odr_true_false}")
    logger.debug(f"Scenario 1 inputs: location_high1={location_high1}, high_level_hit1={high_level_hit1}, color1={color1}")
    logger.debug(f"Scenario 2 inputs: location_high2={location_high2}, high_level_hit2={high_level_hit2}, color2={color2}")

    # Calculate probabilities
    matching_rows1, prob1 = calculate_scenario_probability(
        selected_odr_start, selected_start_color, color1, selected_odr_model, selected_odr_true_false,
        location_high1, high_level_hit1, color_high1, location_low1, low_level_hit1
    )
    matching_rows2, prob2 = calculate_scenario_probability(
        selected_odr_start, selected_start_color, color2, selected_odr_model, selected_odr_true_false,
        location_high2, high_level_hit2, color_high2, location_low2, low_level_hit2
    )

    # Normalize probabilities
    total_prob = prob1 + prob2 if prob1 + prob2 > 0 else 1  # Avoid division by zero
    normalized_prob1 = (prob1 / total_prob) * 100 if total_prob > 0 else 0.0
    normalized_prob2 = (prob2 / total_prob) * 100 if total_prob > 0 else 0.0

    # Calculate target probabilities
    scenario1_lines = calculate_target_probabilities(
        selected_odr_start, selected_start_color, color1, selected_odr_model, selected_odr_true_false,
        location_high1, high_level_hit1, color_high1, location_low1, low_level_hit1
    )
    scenario2_lines = calculate_target_probabilities(
        selected_odr_start, selected_start_color, color2, selected_odr_model, selected_odr_true_false,
        location_high2, high_level_hit2, color_high2, location_low2, low_level_hit2
    )

    # Format output with proper variable substitution
    output = [
        f"{'Scenario 1:':<50} {'Scenario 2:':>50}",
        f"{'Scenario 1: ' + f'{normalized_prob1:.1f}% chance':<50} {'Scenario 2: ' + f'{normalized_prob2:.1f}% chance':>50}",
        f"Dataset: Scenario 1 ({matching_rows1}/{total_rows} rows)         Dataset: Scenario 2 ({matching_rows2}/{total_rows} rows)",
        "=" * 100
    ]
    output.extend(f"{line1:<50} {line2:>50}" for line1, line2 in zip(scenario1_lines, scenario2_lines))
    logger.debug(f"Final output: {output}")

    return render_template(
        "index.html",
        odr_starts=["Any"] + odr_starts,
        start_colors=["Any"] + start_colors,
        odr_models=["Any"] + odr_models,
        odr_true_false=["Any"] + odr_true_false,
        locations_high=["Any"] + locations_high,
        high_level_hits=["Any"] + high_level_hits,
        colors_high=["Any"] + colors_high,
        locations_low=["Any"] + locations_low,
        low_level_hits=["Any"] + low_level_hits,
        colors=["Any"] + colors,
        result="\n".join(output),
        # Pass selected values
        selected_odr_start=selected_odr_start,
        selected_start_color=selected_start_color,
        selected_odr_model=selected_odr_model,
        selected_odr_true_false=selected_odr_true_false,
        selected_location_high1=location_high1,
        selected_high_level_hit1=high_level_hit1,
        selected_color_high1=color_high1,
        selected_location_low1=location_low1,
        selected_low_level_hit1=low_level_hit1,
        selected_color1=color1,
        selected_location_high2=location_high2,
        selected_high_level_hit2=high_level_hit2,
        selected_color_high2=color_high2,
        selected_location_low2=location_low2,
        selected_low_level_hit2=low_level_hit2,
        selected_color2=color2,
        days=days,
        model_options=model_options,
        week_role_options=week_role_options,
        selected_day_models={day: "Any" for day in days},
        selected_day_roles={day: "Any" for day in days},
        day_model_result="Day model results will appear here"
    )

return render_template(
    "index.html",
    odr_starts=["Any"] + odr_starts,
    start_colors=["Any"] + start_colors,
    odr_models=["Any"] + odr_models,
    odr_true_false=["Any"] + odr_true_false,
    locations_high=["Any"] + locations_high,
    high_level_hits=["Any"] + high_level_hits,
    colors_high=["Any"] + colors_high,
    locations_low=["Any"] + locations_low,
    low_level_hits=["Any"] + low_level_hits,
    colors=["Any"] + colors,
    result="Comparison will appear here",
    days=days,
    model_options=model_options,
    week_role_options=week_role_options,
    selected_day_models={day: "Any" for day in days},
    selected_day_roles={day: "Any" for day in days},
    day_model_result="Day model results will appear here"
)@app.route("/day_model", methods=["POST"])
def day_model():
    conditions = {}
    selected_day_models = {}
    selected_day_roles = {}
    for day in days:
        model = request.form.get(f"{day.lower().replace(' ', '_')}_model", "Any")
        role = request.form.get(f"{day.lower().replace(' ', '_')}_role", "Any")
        selected_day_models[day] = model
        selected_day_roles[day] = role
        if model != "Any" or role != "Any":
            conditions[day] = {'model': model, 'role': role}day_model_result = "Please select at least one condition."
if conditions:
    day_model_result = compute_day_model_probabilities(conditions)

return render_template(
    "index.html",
    odr_starts=["Any"] + odr_starts,
    start_colors=["Any"] + start_colors,
    odr_models=["Any"] + odr_models,
    odr_true_false=["Any"] + odr_true_false,
    locations_high=["Any"] + locations_high,
    high_level_hits=["Any"] + high_level_hits,
    colors_high=["Any"] + colors_high,
    locations_low=["Any"] + locations_low,
    low_level_hits=["Any"] + low_level_hits,
    colors=["Any"] + colors,
    result="Comparison will appear here",
    days=days,
    model_options=model_options,
    week_role_options=week_role_options,
    selected_day_models=selected_day_models,
    selected_day_roles=selected_day_roles,
    day_model_result=day_model_result
)if __name__ == "__main__":
    app.run(debug=True)