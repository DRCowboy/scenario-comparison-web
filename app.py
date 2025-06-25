from flask import Flask, render_template, request
import pandas as pd
import os
import logging
import gc
from datetime import datetime, timedelta

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(__file__)
excel_file_path = os.path.join(BASE_DIR, "data", "DDR_Predictor.xlsx")
csv_path = os.path.join(BASE_DIR, "data", "CLhistorical5m.csv")
output_path = os.path.join(BASE_DIR, "data", "day_model_probabilities.txt")

# Ensure data directory exists
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

# Level hierarchy
LEVEL_HIERARCHY = {
    "Min": ["Min"],
    "Min-Med": ["Min-Med", "Min"],
    "Med-Max": ["Med-Max", "Min-Med", "Min"],
    "Max Extreme": ["Max Extreme", "Med-Max", "Min-Med", "Min"],
    "Unknown": ["Unknown"]
}

# Dropdown options (loaded dynamically)
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
model_options = ['Any', 'Upside', 'Downside', 'Inside', 'Outside']
week_role_options = ['Any', 'High of Week (HOW)', 'Low of Week (LOW)']

def load_excel_file():
    if not os.path.exists(excel_file_path):
        logger.error(f"Excel file {excel_file_path} not found.")
        return None, [], [], [], [], [], [], [], [], [], []
    
    try:
        xl = pd.ExcelFile(excel_file_path)
        for sheet in xl.sheet_names:
            temp_df = pd.read_excel(
                excel_file_path,
                sheet_name=sheet,
                header=0,
                usecols=range(11),  # Limit to max 11 columns
                dtype=str  # Use strings to save memory
            )
            num_columns = len(temp_df.columns)
            if num_columns in [9, 11]:
                df = temp_df
                break
        else:
            logger.error("No sheet with 9 or 11 columns found.")
            return None, [], [], [], [], [], [], [], [], [], []
    except Exception as e:
        logger.error(f"Failed to load Excel file: {e}")
        return None, [], [], [], [], [], [], [], [], [], []

    # Define column names
    if num_columns == 11:
        column_names = [
            "Date", "Odr start", "Start color", "Location of Low", "Low Level Hit", "Low color",
            "Location of High", "High Level Hit", "High color", "ODR Model", "ODR True/False"
        ]
    else:
        column_names = [
            "Date", "Location of Low", "Low Level Hit", "Low color",
            "Location of High", "High Level Hit", "High color", "ODR Model", "ODR True/False"
        ]
    df.columns = column_names[:num_columns]
    df = df.drop(columns=["Date"]).fillna("Unknown")

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

    total_rows = len(df)
    logger.debug(f"Excel loaded: rows={total_rows}, columns={num_columns}")
    return (
        df, odr_starts, start_colors, odr_models, odr_true_false, locations_low,
        colors, locations_high, high_level_hits, colors_high, low_level_hits
    )

def load_csv_file():
    if not os.path.exists(csv_path):
        logger.error(f"CSV file {csv_path} not found.")
        return None
    
    try:
        # Load CSV in chunks to check size
        chunks = pd.read_csv(
            csv_path,
            parse_dates=['time'],
            usecols=['time', 'open', 'high', 'low', 'close'],
            chunksize=10000,
            dtype={
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32'
            }
        )
        df_chunks = []
        for chunk in chunks:
            df_chunks.append(chunk)
            if len(df_chunks) * 10000 > 100000:  # Limit to ~100k rows
                logger.warning("CSV truncated to ~100k rows to save memory.")
                break
        df = pd.concat(df_chunks, ignore_index=True)
        logger.debug(f"CSV loaded: rows={len(df)}, columns={df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None

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
            if len(week_days) >= 2:
                week_days['day_of_week'] = pd.to_datetime(week_days['time']).dt.day_name()
                high_day_idx = week_days['high'].idxmax()
                high_day = week_days.loc[high_day_idx, 'day_of_week']
                low_day_idx = week_days['low'].idxmin()
                low_day = week_days.loc[low_day_idx, 'day_of_week']
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
        return []

def compute_day_model_probabilities(conditions):
    df_day_model = load_csv_file()
    if df_day_model is None:
        result = f"Error: CSV file {csv_path} not loaded."
        return result
    
    weekly_data = get_week_data(df_day_model)
    del df_day_model
    gc.collect()
    
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
                match = False
                break
            if cond['model'] != 'Any' and days_df.iloc[day_idx]['model'] != cond['model']:
                match = False
                break
            if cond['role'] == 'High of Week (HOW)' and days_df.iloc[day_idx]['day_of_week'] != week['high_day']:
                match = False
                break
            if cond['role'] == 'Low of Week (LOW)' and days_df.iloc[day_idx]['day_of_week'] != week['low_day']:
                match = False
                break
        if match:
            matching_weeks.append(week)
    
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
    
    return result

def calculate_scenario_probability(df, total_rows, **kwargs):
    if df is None or total_rows == 0:
        return 0, 0.0
    
    filtered_df = df.copy()
    for key, value in kwargs.items():
        if value != "Any":
            filtered_df = filtered_df[filtered_df[key.replace('_', ' ').title()] == value]
    
    matching_rows = len(filtered_df)
    probability = matching_rows / total_rows if total_rows > 0 else 0
    return matching_rows, probability

def calculate_target_probabilities(df, total_rows, high_level_hits, low_level_hits, **kwargs):
    if df is None or total_rows == 0:
        return ["No data available"]
    
    filtered_df = df.copy()
    for key, value in kwargs.items():
        if value != "Any":
            filtered_df = filtered_df[filtered_df[key.replace('_', ' ').title()] == value]
    
    if len(filtered_df) == 0:
        return ["0.0% High Unknown-Low Unknown", "No matching data found for this scenario."]
    
    loc_pairs = filtered_df.groupby(['Location of High', 'Location of Low']).size().reset_index(name='counts')
    if not loc_pairs.empty:
        max_pair = loc_pairs.loc[loc_pairs['counts'].idxmax()]
        prob_high_low = (max_pair['counts'] / len(filtered_df)) * 100
        location_summary = f"{prob_high_low:.1f}% High {max_pair['Location of High']}-Low {max_pair['Location of Low']}"
    else:
        location_summary = "0.0% High Unknown-Low Unknown"
    
    output = [location_summary]
    for level in high_level_hits:
        matching_levels = LEVEL_HIERARCHY.get(level, [level])
        high_level_count = filtered_df[filtered_df['High Level Hit'].isin(matching_levels)].shape[0]
        percentage = (high_level_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        output.append(f"High {level}: {percentage:.1f}%")
    
    for level in low_level_hits:
        matching_levels = LEVEL_HIERARCHY.get(level, [level])
        low_level_count = filtered_df[filtered_df['Low Level Hit'].isin(matching_levels)].shape[0]
        percentage = (low_level_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        output.append(f"Low {level}: {percentage:.1f}%")
    
    return output

@app.route("/upload", methods=["POST"])
def upload_file():
    (
        df, odr_starts, start_colors, odr_models, odr_true_false, locations_low,
        colors, locations_high, high_level_hits, colors_high, low_level_hits
    ) = load_excel_file()
    total_rows = len(df) if df is not None else 0
    
    if 'file' not in request.files:
        return render_template(
            "index.html",
            error="No file part in the request",
            **locals()
        )
    
    file = request.files['file']
    if file.filename == '':
        return render_template(
            "index.html",
            error="No file selected",
            **locals()
        )
    
    if file and file.filename.endswith('.xlsx'):
        try:
            file.save(excel_file_path)
            (
                df, odr_starts, start_colors, odr_models, odr_true_false, locations_low,
                colors, locations_high, high_level_hits, colors_high, low_level_hits
            ) = load_excel_file()
            total_rows = len(df) if df is not None else 0
            return render_template(
                "index.html",
                success="File uploaded successfully",
                **locals()
            )
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return render_template(
                "index.html",
                error=f"Error processing file: {str(e)}",
                **locals()
            )
    return render_template(
        "index.html",
        error="Invalid file format. Please upload an .xlsx file.",
        **locals()
    )

@app.route("/", methods=["GET", "POST"])
def index():
    (
        df, odr_starts, start_colors, odr_models, odr_true_false, locations_low,
        colors, locations_high, high_level_hits, colors_high, low_level_hits
    ) = load_excel_file()
    total_rows = len(df) if df is not None else 0
    
    if request.method == "POST" and "file" not in request.form:
        # Get inputs
        inputs1 = {
            'odr_start': request.form.get("odr_start1", "Any"),
            'start_color': request.form.get("start_color1", "Any"),
            'odr_model': request.form.get("odr_model1", "Any"),
            'odr_true_false': request.form.get("odr_true_false1", "Any"),
            'location_high': request.form.get("location_high1", "Any"),
            'high_level_hit': request.form.get("high_level_hit1", "Any"),
            'color_high': request.form.get("color_high1", "Any"),
            'location_low': request.form.get("location_low1", "Any"),
            'low_level_hit': request.form.get("low_level_hit1", "Any"),
            'color': request.form.get("color1", "Any")
        }
        inputs2 = {
            'odr_start': inputs1['odr_start'],
            'start_color': inputs1['start_color'],
            'odr_model': inputs1['odr_model'],
            'odr_true_false': inputs1['odr_true_false'],
            'location_high': request.form.get("location_high2", "Any"),
            'high_level_hit': request.form.get("high_level_hit2", "Any"),
            'color_high': request.form.get("color_high2", "Any"),
            'location_low': request.form.get("location_low2", "Any"),
            'low_level_hit': request.form.get("low_level_hit2", "Any"),
            'color': request.form.get("color2", "Any")
        }
        
        # Calculate probabilities
        matching_rows1, prob1 = calculate_scenario_probability(df, total_rows, **inputs1)
        matching_rows2, prob2 = calculate_scenario_probability(df, total_rows, **inputs2)
        
        total_prob = prob1 + prob2 if prob1 + prob2 > 0 else 1
        normalized_prob1 = (prob1 / total_prob) * 100
        normalized_prob2 = (prob2 / total_prob) * 100
        
        scenario1_lines = calculate_target_probabilities(df, total_rows, high_level_hits, low_level_hits, **inputs1)
        scenario2_lines = calculate_target_probabilities(df, total_rows, high_level_hits, low_level_hits, **inputs2)
        
        output = [
            f"{'Scenario 1:':<50} {'Scenario 2:':>50}",
            f"{'Scenario 1: ' + f'{normalized_prob1:.1f}% chance':<50} {'Scenario 2: ' + f'{normalized_prob2:.1f}% chance':>50}",
            f"Dataset: Scenario 1 ({matching_rows1}/{total_rows} rows)         Dataset: Scenario 2 ({matching_rows2}/{total_rows} rows)",
            "=" * 100
        ]
        output.extend(f"{line1:<50} {line2:>50}" for line1, line2 in zip(scenario1_lines, scenario2_lines))
        
        del df
        gc.collect()
        
        return render_template(
            "index.html",
            result="\n".join(output),
            selected_odr_start=inputs1['odr_start'],
            selected_start_color=inputs1['start_color'],
            selected_odr_model=inputs1['odr_model'],
            selected_odr_true_false=inputs1['odr_true_false'],
            selected_location_high1=inputs1['location_high'],
            selected_high_level_hit1=inputs1['high_level_hit'],
            selected_color_high1=inputs1['color_high'],
            selected_location_low1=inputs1['location_low'],
            selected_low_level_hit1=inputs1['low_level_hit'],
            selected_color1=inputs1['color'],
            selected_location_high2=inputs2['location_high'],
            selected_high_level_hit2=inputs2['high_level_hit'],
            selected_color_high2=inputs2['color_high'],
            selected_location_low2=inputs2['location_low'],
            selected_low_level_hit2=inputs2['low_level_hit'],
            selected_color2=inputs2['color'],
            **locals()
        )
    
    return render_template(
        "index.html",
        result="Comparison will appear here",
        **locals()
    )

@app.route("/day_model", methods=["POST"])
def day_model():
    (
        df, odr_starts, start_colors, odr_models, odr_true_false, locations_low,
        colors, locations_high, high_level_hits, colors_high, low_level_hits
    ) = load_excel_file()
    total_rows = len(df) if df is not None else 0
    
    conditions = {}
    selected_day_models = {}
    selected_day_roles = {}
    for day in days:
        model = request.form.get(f"{day.lower().replace(' ', '_')}_model", "Any")
        role = request.form.get(f"{day.lower().replace(' ', '_')}_role", "Any")
        selected_day_models[day] = model
        selected_day_roles[day] = role
        if model != "Any" or role != "Any":
            conditions[day] = {'model': model, 'role': role}
    
    day_model_result = "Please select at least one condition."
    if conditions:
        day_model_result = compute_day_model_probabilities(conditions)
    
    del df
    gc.collect()
    
    return render_template(
        "index.html",
        day_model_result=day_model_result,
        selected_day_models=selected_day_models,
        selected_day_roles=selected_day_roles,
        **locals()
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)