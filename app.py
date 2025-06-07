from flask import Flask, render_template, request
import pandas as pd
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the path to the Excel file in the project’s data folder
excel_file_path = os.path.join(os.path.dirname(__file__), "data", "DDR_Predictor.xlsx")

# Level hierarchy for target probability calculations
LEVEL_HIERARCHY = {
    'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4, 'D5': 5, 'D6': 6, 'D7': 7, 'D8': 8, 'D9': 9, 'D10': 10,
    'W1': 11, 'W2': 12, 'W3': 13, 'W4': 14, 'W5': 15, 'W6': 16, 'W7': 17, 'W8': 18, 'W9': 19, 'W10': 20,
    'M1': 21, 'M2': 22, 'M3': 23, 'M4': 24, 'M5': 25, 'M6': 26, 'M7': 27, 'M8': 28, 'M9': 29, 'M10': 30
}

# Target levels for probability calculations
TARGETS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
           'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10',
           'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10']

# Initialize global variables to avoid NameError
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

# Load and process the Excel file
def load_excel_file():
    global df, num_columns, total_rows, odr_starts, start_colors, odr_models, odr_true_false, locations_low, colors, locations_high, high_level_hits, colors_high, low_level_hits
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"The file {excel_file_path} does not exist.")
    
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
            raise ValueError("No sheet found with 9 or 11 columns.")
    except Exception as e:
        raise Exception(f"Failed to load Excel file: {e}")

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
    logger.debug(f"Loaded dropdowns: low_level_hits={low_level_hits}, high_level_hits={high_level_hits}, total_rows={total_rows}")

# Initial load of the Excel file
try:
    load_excel_file()
except Exception as e:
    logger.error(f"Error loading Excel file: {e}")

def calculate_scenario_probability(odr_start, start_color, color, odr_model, odr_true_false, location_high, high_level_hit, color_high, location_low, low_level_hit):
    if df is None or total_rows == 0:
        logger.warning("No data available for probability calculation")
        return 0, 0.0
    
    filtered_df = df.copy()
    
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
        filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
    if color_high != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == color_high]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
    if color != "Any":
        filtered_df = filtered_df[filtered_df["Low color"] == color]
    
    matching_rows = len(filtered_df)
    probability = matching_rows / total_rows if total_rows > 0 else 0
    logger.debug(f"Scenario probability: matching_rows={matching_rows}, probability={probability}, filters={{odr_start={odr_start}, low_level_hit={low_level_hit}, high_level_hit={high_level_hit}}}")
    return matching_rows, probability

def calculate_target_probabilities(odr_start, start_color, color, odr_model, odr_true_false, location_high, high_level_hit, color_high, location_low, low_level_hit):
    if df is None or total_rows == 0:
        logger.warning("No data available for target probabilities")
        return ["No data available"]
    
    filtered_df = df.copy()
    
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
        filtered_df = filtered_df[filtered_df["High Level Hit"] == high_level_hit]
    if color_high != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == color_high]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_level_hit != "Any":
        filtered_df = filtered_df[filtered_df["Low Level Hit"] == low_level_hit]
    if color != "Any":
        filtered_df = filtered_df[filtered_df["Low color"] == color]
    
    logger.debug(f"Filtered rows: {len(filtered_df)}, filters={{odr_start={odr_start}, low_level_hit={low_level_hit}, high_level_hit={high_level_hit}}}")
    
    target_counts = {}
    for target in TARGETS:
        target_counts[target] = 0
    
    unique_high = filtered_df["High Level Hit"].unique().tolist()
    unique_low = filtered_df["Low Level Hit"].unique().tolist()
    logger.debug(f"Unique High Level Hit: {unique_high}, Unique Low Level Hit: {unique_low}")
    
    for _, row in filtered_df.iterrows():
        high_level = row["High Level Hit"]
        low_level = row["Low Level Hit"]
        if high_level in LEVEL_HIERARCHY:
            target_counts[high_level] += 1
        else:
            logger.debug(f"High Level Hit '{high_level}' not in LEVEL_HIERARCHY")
        if low_level in LEVEL_HIERARCHY:
            target_counts[low_level] += 1
        else:
            logger.debug(f"Low Level Hit '{low_level}' not in LEVEL_HIERARCHY")
    
    total_count = sum(target_counts.values())
    output_lines = []
    output_lines.append(f"Filtered Rows: {len(filtered_df)}")
    output_lines.append(f"Total Target Counts: {total_count}")
    
    if total_count > 0:
        for target in TARGETS:
            count = target_counts[target]
            percentage = (count / total_count) * 100
            if count > 0:
                output_lines.append(f"{target}: {count} time{'s' if count != 1 else ''}, {percentage:.1f}%")
    else:
        output_lines.append("No matching targets found")
    
    logger.debug(f"Target probabilities: {output_lines}")
    return output_lines

@app.route("/upload", methods=["POST"])
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
            result="Comparison will appear here"
        )
    
    file = request.files['file']
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
            result="Comparison will appear here"
        )
    
    if file and file.filename.endswith('.xlsx'):
        try:
            global excel_file_path
            file.save(excel_file_path)
            load_excel_file()
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
                result="Comparison will appear here"
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
                result="Comparison will appear here"
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
            result="Comparison will appear here"
        )

@app.route("/", methods=["GET", "POST"])
def index():
    logger.debug(f"Request method: {request.method}, Form data: {request.form}")
    
    if request.method == "POST" and "file" not in request.form:
        # Get Scenario 1 inputs
        odr_start1 = request.form.get("odr_start1", "Any")
        start_color1 = request.form.get("start_color1", "Any")
        odr_model1 = request.form.get("odr_model1", "Any")
        odr_true_false1 = request.form.get("odr_true_false1", "Any")
        location_high1 = request.form.get("location_high1", "Any")
        high_level_hit1 = request.form.get("high_level_hit1", "Any")
        color_high1 = request.form.get("color_high1", "Any")
        location_low1 = request.form.get("location_low1", "Any")
        low_level_hit1 = request.form.get("low_level_hit1", "Any")
        color1 = request.form.get("color1", "Any")

        # Get Scenario 2 inputs
        odr_start2 = request.form.get("odr_start2", "Any")
        start_color2 = request.form.get("start_color2", "Any")
        odr_model2 = request.form.get("odr_model2", "Any")
        odr_true_false2 = request.form.get("odr_true_false2", "Any")
        location_high2 = request.form.get("location_high2", "Any")
        high_level_hit2 = request.form.get("high_level_hit2", "Any")
        color_high2 = request.form.get("color_high2", "Any")
        location_low2 = request.form.get("location_low2", "Any")
        low_level_hit2 = request.form.get("low_level_hit2", "Any")
        color2 = request.form.get("color2", "Any")

        logger.debug(f"Scenario 1 inputs: odr_start1={odr_start1}, low_level_hit1={low_level_hit1}, high_level_hit1={high_level_hit1}")
        logger.debug(f"Scenario 2 inputs: odr_start2={odr_start2}, low_level_hit2={low_level_hit2}, high_level_hit2={high_level_hit2}")

        # Calculate probabilities
        matching_rows1, prob1 = calculate_scenario_probability(
            odr_start1, start_color1, color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, color_high1, location_low1, low_level_hit1
        )
        matching_rows2, prob2 = calculate_scenario_probability(
            odr_start2, start_color2, color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, color_high2, location_low2, low_level_hit2
        )

        # Normalize probabilities
        total_prob = prob1 + prob2
        normalized_prob1 = (prob1 / total_prob) * 100 if total_prob > 0 else 0.0
        normalized_prob2 = (prob2 / total_prob) * 100 if total_prob > 0 else 0.0

        # Calculate target probabilities
        scenario1_lines = calculate_target_probabilities(
            odr_start1, start_color1, color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, color_high1, location_low1, low_level_hit1
        )
        scenario2_lines = calculate_target_probabilities(
            odr_start2, start_color2, color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, color_high2, location_low2, low_level_hit2
        )

        # Format output
        scenario1_lines.insert(0, f"Filtered Dataset: {matching_rows1} out of {total_rows} rows")
        scenario2_lines.insert(0, f"Filtered Dataset: {matching_rows2} out of {total_rows} rows")
        max_lines = max(len(scenario1_lines), len(scenario2_lines))
        scenario1_lines.extend([""] * (max_lines - len(scenario1_lines)))
        scenario2_lines.extend([""] * (max_lines - len(scenario2_lines)))
        output = [
            f"{'Scenario 1:':<50} {'Scenario 2:':>50}",
            f"{'Scenario 1: ' + f'{normalized_prob1:.1f}% chance':<50} {'Scenario 2: ' + f'{normalized_prob2:.1f}% chance':>50}",
            f"{'Dataset: ' + f'Scenario 1 ({matching_rows1}/{total_rows} rows)':<50} {'Scenario 2 ({matching_rows2}/{total_rows} rows)':>50}",
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
            selected_odr_start1=odr_start1,
            selected_start_color1=start_color1,
            selected_odr_model1=odr_model1,
            selected_odr_true_false1=odr_true_false1,
            selected_location_high1=location_high1,
            selected_high_level_hit1=high_level_hit1,
            selected_color_high1=color_high1,
            selected_location_low1=location_low1,
            selected_low_level_hit1=low_level_hit1,
            selected_color1=color1,
            selected_odr_start2=odr_start2,
            selected_start_color2=start_color2,
            selected_odr_model2=odr_model2,
            selected_odr_true_false2=odr_true_false2,
            selected_location_high2=location_high2,
            selected_high_level_hit2=high_level_hit2,
            selected_color_high2=color_high2,
            selected_location_low2=location_low2,
            selected_low_level_hit2=low_level_hit2,
            selected_color2=color2
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
        result="Comparison will appear here"
    )

if __name__ == "__main__":
    app.run(debug=True)