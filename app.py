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

# Level hierarchy for reference (simplified usage)
LEVEL_HIERARCHY = {
    "Min": ["Min"],
    "Min-Med": ["Min-Med", "Min"],
    "Med-Max": ["Med-Max", "Min-Med", "Min"],
    "Max Extreme": ["Max Extreme", "Med-Max", "Min-Med", "Min"],
    "Unknown": ["Unknown"]
}

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
    logger.debug(f"Loaded dropdowns: locations_low={locations_low}, locations_high={locations_high}, total_rows={total_rows}")

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
        matching_levels = LEVEL_HIERARCHY.get(high_level_hit, [high_level_hit])
        filtered_df = filtered_df[filtered_df["High Level Hit"].isin(matching_levels)]
    if color_high != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == color_high]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_level_hit != "Any":
        matching_levels = LEVEL_HIERARCHY.get(low_level_hit, [low_level_hit])
        filtered_df = filtered_df[filtered_df["Low Level Hit"].isin(matching_levels)]
    if color != "Any":
        filtered_df = filtered_df[filtered_df["Low color"] == color]
    
    matching_rows = len(filtered_df)
    probability = matching_rows / total_rows if total_rows > 0 else 0
    logger.debug(f"Scenario probability: matching_rows={matching_rows}, probability={probability}, filters={{odr_start={odr_start}, color={color}, high_level_hit={high_level_hit}}}")
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
        matching_levels = LEVEL_HIERARCHY.get(high_level_hit, [high_level_hit])
        filtered_df = filtered_df[filtered_df["High Level Hit"].isin(matching_levels)]
    if color_high != "Any":
        filtered_df = filtered_df[filtered_df["High color"] == color_high]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_level_hit != "Any":
        matching_levels = LEVEL_HIERARCHY.get(low_level_hit, [low_level_hit])
        filtered_df = filtered_df[filtered_df["Low Level Hit"].isin(matching_levels)]
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

    # Dynamically calculate target probabilities
    output = [location_summary]
    # Calculate for High Level Hit
    high_level_counts = filtered_df['High Level Hit'].value_counts().sort_index()
    total_high = high_level_counts.sum()
    for level, count in high_level_counts.items():
        percentage = (count / total_high) * 100 if total_high > 0 else 0
        output.append(f"High {level}: {percentage:.1f}%")
    
    # Calculate for Low Level Hit
    low_level_counts = filtered_df['Low Level Hit'].value_counts().sort_index()
    total_low = low_level_counts.sum()
    for level, count in low_level_counts.items():
        percentage = (count / total_low) * 100 if total_low > 0 else 0
        output.append(f"Low {level}: {percentage:.1f}%")

    return output

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
    logger.debug(f"Request method: {request.method}, Form data: {request.form.to_dict()}")
    
    if request.method == "POST" and "file" not in request.form:
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