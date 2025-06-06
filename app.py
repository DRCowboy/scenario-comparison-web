from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Define the path to the Excel file in the project’s data folder
excel_file_path = os.path.join(os.path.dirname(__file__), "data", "DDR_Predictor.xlsx")

# Load and process the Excel file
def load_excel_file():
    global df, num_columns, total_rows, odr_starts, start_colors, odr_models, odr_true_false, locations_low, low_level_hits, colors, locations_high, high_level_hits, colors_high
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(f"The file {excel_file_path} does not exist.")
    
    try:
        xl = pd.ExcelFile(excel_file_path)
        sheet_names = xl.sheet_names
        df_raw = None
        for sheet in sheet_names:
            temp_df = pd.read_excel(excel_file_path, sheet_name=sheet, header=0)
            num_columns = len(temp_df.columns)
            if num_columns == 11:
                df_raw = temp_df
                break
            elif num_columns == 9:
                df_raw = temp_df
        
        if df_raw is None:
            raise ValueError("No sheet found with 9 or 11 columns.")
    except Exception as e:
        raise Exception(f"Failed to load the Excel file: {e}")

    # Define column names
    if num_columns == 11:
        column_names = [
            "Date", "Odr start", "Start color", "Location of Low", "Low Level Hit", "Low Color",
            "Location of High", "High Level Hit", "High color", "ODR Model", "ODR True/False"
        ]
    elif num_columns == 9:
        column_names = [
            "Date", "Location of Low", "Low Level Hit", "Low Color",
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
    low_level_hits = sorted(df["Low Level Hit"].unique())
    colors = sorted(df["Low Color"].unique())
    locations_high = sorted(df["Location of High"].unique())
    high_level_hits = sorted(df["High Level Hit"].unique())
    colors_high = sorted(df["High color"].unique())

# Initial load of the Excel file
load_excel_file()

# Define LEVEL_HIERARCHY and TARGETS
LEVEL_HIERARCHY = {
    "Min": ["Min"],
    "Min-Med": ["Min-Med", "Min"],
    "Med-Max": ["Med-Max", "Min-Med", "Min"],
    "Max Extreme": ["Max Extreme", "Med-Max", "Min-Med", "Min"],
    "Unknown": ["Unknown"]
}

TARGETS = {
    "Med-Max": ["Med-Max", "Max Extreme"],
    "Max Extreme": ["Max Extreme"],
    "Min": ["Min"]
}

# Functions for probability calculations
def get_matching_levels(selected_level):
    return LEVEL_HIERARCHY.get(selected_level, [selected_level])

def calculate_scenario_probability(odr_start, start_color, odr_model, odr_true_false, location_high, high_level_hit, high_color, location_low, low_level_hit, low_color):
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
        matching_levels = get_matching_levels(high_level_hit)
        filtered_df = filtered_df[filtered_df["High Level Hit"].isin(matching_levels)]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_color != "Any":
        filtered_df = filtered_df[filtered_df["Low Color"] == low_color]
    if low_level_hit != "Any":
        matching_levels = get_matching_levels(low_level_hit)
        filtered_df = filtered_df[filtered_df["Low Level Hit"].isin(matching_levels)]
    matching_rows = len(filtered_df)
    probability = (matching_rows / total_rows) * 100 if total_rows > 0 else 0.0
    return matching_rows, probability

def calculate_target_probabilities(odr_start, start_color, odr_model, odr_true_false, location_high, high_level_hit, high_color, location_low, low_level_hit, low_color):
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
        matching_levels = get_matching_levels(high_level_hit)
        filtered_df = filtered_df[filtered_df["High Level Hit"].isin(matching_levels)]
    if location_low != "Any":
        filtered_df = filtered_df[filtered_df["Location of Low"] == location_low]
    if low_color != "Any":
        filtered_df = filtered_df[filtered_df["Low Color"] == low_color]
    if low_level_hit != "Any":
        matching_levels = get_matching_levels(low_level_hit)
        filtered_df = filtered_df[filtered_df["Low Level Hit"].isin(matching_levels)]
    matching_rows = len(filtered_df)
    if matching_rows == 0:
        return ["No matching data found for this scenario."]
    output = []
    for target, target_levels in TARGETS.items():
        high_target_df = filtered_df[filtered_df["High Level Hit"].isin(target_levels)]
        low_target_df = filtered_df[filtered_df["Low Level Hit"].isin(target_levels)]
        high_target_count = len(high_target_df)
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
        output.extend(location_probs if location_probs else ["    No specific location combinations found."])
    return output

# File upload route
@app.route("/upload", methods=["POST"])
def upload_file():
    global excel_file_path
    if "file" not in request.files:
        return render_template("index.html", error="No file uploaded", 
                             odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
                             start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
                             odr_models=["Any"] + odr_models,
                             odr_true_false=["Any"] + odr_true_false,
                             locations_high=["Any"] + locations_high,
                             high_level_hits=["Any"] + high_level_hits,
                             colors_high=["Any"] + colors_high,
                             locations_low=["Any"] + locations_low,
                             low_level_hits=["Any"] + low_level_hits,
                             colors=["Any"] + colors,
                             result="Comparison will appear here")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error="No file selected", 
                             odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
                             start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
                             odr_models=["Any"] + odr_models,
                             odr_true_false=["Any"] + odr_true_false,
                             locations_high=["Any"] + locations_high,
                             high_level_hits=["Any"] + high_level_hits,
                             colors_high=["Any"] + colors_high,
                             locations_low=["Any"] + locations_low,
                             low_level_hits=["Any"] + low_level_hits,
                             colors=["Any"] + colors,
                             result="Comparison will appear here")
    if file and file.filename.endswith(".xlsx"):
        file.save(excel_file_path)
        try:
            load_excel_file()  # Reload the Excel file to update dropdowns
            return render_template("index.html", success="File uploaded successfully", 
                                 odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
                                 start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
                                 odr_models=["Any"] + odr_models,
                                 odr_true_false=["Any"] + odr_true_false,
                                 locations_high=["Any"] + locations_high,
                                 high_level_hits=["Any"] + high_level_hits,
                                 colors_high=["Any"] + colors_high,
                                 locations_low=["Any"] + locations_low,
                                 low_level_hits=["Any"] + low_level_hits,
                                 colors=["Any"] + colors,
                                 result="Comparison will appear here")
        except Exception as e:
            return render_template("index.html", error=f"Failed to load uploaded file: {str(e)}",
                                 odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
                                 start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
                                 odr_models=["Any"] + odr_models,
                                 odr_true_false=["Any"] + odr_true_false,
                                 locations_high=["Any"] + locations_high,
                                 high_level_hits=["Any"] + high_level_hits,
                                 colors_high=["Any"] + colors_high,
                                 locations_low=["Any"] + locations_low,
                                 low_level_hits=["Any"] + low_level_hits,
                                 colors=["Any"] + colors,
                                 result="Comparison will appear here")
    return render_template("index.html", error="Invalid file type. Please upload an .xlsx file",
                         odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
                         start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
                         odr_models=["Any"] + odr_models,
                         odr_true_false=["Any"] + odr_true_false,
                         locations_high=["Any"] + locations_high,
                         high_level_hits=["Any"] + high_level_hits,
                         colors_high=["Any"] + colors_high,
                         locations_low=["Any"] + locations_low,
                         low_level_hits=["Any"] + low_level_hits,
                         colors=["Any"] + colors,
                         result="Comparison will appear here")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" and "file" not in request.form:
        # Get Scenario 1 inputs
        odr_start1 = request.form.get("odr_start1", "Any") if num_columns == 11 else "Any"
        start_color1 = request.form.get("start_color1", "Any") if num_columns == 11 else "Any"
        odr_model1 = request.form.get("odr_model1", "Any")
        odr_true_false1 = request.form.get("odr_true_false1", "Any")
        location_high1 = request.form.get("location_high1", "Any")
        high_level_hit1 = request.form.get("high_level_hit1", "Any")
        high_color1 = request.form.get("color_high1", "Any")
        location_low1 = request.form.get("location_low1", "Any")
        low_level_hit1 = request.form.get("low_level_hit1", "Any")
        low_color1 = request.form.get("color1", "Any")

        # Get Scenario 2 inputs
        odr_start2 = request.form.get("odr_start2", "Any") if num_columns == 11 else "Any"
        start_color2 = request.form.get("start_color2", "Any") if num_columns == 11 else "Any"
        odr_model2 = request.form.get("odr_model2", "Any")
        odr_true_false2 = request.form.get("odr_true_false2", "Any")
        location_high2 = request.form.get("location_high2", "Any")
        high_level_hit2 = request.form.get("high_level_hit2", "Any")
        high_color2 = request.form.get("color_high2", "Any")
        location_low2 = request.form.get("location_low2", "Any")
        low_level_hit2 = request.form.get("low_level_hit2", "Any")
        low_color2 = request.form.get("color2", "Any")

        # Calculate probabilities
        matching_rows1, prob1 = calculate_scenario_probability(
            odr_start1, start_color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, high_color1,
            location_low1, low_level_hit1, low_color1
        )
        matching_rows2, prob2 = calculate_scenario_probability(
            odr_start2, start_color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, high_color2,
            location_low2, low_level_hit2, low_color2
        )

        # Normalize probabilities
        total_prob = prob1 + prob2
        normalized_prob1 = (prob1 / total_prob) * 100 if total_prob > 0 else 0.0
        normalized_prob2 = (prob2 / total_prob) * 100 if total_prob > 0 else 0.0

        # Calculate target probabilities
        scenario1_lines = calculate_target_probabilities(
            odr_start1, start_color1, odr_model1, odr_true_false1,
            location_high1, high_level_hit1, high_color1,
            location_low1, low_level_hit1, low_color1
        )
        scenario2_lines = calculate_target_probabilities(
            odr_start2, start_color2, odr_model2, odr_true_false2,
            location_high2, high_level_hit2, high_color2,
            location_low2, low_level_hit2, low_color2
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

        return render_template(
            "index.html",
            odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
            start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
            odr_models=["Any"] + odr_models,
            odr_true_false=["Any"] + odr_true_false,
            locations_high=["Any"] + locations_high,
            high_level_hits=["Any"] + high_level_hits,
            colors_high=["Any"] + colors_high,
            locations_low=["Any"] + locations_low,
            low_level_hits=["Any"] + low_level_hits,
            colors=["Any"] + colors,
            result="\n".join(output)
        )

    return render_template(
        "index.html",
        odr_starts=["Any"] + odr_starts if num_columns == 11 else ["Any"],
        start_colors=["Any"] + start_colors if num_columns == 11 else ["Any"],
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