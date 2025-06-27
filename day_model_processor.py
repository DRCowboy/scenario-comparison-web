import pandas as pd
import os
from datetime import datetime, timedelta

# --- Set File Paths ---
home_dir = os.path.expanduser("~")
csv_path = os.path.join(home_dir, "ScenarioComparisonWeb", "data", "CLhistorical5m.csv")
output_path = os.path.join(home_dir, "ScenarioComparisonWeb", "data", "day_model_probabilities.txt")

# --- Load Historical Data ---
try:
    df = pd.read_csv(csv_path, parse_dates=['time'])
except FileNotFoundError:
    with open(output_path, 'w') as f:
        f.write(f"Error: File {csv_path} not found. Please check the file path.\n")
    print(f"Error: File {csv_path} not found. Check output in {output_path}")
    raise SystemExit
except Exception as e:
    with open(output_path, 'w') as f:
        f.write(f"Error loading CSV: {e}\n")
    print(f"Error loading CSV: {e}. Check output in {output_path}")
    raise SystemExit

# Ensure required columns exist
required_columns = ['time', 'open', 'high', 'low', 'close']
if not all(col in df.columns for col in required_columns):
    with open(output_path, 'w') as f:
        f.write(f"Error: CSV must contain columns: {required_columns}\n")
    print(f"Error: CSV must contain columns: {required_columns}. Check output in {output_path}")
    raise SystemExit

# --- Helper Function ---
def identify_day_type(day1_high, day1_low, day2_high, day2_low):
    """Identify day model based on high/low comparison."""
    if day2_high > day1_high and day2_low > day1_low:
        return "Upside"
    elif day2_high < day1_high and day2_low < day1_low:
        return "Downside"
    elif day2_high < day1_high and day2_low > day1_low:
        return "Inside"
    elif day2_high > day1_high and day2_low < day1_low:
        return "Outside"
    return "Undefined"

# --- Process Daily Data with Proper Day Timing ---
def process_daily_data_with_timing(df):
    """Process data with proper day timing starting from Tuesday 9:30 AM."""
    
    # Filter data to only include trading hours (9:30 AM to 4:00 PM ET)
    df['time_et'] = df['time']  # Assuming time is already in ET
    df['hour'] = df['time_et'].dt.hour
    df['minute'] = df['time_et'].dt.minute
    
    # Filter for trading hours (9:30 AM to 4:00 PM)
    trading_mask = (
        ((df['hour'] == 9) & (df['minute'] >= 30)) |
        ((df['hour'] > 9) & (df['hour'] < 16)) |
        ((df['hour'] == 16) & (df['minute'] == 0))
    )
    df_trading = df[trading_mask].copy()
    
    # Add date and day of week
    df_trading['date'] = df_trading['time_et'].dt.date
    df_trading['day_of_week'] = df_trading['time_et'].dt.day_name()
    
    # Aggregate to daily high/low
    daily_data = df_trading.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'time_et': 'first'  # Keep first timestamp to get day of week
    }).reset_index()
    
    # Add day of week
    daily_data['day_of_week'] = pd.to_datetime(daily_data['time_et']).dt.day_name()
    
    # Sort by date to ensure proper order
    daily_data = daily_data.sort_values('date').reset_index(drop=True)
    
    return daily_data

# --- Identify Trading Days Starting from Tuesday ---
def get_trading_week_days(daily_data):
    """Get trading days starting from Tuesday, ensuring proper day sequence."""
    
    # Find the first Tuesday in the data
    tuesday_data = daily_data[daily_data['day_of_week'] == 'Tuesday']
    
    if tuesday_data.empty:
        with open(output_path, 'w') as f:
            f.write("Error: No Tuesday data found in the dataset.\n")
        print("Error: No Tuesday data found in the dataset.")
        raise SystemExit
    
    first_tuesday = pd.to_datetime(tuesday_data.iloc[0]['date']).date()
    
    # Get the week starting from the first Tuesday
    week_start = first_tuesday
    week_end = first_tuesday + timedelta(days=6)
    
    # Get the first 5 trading days starting from Tuesday
    week_days = daily_data[
        (daily_data['date'] >= week_start) & 
        (daily_data['date'] <= week_end)
    ].copy()
    
    # Ensure we have exactly 5 days (Tuesday through Saturday, or Monday through Friday)
    if len(week_days) < 5:
        # Try to get the next week if we don't have enough days
        next_week_start = first_tuesday + timedelta(days=7)
        next_week_end = next_week_start + timedelta(days=6)
        
        next_week_days = daily_data[
            (daily_data['date'] >= next_week_start) & 
            (daily_data['date'] <= next_week_end)
        ].copy()
        
        if len(next_week_days) >= 5:
            week_days = next_week_days.head(5)
        else:
            # Combine weeks if needed
            combined_weeks = pd.concat([week_days, next_week_days]).drop_duplicates(subset=['date']).head(5)
            week_days = combined_weeks
    
    # Ensure we have exactly 5 days
    week_days = week_days.head(5).reset_index(drop=True)
    
    if len(week_days) < 5:
        with open(output_path, 'w') as f:
            f.write(f"Error: Not enough trading days found. Found {len(week_days)} days, need 5.\n")
        print(f"Error: Not enough trading days found. Found {len(week_days)} days, need 5.")
        raise SystemExit
    
    return week_days

# --- Process the Data ---
daily_data = process_daily_data_with_timing(df)
first_five_days = get_trading_week_days(daily_data)

# Identify day models by comparing consecutive days (Day 2 to 3, Day 3 to 4, Day 4 to 5)
first_five_days['day_model'] = 'Undefined'
for i in range(1, len(first_five_days)):
    day1 = first_five_days.iloc[i-1]
    day2 = first_five_days.iloc[i]
    first_five_days.at[i, 'day_model'] = identify_day_type(
        day1['high'], day1['low'], day2['high'], day2['low']
    )

# --- Identify High/Low Days for First Five Days ---
high_low_days = []
for i, row in first_five_days.iterrows():
    day_num = i + 1
    high_low_days.append({
        'Day': f"Day {day_num}",
        'Date': row['date'],
        'Day_of_Week': row['day_of_week'],
        'High': row['high'],
        'Low': row['low'],
        'Day_Model': row['day_model']
    })

# --- Save Output to File ---
with open(output_path, 'w') as f:
    f.write(f"Results as of {datetime.now().strftime('%B %d, %Y, %I:%M %p %Z')}\n\n")
    
    f.write("Day Models (applied between consecutive days):\n")
    f.write("Day 1: Undefined (no previous day to compare)\n")
    for i in range(1, len(first_five_days)):
        day_num = i + 1
        day_model = first_five_days.iloc[i]['day_model']
        day_name = first_five_days.iloc[i]['day_of_week']
        date = first_five_days.iloc[i]['date']
        f.write(f"Day {day_num} ({day_name}, {date}): {day_model}\n")
    
    f.write("\nHigh and Low Days for Days 1-5:\n")
    for entry in high_low_days:
        f.write(f"{entry['Day']} ({entry['Day_of_Week']}, {entry['Date']}): High={entry['High']:.2f}, Low={entry['Low']:.2f}\n")
    
    f.write("\nDay Model Logic Applied:\n")
    f.write("- Day 2 model: Compares Day 1 vs Day 2\n")
    f.write("- Day 3 model: Compares Day 2 vs Day 3\n")
    f.write("- Day 4 model: Compares Day 3 vs Day 4\n")
    f.write("- Day 5 model: Compares Day 4 vs Day 5\n")
    
    f.write("\nDay Model Definitions:\n")
    f.write("- Upside: Day 2 high > Day 1 high AND Day 2 low > Day 1 low\n")
    f.write("- Downside: Day 2 high < Day 1 high AND Day 2 low < Day 1 low\n")
    f.write("- Inside: Day 2 high < Day 1 high AND Day 2 low > Day 1 low\n")
    f.write("- Outside: Day 2 high > Day 1 high AND Day 2 low < Day 1 low\n")

print(f"Output saved to: {output_path}")
print("\nDay Models and High/Low Analysis:")
print("=" * 50)
with open(output_path, 'r') as f:
    print(f.read()) 