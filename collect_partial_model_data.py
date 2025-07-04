import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import calendar

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_data():
    """Load the historical data CSV file."""
    try:
        df = pd.read_csv('data/CLhistorical5m.csv')
        df['time'] = pd.to_datetime(df['time'])
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def assign_trading_day(df):
    """Assign each row to the correct trading day (1-5) using the same logic as collect_data.py"""
    def get_trading_day(row):
        dt = row['time']
        weekday = dt.weekday()  # Monday=0, Tuesday=1, etc.
        
        # Day 1: Tuesday 9:30 AM to Wednesday 9:25 AM
        if weekday == 1 and dt.hour >= 9 and dt.minute >= 30:  # Tuesday 9:30+
            return 1
        elif weekday == 2 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 25):  # Wednesday before 9:25
            return 1
        
        # Day 2: Wednesday 9:30 AM to Thursday 9:25 AM
        elif weekday == 2 and dt.hour >= 9 and dt.minute >= 30:  # Wednesday 9:30+
            return 2
        elif weekday == 3 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 25):  # Thursday before 9:25
            return 2
        
        # Day 3: Thursday 9:30 AM to Friday 9:25 AM
        elif weekday == 3 and dt.hour >= 9 and dt.minute >= 30:  # Thursday 9:30+
            return 3
        elif weekday == 4 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 25):  # Friday before 9:25
            return 3
        
        # Day 4: Friday 9:30 AM to Monday 9:30 AM
        elif weekday == 4 and dt.hour >= 9 and dt.minute >= 30:  # Friday 9:30+
            return 4
        elif weekday == 0 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):  # Monday before 9:30
            return 4
        
        # Day 5: Monday 9:30 AM to Tuesday 9:25 AM
        elif weekday == 0 and dt.hour >= 9 and dt.minute >= 30:  # Monday 9:30+
            return 5
        elif weekday == 1 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 25):  # Tuesday before 9:25
            return 5
        
        return None
    
    df['trading_day'] = df.apply(get_trading_day, axis=1)
    return df

def determine_partial_model_from_high_low(current_high, current_low, prev_high, prev_low):
    """Determine the model using only high/low, matching the full day model logic."""
    if pd.isna(prev_high) or pd.isna(prev_low):
        return 'Undefined'
    
    # Debug: Print the comparison values
    logger.debug(f"Comparing: Current({current_high:.2f}, {current_low:.2f}) vs Previous({prev_high:.2f}, {prev_low:.2f})")
    
    # Upside: current day high is higher than previous day AND current day low is higher than previous day low
    if current_high > prev_high and current_low > prev_low:
        logger.debug("Result: Upside")
        return 'Upside'
    # Downside: current day low is lower than previous day AND current day high is lower than previous day
    elif current_low < prev_low and current_high < prev_high:
        logger.debug("Result: Downside")
        return 'Downside'
    # Outside: current day high is higher than previous day AND current day low is lower than previous day
    elif current_high > prev_high and current_low < prev_low:
        logger.debug("Result: Outside")
        return 'Outside'
    # Inside: current day high is lower than previous day AND current day low is higher than previous day low
    elif current_high < prev_high and current_low > prev_low:
        logger.debug("Result: Inside")
        return 'Inside'
    else:
        logger.debug("Result: Undefined")
        return 'Undefined'

def collect_partial_models(df):
    """Collect partial model data for each trading day"""
    df['date'] = df['time'].dt.date
    df['time_only'] = df['time'].dt.time
    
    # Create week start (Tuesday) for each date
    def get_week_start(date):
        days_since_tuesday = (date.weekday() - 1) % 7
        return date - timedelta(days=days_since_tuesday)
    
    df['week_start'] = df['date'].apply(get_week_start)
    
    partial_model_data = []
    all_weeks = list(df['week_start'].unique())
    for i, week_start in enumerate(all_weeks):
        week_group = df[df['week_start'] == week_start]
        if len(week_group) == 0:
            continue
        week_trading_days = sorted(week_group['trading_day'].unique())
        for day_num in week_trading_days:
            if day_num is None:
                continue
            # Get the trading day session to determine the calendar date
            current_day_data = week_group[week_group['trading_day'] == day_num]
            if len(current_day_data) == 0:
                continue
            current_day_start = current_day_data['time'].min()
            current_calendar_date = current_day_start.date()
            
            # Get the FULL CALENDAR DAY session (not just trading day session)
            full_calendar_day_data = df[df['date'] == current_calendar_date]
            current_day_full_high = full_calendar_day_data['high'].max()
            current_day_full_low = full_calendar_day_data['low'].min()
            
            # Get previous calendar day
            prev_calendar_day = current_calendar_date - timedelta(days=1)
            prev_day_data = df[df['date'] == prev_calendar_day]
            # Filter for 4:00–9:25 AM session
            prev_day_partial_session = prev_day_data[(prev_day_data['time_only'] >= pd.to_datetime('04:00:00').time()) & (prev_day_data['time_only'] <= pd.to_datetime('09:25:00').time())]
            if len(prev_day_partial_session) == 0:
                partial_model = "N/A"
                prev_day_partial_high = float('nan')
                prev_day_partial_low = float('nan')
            else:
                prev_day_partial_high = prev_day_partial_session['high'].max()
                prev_day_partial_low = prev_day_partial_session['low'].min()
                # Compare current day full session to previous day's 4:00–9:25 AM session
                partial_model = determine_partial_model_from_high_low(current_day_full_high, current_day_full_low, prev_day_partial_high, prev_day_partial_low)
            # Debug output for recent weeks
            if i > len(all_weeks) - 5:
                logger.info(f"Week {week_start}, Day {day_num}, Date {current_calendar_date}")
                logger.info(f"  Current day full: High={current_day_full_high}, Low={current_day_full_low}")
                logger.info(f"  Prev day ({prev_calendar_day}) 4:00–9:25: High={prev_day_partial_high}, Low={prev_day_partial_low}")
                logger.info(f"  Partial model: {partial_model}")
            partial_model_data.append({
                'date': current_calendar_date,
                'week_start': week_start,
                'trading_day': day_num,
                'partial_model': partial_model
            })
    return pd.DataFrame(partial_model_data)

def main():
    """Main function to collect partial model data"""
    logger.info("Starting partial model data collection...")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Assign trading days
    df = assign_trading_day(df)
    logger.info(f"Trading days assigned. Unique days: {sorted(df['trading_day'].unique())}")
    
    # Collect partial models
    partial_models = collect_partial_models(df)
    
    # Save to CSV
    output_file = 'data/partial_model_data.csv'
    partial_models.to_csv(output_file, index=False)
    
    logger.info(f"Partial model data saved to {output_file}")
    logger.info(f"Total records: {len(partial_models)}")
    
    # Show distribution
    model_counts = partial_models['partial_model'].value_counts()
    logger.info("Partial model distribution:")
    for model, count in model_counts.items():
        percentage = (count / len(partial_models)) * 100
        logger.info(f"  {model}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    main() 