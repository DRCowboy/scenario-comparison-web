import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    # Upside: current day high is higher than previous day AND current day low is higher than previous day low
    if current_high > prev_high and current_low > prev_low:
        return 'Upside'
    # Downside: current day low is lower than previous day AND current day high is lower than previous day
    elif current_low < prev_low and current_high < prev_high:
        return 'Downside'
    # Outside: current day high is higher than previous day AND current day low is lower than previous day
    elif current_high > prev_high and current_low < prev_low:
        return 'Outside'
    # Inside: current day high is lower than previous day AND current day low is higher than previous day low
    elif current_high < prev_high and current_low > prev_low:
        return 'Inside'
    else:
        return 'Undefined'

def collect_partial_models(df):
    """Collect partial model data for each trading day"""
    # Group by calendar week (starting Tuesday)
    df['date'] = df['time'].dt.date
    
    # Create week start (Tuesday) for each date
    def get_week_start(date):
        days_since_tuesday = (date.weekday() - 1) % 7
        return date - timedelta(days=days_since_tuesday)
    
    df['week_start'] = df['date'].apply(get_week_start)
    
    partial_model_data = []
    
    # Process each week
    for week_start, week_group in df.groupby('week_start'):
        if len(week_group) == 0:
            continue
        
        # Get trading days for this week
        week_trading_days = sorted(week_group['trading_day'].unique())
        
        # For each trading day, get the previous day's partial session
        for day_num in week_trading_days:
            if day_num is None:
                continue
            
            # Get the current day's data
            current_day_data = week_group[week_group['trading_day'] == day_num]
            
            if len(current_day_data) == 0:
                continue
            
            # For partial models, we need the previous day's 4:00 AM - 9:25 AM session
            # Day 2 partial model is based on Day 1's 4:00 AM - 9:25 AM
            # Day 3 partial model is based on Day 2's 4:00 AM - 9:25 AM
            # etc.
            
            if day_num == 1:
                # Day 1 partial model - need to look at previous week's Day 5
                # This is complex, so we'll skip Day 1 partial for now
                partial_model = "N/A"
            else:
                # Get the previous day's data
                prev_day_num = day_num - 1
                prev_day_data = week_group[week_group['trading_day'] == prev_day_num]
                
                if len(prev_day_data) == 0:
                    partial_model = "N/A"
                else:
                    # Get the current day's full high and low
                    current_day_full_high = current_day_data['high'].max()
                    current_day_full_low = current_day_data['low'].min()
                    
                    # Get the previous day's full high and low
                    prev_day_full_high = prev_day_data['high'].max()
                    prev_day_full_low = prev_day_data['low'].min()
                    
                    # Determine partial model using full day high/low comparison
                    partial_model = determine_partial_model_from_high_low(
                        current_day_full_high, current_day_full_low,
                        prev_day_full_high, prev_day_full_low
                    )
            
            # Get the first timestamp of the current day for the date
            current_day_start = current_day_data['time'].min()
            
            partial_model_data.append({
                'date': current_day_start.date(),
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