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
        for trading_day in week_trading_days:
            day_group = week_group[week_group['trading_day'] == trading_day]
            if len(day_group) == 0:
                continue
            current_date = day_group['date'].iloc[0]
            # Day 1: no partial model
            if trading_day == 1:
                partial_model = 'N/A'
            else:
                # Current day full session: 9:30am–next 9:25am (use all data for the trading day)
                current_full = day_group
                current_high = current_full['high'].max()
                current_low = current_full['low'].min()
                # Previous session: same calendar day, 4:00–9:25am
                prev_session = df[(df['date'] == current_date) &
                                  (df['time_only'] >= pd.to_datetime('04:00:00').time()) &
                                  (df['time_only'] <= pd.to_datetime('09:25:00').time())]
                prev_high = prev_session['high'].max() if not prev_session.empty else float('nan')
                prev_low = prev_session['low'].min() if not prev_session.empty else float('nan')
                # Model logic
                if pd.isna(prev_high) or pd.isna(prev_low):
                    partial_model = 'Undefined'
                elif current_high > prev_high and current_low > prev_low:
                    partial_model = 'Upside'
                elif current_high < prev_high and current_low < prev_low:
                    partial_model = 'Downside'
                elif current_high > prev_high and current_low < prev_low:
                    partial_model = 'Outside'
                elif current_high < prev_high and current_low > prev_low:
                    partial_model = 'Inside'
                else:
                    partial_model = 'Undefined'
            partial_model_data.append({
                'week_start': week_start,
                'trading_day': trading_day,
                'date': current_date,
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