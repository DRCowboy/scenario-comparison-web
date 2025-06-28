import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_historical_data():
    """Load the historical 5-minute data"""
    try:
        df = pd.read_csv('data/CLhistorical5m.csv')
        df['time'] = pd.to_datetime(df['time'])
        
        # Convert to NY timezone
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        elif df['time'].dt.tz != pytz.timezone('America/New_York'):
            df['time'] = df['time'].dt.tz_convert('America/New_York')
        
        logger.info(f"Loaded {len(df)} rows of historical data")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def assign_trading_day(df):
    """Assign each row to the correct trading day (1-5) using the same logic as collect_data.py"""
    def get_trading_day(row):
        timestamp = row['time']
        day_of_week = timestamp.day_name()
        time_str = timestamp.strftime('%H:%M')
        
        # Day 1: Tuesday 9:30 AM to Wednesday 9:25 AM
        if day_of_week == 'Tuesday' and time_str >= '09:30':
            return 1
        elif day_of_week == 'Wednesday' and time_str < '09:25':
            return 1
        
        # Day 2: Wednesday 9:30 AM to Thursday 9:25 AM
        elif day_of_week == 'Wednesday' and time_str >= '09:30':
            return 2
        elif day_of_week == 'Thursday' and time_str < '09:25':
            return 2
        
        # Day 3: Thursday 9:30 AM to Friday 9:25 AM
        elif day_of_week == 'Thursday' and time_str >= '09:30':
            return 3
        elif day_of_week == 'Friday' and time_str < '09:25':
            return 3
        
        # Day 4: Friday 9:30 AM to Monday 9:30 AM
        elif day_of_week == 'Friday' and time_str >= '09:30':
            return 4
        elif day_of_week in ['Saturday', 'Sunday']:
            return 4
        elif day_of_week == 'Monday' and time_str < '09:30':
            return 4
        
        # Day 5: Monday 9:30 AM to Tuesday 9:25 AM
        elif day_of_week == 'Monday' and time_str >= '09:30':
            return 5
        elif day_of_week == 'Tuesday' and time_str < '09:25':
            return 5
        
        return None
    
    df['trading_day'] = df.apply(get_trading_day, axis=1)
    return df

def determine_model(day_data, prev_day_data=None):
    """
    Determine the model for a trading day based on price action:
    - Upside: High > Previous High and Low >= Previous Low
    - Downside: Low < Previous Low and High <= Previous High
    - Inside: High < Previous High and Low >= Previous Low
    - Outside: High > Previous High and Low < Previous Low
    - Undefined: Doesn't fit any of the above patterns
    """
    if prev_day_data is None:
        return 'Undefined'  # First day of data
    
    current_high = day_data['high']
    current_low = day_data['low']
    
    prev_high = prev_day_data['high']
    prev_low = prev_day_data['low']
    
    # Upside: High > Previous High and Low >= Previous Low
    if current_high > prev_high and current_low >= prev_low:
        return 'Upside'
    
    # Downside: Low < Previous Low and High <= Previous High
    elif current_low < prev_low and current_high <= prev_high:
        return 'Downside'
    
    # Inside: High < Previous High and Low >= Previous Low
    elif current_high < prev_high and current_low >= prev_low:
        return 'Inside'
    
    # Outside: High > Previous High and Low < Previous Low
    elif current_high > prev_high and current_low < prev_low:
        return 'Outside'
    
    # Undefined: Doesn't fit any of the above patterns
    else:
        return 'Undefined'

def get_week_model_data(df):
    """Process data into weekly structure with model classifications"""
    # Filter out rows that don't belong to any trading day
    df = df[df['trading_day'].notna()].copy()
    
    # Group by calendar week (starting Tuesday)
    df['date'] = df['time'].dt.date
    df['day_of_week'] = df['time'].dt.day_name()
    
    # Create week start (Tuesday) for each date
    def get_week_start(date):
        days_since_tuesday = (date.weekday() - 1) % 7
        return date - timedelta(days=days_since_tuesday)
    
    df['week_start'] = df['date'].apply(get_week_start)
    
    weekly_data = []
    
    for week_start, week_group in df.groupby('week_start'):
        if len(week_group) == 0:
            continue
            
        # Group by trading day within the week
        daily_data = {}
        for trading_day in range(1, 6):
            day_data = week_group[week_group['trading_day'] == trading_day]
            if len(day_data) > 0:
                daily_data[trading_day] = {
                    'open': day_data['open'].iloc[0],
                    'high': day_data['high'].max(),
                    'low': day_data['low'].min(),
                    'close': day_data['close'].iloc[-1],
                    'day_name': day_data['day_of_week'].iloc[0]
                }
        
        if len(daily_data) >= 3:  # At least 3 trading days
            # Determine models for each day
            models = {}
            prev_day_data = None
            
            for day_num in sorted(daily_data.keys()):
                day_data = daily_data[day_num]
                model = determine_model(day_data, prev_day_data)
                models[day_num] = model
                prev_day_data = day_data
            
            # Find high and low of week
            all_highs = [(day, data['high']) for day, data in daily_data.items()]
            all_lows = [(day, data['low']) for day, data in daily_data.items()]
            
            high_day = max(all_highs, key=lambda x: x[1])[0]
            low_day = min(all_lows, key=lambda x: x[1])[0]
            
            weekly_data.append({
                'week_start': week_start,
                'days': daily_data,
                'models': models,
                'high_day': high_day,
                'low_day': low_day,
                'num_days': len(daily_data)
            })
    
    logger.info(f"Processed {len(weekly_data)} weeks for model analysis")
    return weekly_data

def analyze_model_probabilities(weekly_data):
    """Analyze model probabilities for each day"""
    total_weeks = len(weekly_data)
    
    # Count model occurrences for each day
    model_counts = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
    for day in range(1, 6):
        model_counts[day] = {'Upside': 0, 'Downside': 0, 'Inside': 0, 'Outside': 0, 'Undefined': 0}
    
    for week in weekly_data:
        for day_num, model in week['models'].items():
            if model in model_counts[day_num]:
                model_counts[day_num][model] += 1
    
    # Calculate probabilities
    model_probs = {}
    for day in range(1, 6):
        model_probs[day] = {}
        for model in ['Upside', 'Downside', 'Inside', 'Outside', 'Undefined']:
            count = model_counts[day][model]
            prob = (count / total_weeks) * 100
            model_probs[day][model] = prob
    
    logger.info(f"Total weeks analyzed: {total_weeks}")
    logger.info("Model probabilities by day:")
    for day in range(1, 6):
        logger.info(f"  Day {day}:")
        for model in ['Upside', 'Downside', 'Inside', 'Outside', 'Undefined']:
            count = model_counts[day][model]
            prob = model_probs[day][model]
            logger.info(f"    {model}: {prob:.1f}% ({count} occurrences)")
    
    return model_probs, model_counts

def save_model_data(weekly_data):
    """Save the model data for use in the main app"""
    model_data = []
    
    for week in weekly_data:
        week_data = {
            'week_start': week['week_start'],
            'high_day': week['high_day'],
            'low_day': week['low_day'],
            'num_days': week['num_days']
        }
        
        # Add daily data and models
        for day_num in range(1, 6):
            if day_num in week['days']:
                day_data = week['days'][day_num]
                week_data[f'day_{day_num}_open'] = day_data['open']
                week_data[f'day_{day_num}_high'] = day_data['high']
                week_data[f'day_{day_num}_low'] = day_data['low']
                week_data[f'day_{day_num}_close'] = day_data['close']
                week_data[f'day_{day_num}_name'] = day_data['day_name']
                week_data[f'day_{day_num}_model'] = week['models'].get(day_num, 'Undefined')
            else:
                week_data[f'day_{day_num}_open'] = None
                week_data[f'day_{day_num}_high'] = None
                week_data[f'day_{day_num}_low'] = None
                week_data[f'day_{day_num}_close'] = None
                week_data[f'day_{day_num}_name'] = None
                week_data[f'day_{day_num}_model'] = None
        
        model_data.append(week_data)
    
    # Save to CSV
    df_model = pd.DataFrame(model_data)
    df_model.to_csv('data/model_weekly_data.csv', index=False)
    logger.info(f"Saved {len(model_data)} weeks to data/model_weekly_data.csv")
    
    # Also save just the model probabilities summary
    model_summary = []
    for day in range(1, 6):
        day_models = [week['models'].get(day, 'Undefined') for week in weekly_data if day in week['models']]
        model_counts = pd.Series(day_models).value_counts()
        total = len(day_models)
        
        for model in ['Upside', 'Downside', 'Inside', 'Outside', 'Undefined']:
            count = model_counts.get(model, 0)
            if total > 0:
                percentage = (count / total) * 100
            else:
                percentage = 0
            model_summary.append({
                'day': day,
                'model': model,
                'count': count,
                'percentage': percentage
            })
    
    df_summary = pd.DataFrame(model_summary)
    df_summary.to_csv('data/day_model_probabilities.txt', index=False, sep='\t')
    logger.info("Saved model probabilities summary to data/day_model_probabilities.txt")

def main():
    logger.info("Starting model data collection and analysis...")
    
    # Load data
    df = load_historical_data()
    if df is None:
        return
    
    # Assign trading days
    logger.info("Assigning trading days...")
    df = assign_trading_day(df)
    
    # Process into weekly data with models
    logger.info("Processing weekly data with models...")
    weekly_data = get_week_model_data(df)
    
    # Analyze model probabilities
    logger.info("Analyzing model probabilities...")
    model_probs, model_counts = analyze_model_probabilities(weekly_data)
    
    # Save model data
    logger.info("Saving model data...")
    save_model_data(weekly_data)
    
    logger.info("Model analysis complete!")

if __name__ == "__main__":
    main() 