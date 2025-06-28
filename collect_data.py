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

def get_trading_day_boundaries():
    """Define trading day boundaries"""
    return {
        1: {'start': '09:30', 'end': '09:25', 'start_day': 'Tuesday', 'end_day': 'Wednesday'},
        2: {'start': '09:30', 'end': '09:25', 'start_day': 'Wednesday', 'end_day': 'Thursday'},
        3: {'start': '09:30', 'end': '09:25', 'start_day': 'Thursday', 'end_day': 'Friday'},
        4: {'start': '09:30', 'end': '09:30', 'start_day': 'Friday', 'end_day': 'Monday'},
        5: {'start': '09:30', 'end': '09:25', 'start_day': 'Monday', 'end_day': 'Tuesday'}
    }

def assign_trading_day(df):
    """Assign each row to the correct trading day (1-5)"""
    boundaries = get_trading_day_boundaries()
    
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

def get_week_data(df):
    """Process data into weekly structure with proper day boundaries"""
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
            # Find high and low of week
            all_highs = [(day, data['high']) for day, data in daily_data.items()]
            all_lows = [(day, data['low']) for day, data in daily_data.items()]
            
            high_day = max(all_highs, key=lambda x: x[1])[0]
            low_day = min(all_lows, key=lambda x: x[1])[0]
            
            weekly_data.append({
                'week_start': week_start,
                'days': daily_data,
                'high_day': high_day,
                'low_day': low_day,
                'num_days': len(daily_data)
            })
    
    logger.info(f"Processed {len(weekly_data)} weeks")
    return weekly_data

def analyze_high_low_probabilities(weekly_data):
    """Analyze high/low of week probabilities for each day"""
    total_weeks = len(weekly_data)
    
    # Count high and low occurrences for each day
    high_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    low_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for week in weekly_data:
        high_counts[week['high_day']] += 1
        low_counts[week['low_day']] += 1
    
    # Calculate probabilities
    high_probs = {day: (count / total_weeks) * 100 for day, count in high_counts.items()}
    low_probs = {day: (count / total_weeks) * 100 for day, count in low_counts.items()}
    
    logger.info(f"Total weeks analyzed: {total_weeks}")
    logger.info("High of Week probabilities:")
    for day in range(1, 6):
        logger.info(f"  Day {day}: {high_probs[day]:.1f}% ({high_counts[day]} occurrences)")
    
    logger.info("Low of Week probabilities:")
    for day in range(1, 6):
        logger.info(f"  Day {day}: {low_probs[day]:.1f}% ({low_counts[day]} occurrences)")
    
    return high_probs, low_probs, high_counts, low_counts

def answer_specific_question(weekly_data):
    """Answer: What percentage of weeks have high on Day 5 when low is on Day 1?"""
    weeks_with_low_day1 = [week for week in weekly_data if week['low_day'] == 1]
    weeks_with_low_day1_and_high_day5 = [week for week in weeks_with_low_day1 if week['high_day'] == 5]
    
    total_weeks_low_day1 = len(weeks_with_low_day1)
    weeks_matching = len(weeks_with_low_day1_and_high_day5)
    
    if total_weeks_low_day1 > 0:
        percentage = (weeks_matching / total_weeks_low_day1) * 100
        logger.info(f"\nSPECIFIC ANSWER:")
        logger.info(f"Weeks with low on Day 1: {total_weeks_low_day1}")
        logger.info(f"Of those, weeks with high on Day 5: {weeks_matching}")
        logger.info(f"Percentage: {percentage:.1f}%")
        return percentage
    else:
        logger.info("No weeks found with low on Day 1")
        return 0

def main():
    logger.info("Starting data collection and analysis...")
    
    # Load data
    df = load_historical_data()
    if df is None:
        return
    
    # Assign trading days
    logger.info("Assigning trading days...")
    df = assign_trading_day(df)
    
    # Process into weekly data
    logger.info("Processing weekly data...")
    weekly_data = get_week_data(df)
    
    # Analyze probabilities
    logger.info("Analyzing high/low probabilities...")
    high_probs, low_probs, high_counts, low_counts = analyze_high_low_probabilities(weekly_data)
    
    # Answer specific question
    answer_specific_question(weekly_data)
    
    # Save processed data
    logger.info("Saving processed data...")
    save_processed_data(weekly_data)
    
    logger.info("Analysis complete!")

def save_processed_data(weekly_data):
    """Save the processed weekly data for use in the main app"""
    processed_data = []
    
    for week in weekly_data:
        week_data = {
            'week_start': week['week_start'],
            'high_day': week['high_day'],
            'low_day': week['low_day'],
            'num_days': week['num_days']
        }
        
        # Add daily data
        for day_num in range(1, 6):
            if day_num in week['days']:
                day_data = week['days'][day_num]
                week_data[f'day_{day_num}_open'] = day_data['open']
                week_data[f'day_{day_num}_high'] = day_data['high']
                week_data[f'day_{day_num}_low'] = day_data['low']
                week_data[f'day_{day_num}_close'] = day_data['close']
                week_data[f'day_{day_num}_name'] = day_data['day_name']
            else:
                week_data[f'day_{day_num}_open'] = None
                week_data[f'day_{day_num}_high'] = None
                week_data[f'day_{day_num}_low'] = None
                week_data[f'day_{day_num}_close'] = None
                week_data[f'day_{day_num}_name'] = None
        
        processed_data.append(week_data)
    
    # Save to CSV
    df_processed = pd.DataFrame(processed_data)
    df_processed.to_csv('data/processed_weekly_data.csv', index=False)
    logger.info(f"Saved {len(processed_data)} weeks to data/processed_weekly_data.csv")

if __name__ == "__main__":
    main() 