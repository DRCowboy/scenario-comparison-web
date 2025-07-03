import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load the necessary data files."""
    try:
        # Load partial model data
        partial_df = pd.read_csv('data/partial_model_data.csv')
        partial_df['date'] = pd.to_datetime(partial_df['date'])
        partial_df['week_start'] = pd.to_datetime(partial_df['week_start'])
        logger.info(f"Partial model data loaded. Shape: {partial_df.shape}")
        
        # Load existing weekly data
        weekly_df = pd.read_csv('data/model_weekly_data.csv')
        weekly_df['week_start'] = pd.to_datetime(weekly_df['week_start'])
        logger.info(f"Weekly data loaded. Shape: {weekly_df.shape}")
        
        # Load main daily data (use CLhistorical5m.csv instead of SPY_2008_2024.csv)
        daily_df = pd.read_csv('data/CLhistorical5m.csv')
        daily_df['time'] = pd.to_datetime(daily_df['time'])
        daily_df['date'] = daily_df['time'].dt.date
        logger.info(f"Main daily data loaded. Shape: {daily_df.shape}")
        return partial_df, weekly_df, daily_df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None

def get_trading_dates_for_week(week_start, daily_df, num_days=5):
    """Get the actual trading dates for a given week."""
    # Convert week_start to date
    week_start_date = week_start.date()
    
    # Get all trading dates from daily data
    trading_dates = sorted(daily_df['date'].unique())
    
    # Find the week_start date in trading dates
    try:
        start_idx = trading_dates.index(week_start_date)
    except ValueError:
        # If week_start is not a trading date, find the next trading date
        start_idx = 0
        for i, date in enumerate(trading_dates):
            if date >= week_start_date:
                start_idx = i
                break
    
    # Get the next num_days trading dates
    week_trading_dates = []
    for i in range(num_days):
        if start_idx + i < len(trading_dates):
            week_trading_dates.append(trading_dates[start_idx + i])
    
    return week_trading_dates

def integrate_partial_models():
    """Integrate partial models into the weekly data structure."""
    partial_df, weekly_df, daily_df = load_data()
    if partial_df is None or weekly_df is None or daily_df is None:
        return
    
    # Create a mapping from (week_start, trading_day) to partial model
    partial_mapping = {}
    for _, row in partial_df.iterrows():
        week_start = row['week_start'].date()
        trading_day = row['trading_day']
        partial_model = row['partial_model']
        partial_mapping[(week_start, trading_day)] = partial_model
    
    # Add partial model columns to weekly data
    weekly_df['day_1_partial_model'] = 'N/A'  # Day 1 has no previous day to compare to
    weekly_df['day_2_partial_model'] = 'N/A'
    weekly_df['day_3_partial_model'] = 'N/A'
    weekly_df['day_4_partial_model'] = 'N/A'
    weekly_df['day_5_partial_model'] = 'N/A'
    
    # Process each week
    for idx, row in weekly_df.iterrows():
        week_start = row['week_start'].date()
        num_days = row['num_days']
        
        # Map partial models to each day
        for day_num in range(2, num_days + 1):  # Start from day 2 since day 1 has no previous day
            # Get the partial model for this day
            key = (week_start, day_num)
            if key in partial_mapping:
                partial_model = partial_mapping[key]
                weekly_df.at[idx, f'day_{day_num}_partial_model'] = partial_model
            else:
                logger.debug(f"No partial model found for week_start {week_start}, day {day_num}")
                weekly_df.at[idx, f'day_{day_num}_partial_model'] = 'N/A'
    
    # Save the updated weekly data
    output_file = 'data/model_weekly_data_with_partial.csv'
    weekly_df.to_csv(output_file, index=False)
    logger.info(f"Updated weekly data saved to {output_file}")
    
    # Print summary statistics
    logger.info("Partial model integration summary:")
    for day_num in range(2, 6):  # Days 2-5
        column = f'day_{day_num}_partial_model'
        model_counts = weekly_df[column].value_counts()
        logger.info(f"Day {day_num} partial models:")
        for model, count in model_counts.items():
            percentage = (count / len(weekly_df)) * 100
            logger.info(f"  {model}: {count} ({percentage:.1f}%)")
    
    return weekly_df

if __name__ == "__main__":
    integrate_partial_models() 