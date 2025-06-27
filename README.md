# Scenario Comparison Web App

A Flask web application for analyzing trading data and comparing scenarios for day model analysis.

## Features

- **Scenario Comparison**: Compare high ODR-low RDR vs low ODR-high RDR scenarios
- **Day Model Analysis**: Analyze day models with filtering by role and Week Wed ODR
- **Trading Recommendations**: Get data-driven trading insights and probability analysis
- **Persistent Results**: Results persist until cleared with the Clear All button

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and go to `http://localhost:5001`

## Deployment to Render

### Option 1: Using render.yaml (Recommended)

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" and select "Web Service"
4. Connect your Git repository
5. Render will automatically detect the `render.yaml` configuration
6. Click "Create Web Service"

### Option 2: Manual Configuration

1. Push your code to a Git repository
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" and select "Web Service"
4. Connect your Git repository
5. Configure the service:
   - **Name**: scenario-comparison-web
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
6. Click "Create Web Service"

## Data Files

The app requires the following data files in the `data/` directory:
- `DDR_Predictor.xlsx` - Main scenario data
- `CLhistorical5m.csv` - Historical price data
- `Week Wed ODR.csv` - Week Wednesday ODR data

## Environment Variables

- `PORT` - Port number (automatically set by Render)

## Dependencies

- Flask 2.3.3
- pandas 2.1.1
- openpyxl 3.1.2
- gunicorn 21.2.0
