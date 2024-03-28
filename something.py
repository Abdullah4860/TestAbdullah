# -*- coding: utf-8 -*-
"""
Reorganized Flask application script.
"""
import numpy as np

from flask import Flask, request, jsonify
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX



# Initialize Flask app
app = Flask(__name__)

# Load and preprocess sales data
df = pd.read_csv("sales_performance_data.csv")
df['dated'] = pd.to_datetime(df['dated'])
df.sort_values('dated', inplace=True)

# Function to calculate benchmarks for the sales team
def calculate_benchmarks(df):
    benchmarks = {}
    metrics = ['lead_taken', 'tours_booked', 'applications', 'revenue_confirmed', 'tours_per_lead', 'apps_per_tour', 'apps_per_lead', 'revenue_runrate']
    call_metrics = ['mon_call', 'tue_call', 'wed_call', 'thur_call', 'fri_call', 'sat_call', 'sun_call']
    message_metrics = ['mon_text', 'tue_text', 'wed_text', 'thur_text', 'fri_text', 'sat_text', 'sun_text']

    for metric in metrics + call_metrics + message_metrics:
        benchmarks[metric] = {
            'average': df[metric].mean(),
            'median': df[metric].median()
        }
    return benchmarks

# Function to analyze individual sales representative performance

def analyze_rep_performance(rep_id, df, benchmarks):
    # Filter data for the specific sales representative
    rep_data = df[df['employee_id'] == rep_id]

    # Extracting and calculating individual and team performance metrics
    performance_comparison = {}
    for metric, values in benchmarks.items():
        rep_metric_value = rep_data[metric].mean()
        # Convert NumPy types to Python types
        performance_comparison[metric] = {
            'individual': float(rep_metric_value) if pd.notnull(rep_metric_value) else None,
            'team_average': float(values['average']),
            'team_median': float(values['median']),
            'above_average': bool(rep_metric_value > values['average'])  # Convert numpy bool to Python bool
        }

    # Call and Message Analysis
    call_analysis = {metric: int(rep_data[metric].sum()) for metric in benchmarks.keys() if 'call' in metric}
    message_analysis = {metric: int(rep_data[metric].sum()) for metric in benchmarks.keys() if 'text' in metric}

    # Convert any other NumPy types that may not be serializable
    feedback = {
        'performance_comparison': {k: {kk: (int(vv) if isinstance(vv, np.integer) else vv) for kk, vv in v.items()} for k, v in performance_comparison.items()},
        'call_summary': {k: int(v) for k, v in call_analysis.items()},
        'message_summary': {k: int(v) for k, v in message_analysis.items()},
    }
    return feedback



# Function to convert analysis results to text
def dict_to_text(data):
    performance_parts = []
    for category, metrics in data['performance_comparison'].items():
        above_or_below = "above" if metrics['above_average'] else "below"
        individual_value = metrics['individual'] if metrics['individual'] is not None else 'N/A'
        team_average_value = metrics['team_average'] if metrics['team_average'] is not None else 'N/A'
        part = f"In {category}, the individual's performance was {above_or_below} the team average with a value of {individual_value} compared to the team's average of {team_average_value}."
        performance_parts.append(part)

    call_parts = [f"{day}: {count}" for day, count in data['call_summary'].items()]
    message_parts = [f"{day}: {count}" for day, count in data['message_summary'].items()]

    text = " ".join(performance_parts) + " Weekly call totals are " + ", ".join(call_parts) + ". Weekly message totals are " + ", ".join(message_parts) + "."

    # Assuming that the client object's predict method can handle the text properly
    result = client.predict(
        text,  # str in 'Input Text' Textbox component
        api_name="/predict"
    )
    return result


# Function to generate a performance summary for the sales team
def generate_performance_summary(benchmarks):
    # Extract key metrics for summary
    leads_taken_avg = benchmarks['lead_taken']['average']
    tours_booked_avg = benchmarks['tours_booked']['average']
    revenue_confirmed_avg = benchmarks['revenue_confirmed']['average']
    revenue_runrate_avg = benchmarks['revenue_runrate']['average']

    # Calls and Texts summary
    call_avg = sum(benchmarks[metric]['average'] for metric in benchmarks if 'call' in metric) / 7  # Average over 7 days
    text_avg = sum(benchmarks[metric]['average'] for metric in benchmarks if 'text' in metric) / 7  # Average over 7 days

    # Constructing the summary sentence
    summary = (
        f"The sales team has an average of {leads_taken_avg:.1f} leads taken, {tours_booked_avg:.1f} tours booked, "
        f"and a confirmed revenue of ${revenue_confirmed_avg:.2f} indicating robust sales activity. "
        f"The revenue runrate stands at an average of ${revenue_runrate_avg:.2f}, showcasing potential growth. "
        f"On average, the team makes {call_avg:.1f} calls and sends {text_avg:.1f} texts per day, highlighting active communication."
    )

    return summary

# Function to forecast sales metric
def forecast_sales_metric(df, metric_name='revenue_confirmed', periods=12):
    """
    Forecast a specific sales metric using the ARIMA model.

    Parameters:
    - df: DataFrame containing your sales data.
    - metric_name: String name of the metric to forecast (e.g., 'revenue_confirmed').
    - periods: Number of periods to forecast into the future.

    Returns:
    - forecast: The forecasted values for the specified number of future periods.
    """

    # Ensure 'dated' is in datetime format and sort the DataFrame by date
    df['dated'] = pd.to_datetime(df['dated'])
    df.sort_values('dated', inplace=True)

    # Extract the time series
    ts = df.groupby('dated')[metric_name].sum()  # Assuming you want to sum the metric by date

    # Fit the ARIMA model
    auto_model = auto_arima(ts, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore")

    # Fit the SARIMAX model
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)

    # Forecast
    forecast = model_fit.forecast(steps=periods)

    return forecast



# Flask route definitions
@app.route('/')
def home():
    return "Flask API is running!"

@app.route('/api/rep_performance', methods=['GET'])
def get_rep_performance():
    benchmarks = calculate_benchmarks(df)

    rep_id = request.args.get('rep_id')
    if not rep_id:
        return jsonify({'error': 'Missing rep_id parameter'}), 400

    try:
        rep_id = int(rep_id)  # Ensure that rep_id is an integer
        feedback = analyze_rep_performance(rep_id, df, benchmarks)
        
        # Convert all numpy int64 types to Python int
        for category, metrics in feedback['performance_comparison'].items():
            metrics['individual'] = int(metrics['individual']) if pd.notnull(metrics['individual']) else None
            metrics['team_average'] = int(metrics['team_average']) if pd.notnull(metrics['team_average']) else None
            metrics['team_median'] = int(metrics['team_median']) if pd.notnull(metrics['team_median']) else None

        for key in feedback['call_summary'].keys():
            feedback['call_summary'][key] = int(feedback['call_summary'][key])

        for key in feedback['message_summary'].keys():
            feedback['message_summary'][key] = int(feedback['message_summary'][key])

        return jsonify(feedback)

    except ValueError:
        return jsonify({'error': 'Invalid rep_id parameter'}), 400
    except KeyError:  # If the rep_id does not exist in the data
        return jsonify({'error': 'Sales representative not found'}), 404




@app.route('/api/rep_performance_summary2', methods=['GET'])
def get_rep_performance_summary():
    benchmarks = calculate_benchmarks(df)
    print("0000000000000000000000000000000000000000000000000000000000000000000000000")
    rep_id = request.args.get('rep_id')
    if not rep_id:
        return jsonify({'error': 'Missing rep_id parameter'}), 400

    feedback = analyze_rep_performance(rep_id, df, benchmarks)
    if not feedback['performance_comparison']:  # If the rep_id does not exist in the data
        return jsonify({'error': 'Sales representative not found'}), 404

    summary_text = dict_to_text(feedback)
    return jsonify({'summary': summary_text})

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks():
    benchmarks = calculate_benchmarks(df)
    return jsonify(benchmarks)

@app.route('/api/team_performance', methods=['GET'])
def team_performance_summary():
    benchmarks = calculate_benchmarks(df)
    summary = generate_performance_summary(benchmarks)
    return jsonify({'summary': summary})
@app.route('/api/forecast', methods=['GET'])
def forecast_summary():
 
    summary = forecast_sales_metric(df)
    forecast_list = summary.tolist()

    return jsonify({'forecast': forecast_list})
# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5001)