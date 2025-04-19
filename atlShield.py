
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for compatibility

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.patches as mpatches

app = Flask(__name__)

# ---------- Load and Preprocess Data (Run once on startup) ----------
df = pd.read_csv('updated_patrol_types.csv', low_memory=False)
df['OccurredFromDate'] = pd.to_datetime(df['OccurredFromDate'])
df['Hour'] = df['OccurredFromDate'].dt.hour
df['Month'] = df['OccurredFromDate'].dt.month

# Encode neighborhoods
neighborhoods = df['NhoodName'].unique().tolist()
neighborhood_to_int = {name: idx for idx, name in enumerate(sorted(neighborhoods))}
int_to_neighborhood = {v: k for k, v in neighborhood_to_int.items()}
df['NhoodEncoded'] = df['NhoodName'].map(neighborhood_to_int)

# Encode patrol type labels
patrol_labels = df['Patrol Type'].dropna().unique().tolist()
patrol_label_map = {v: str(v) for v in patrol_labels}

# ---------- Train hourly crime prediction model ----------
df_grouped = df.groupby(['NhoodEncoded', 'Month', 'Hour']).size().reset_index(name='CrimeCount')
X = df_grouped[['NhoodEncoded', 'Month', 'Hour']]
y = df_grouped['CrimeCount']

crime_model = RandomForestRegressor(n_estimators=100, random_state=42)
crime_model.fit(X, y)

# ---------- Train patrol type model ----------
df_patrol = df.dropna(subset=['Patrol Type'])
df_patrol = df_patrol.groupby(['NhoodEncoded', 'Month', 'Hour'])['Patrol Type'].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]).reset_index()
X_patrol = df_patrol[['NhoodEncoded', 'Month', 'Hour']]
y_patrol = df_patrol['Patrol Type']

patrol_model = RandomForestClassifier(n_estimators=100, random_state=42)
patrol_model.fit(X_patrol, y_patrol)

joblib.dump(crime_model, 'hourly_crime_model.pkl')
joblib.dump(patrol_model, 'patrol_model.pkl')

@app.route('/predict_hourly_crime', methods=['POST'])
def predict_hourly_crime():
    data = request.json
    neighborhood = data.get('neighborhood')
    month = int(data.get('month'))

    if neighborhood not in neighborhood_to_int:
        return jsonify({'error': 'Neighborhood not found'}), 400

    nhood_id = neighborhood_to_int[neighborhood]

    input_data = pd.DataFrame({
        'NhoodEncoded': [nhood_id]*24,
        'Month': [month]*24,
        'Hour': list(range(24))
    })

    crime_predictions = crime_model.predict(input_data)
    patrol_predictions = patrol_model.predict(input_data)

    # Save graph
    plt.figure(figsize=(12, 5))
    hours = list(range(24))
    norm_preds = np.array(crime_predictions) / max(crime_predictions)
    colors = plt.cm.RdYlGn_r(norm_preds)

    plt.bar(hours, crime_predictions, color=colors)
    plt.xticks(hours)
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Predicted Crimes per Hour")
    plt.title(f"Hourly Crime Forecast for {neighborhood} (Month {month})")
    red_patch = mpatches.Patch(color='red', label='High Crime')
    yellow_patch = mpatches.Patch(color='yellow', label='Moderate Crime')
    green_patch = mpatches.Patch(color='green', label='Low Crime')
    plt.legend(handles=[green_patch, yellow_patch, red_patch], title='Crime Intensity')
    plt.tight_layout()
    plt.savefig(f"hourly_crime_forecast_{neighborhood}_{month}.png")
    plt.close()

    hourly_rows = [
        {
            "hour": f"{hour:02d}:00",
            "predicted_crimes": int(math.ceil(c_pred/30)),   # dividing by 30, to get a daily average, rather than an aggregation!
            "suggested_patrol": patrol_label_map.get(p_pred, str(p_pred))
        }
        for hour, c_pred, p_pred in zip(range(24), crime_predictions, patrol_predictions)
    ]

    return jsonify(hourly_rows)

@app.route('/average_daily_crime', methods=['POST'])
def average_daily_crime():
    data = request.json
    neighborhood = data.get('neighborhood')
    month = int(data.get('month'))

    if neighborhood not in neighborhood_to_int:
        return jsonify({'error': 'Neighborhood not found'}), 400

    nhood_id = neighborhood_to_int[neighborhood]
    df_filtered = df[(df['NhoodEncoded'] == nhood_id) & (df['Month'] == month)]

    if df_filtered.empty:
        return jsonify({'error': 'No data for given neighborhood and month'}), 404

    daily_counts = df_filtered.groupby(df_filtered['OccurredFromDate'].dt.date).size()
    avg_daily = round(daily_counts.mean(), 2)

    return jsonify({
        'neighborhood': neighborhood,
        'month': month,
        'average_daily_crime_count': avg_daily
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=False, port=5000)
