# app.py
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load models and encoders
try:
    model = joblib.load('models/morocco_match_predictor.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    opponent_encoder = joblib.load('models/opponent_encoder.pkl')
    tournament_encoder = joblib.load('models/tournament_encoder.pkl')
    print("✅ Models loaded successfully!")
except:
    print("❌ Error loading models. Please run train_model.py first!")
    exit(1)

# Feature columns (must match training)
FEATURE_COLS = [
    'is_home',
    'tournament_encoded',
    'opponent_encoded',
    'opponent_strength',
    'year',
    'month',
    'rolling_win_rate',
    'rolling_goals_scored',
    'rolling_goals_conceded'
]

# Sample opponents from dataset
OPPONENTS = sorted([
    'Algeria', 'Tunisia', 'Egypt', 'Senegal', 'Ivory Coast',
    'Cameroon', 'Ghana', 'Nigeria', 'South Africa', 'Zambia',
    'Mali', 'Burkina Faso', 'DR Congo', 'Gabon', 'Guinea',
    'Angola', 'Togo', 'Benin', 'Congo', 'Libya', 'Ethiopia',
    'Sudan', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Burundi'
])

TOURNAMENTS = [
    'Friendly',
    'African Cup of Nations',
    'FIFA World Cup',
    'African Cup of Nations qualification',
    'FIFA World Cup qualification'
]

@app.route('/')
def home():
    return render_template('index.html', opponents=OPPONENTS, tournaments=TOURNAMENTS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Get form data
        opponent = data.get('opponent', 'Algeria')
        location = data.get('location', 'Home')
        tournament = data.get('tournament', 'Friendly')
        year = data.get('year', datetime.now().year)
        month = data.get('month', datetime.now().month)
        
        # Prepare features
        features = {}
        
        # Binary features
        features['is_home'] = 1 if location == 'Home' else 0
        
        # Encode tournament
        try:
            features['tournament_encoded'] = tournament_encoder.transform([tournament])[0]
        except:
            features['tournament_encoded'] = 0  # Default to Friendly
        
        # Encode opponent
        try:
            features['opponent_encoded'] = opponent_encoder.transform([opponent])[0]
        except:
            # If opponent not in training, use most common
            features['opponent_encoded'] = opponent_encoder.transform(['Algeria'])[0]
        
        # Other features (using defaults)
        features['opponent_strength'] = 10
        features['year'] = int(year)
        features['month'] = int(month)
        features['rolling_win_rate'] = 0.5
        features['rolling_goals_scored'] = 1.5
        features['rolling_goals_conceded'] = 1.0
        
        # Create feature array in correct order
        feature_array = np.array([features[col] for col in FEATURE_COLS]).reshape(1, -1)
        
        # Make prediction
        prediction_proba = model.predict_proba(feature_array)[0]
        prediction = model.predict(feature_array)[0]
        
        # Decode result
        result = target_encoder.inverse_transform([prediction])[0]
        
        # Format probabilities
        probabilities = {}
        for i, class_name in enumerate(target_encoder.classes_):
            probabilities[class_name] = float(prediction_proba[i])
        
        return jsonify({
            'success': True,
            'prediction': result,
            'probabilities': probabilities,
            'match_details': {
                'opponent': opponent,
                'location': location,
                'tournament': tournament,
                'year': year,
                'month': month
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/model_info')
def model_info():
    """Get model information"""
    try:
        accuracy = 0.75  # You can update this with actual accuracy
        return jsonify({
            'name': 'Random Forest Classifier',
            'accuracy': accuracy,
            'features': FEATURE_COLS,
            'opponents': OPPONENTS,
            'tournaments': TOURNAMENTS
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 Starting Flask server...")
    print("📱 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)