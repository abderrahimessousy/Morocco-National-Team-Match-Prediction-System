# train_model.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Morocco Football Match Predictor Training...")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n1️⃣ Loading data...")
df = pd.read_csv('data/African-Nations-results.csv')
print(f"   Total matches in dataset: {len(df)}")

# Filter for Morocco matches
morocco_matches = df[(df['home_team'] == 'Morocco') | (df['away_team'] == 'Morocco')].copy()
print(f"   Matches involving Morocco: {len(morocco_matches)}")

# Convert date and sort
morocco_matches['date'] = pd.to_datetime(morocco_matches['date'], format='%m/%d/%Y')
morocco_matches = morocco_matches.sort_values('date').reset_index(drop=True)

# ============================================================================
# STEP 2: CREATE TARGET VARIABLE
# ============================================================================
print("\n2️⃣ Creating target variable...")

def get_result_for_morocco(row):
    if row['home_team'] == 'Morocco':
        if row['home_score'] > row['away_score']:
            return 'Win'
        elif row['home_score'] < row['away_score']:
            return 'Lose'
        else:
            return 'Draw'
    else:  # Morocco is away team
        if row['away_score'] > row['home_score']:
            return 'Win'
        elif row['away_score'] < row['home_score']:
            return 'Lose'
        else:
            return 'Draw'

morocco_matches['result'] = morocco_matches.apply(get_result_for_morocco, axis=1)

# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n3️⃣ Performing Exploratory Data Analysis...")

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 3.1 Basic statistics
result_counts = morocco_matches['result'].value_counts()
print(f"\n   Match Results Distribution:")
print(f"   Wins: {result_counts['Win']} ({result_counts['Win']/len(morocco_matches)*100:.1f}%)")
print(f"   Draws: {result_counts['Draw']} ({result_counts['Draw']/len(morocco_matches)*100:.1f}%)")
print(f"   Losses: {result_counts['Lose']} ({result_counts['Lose']/len(morocco_matches)*100:.1f}%)")

# 3.2 Create location feature
morocco_matches['location'] = morocco_matches['home_team'].apply(
    lambda x: 'Home' if x == 'Morocco' else 'Away'
)

# Home vs Away performance
home_stats = morocco_matches[morocco_matches['location'] == 'Home']
away_stats = morocco_matches[morocco_matches['location'] == 'Away']

print(f"\n   Home Performance:")
print(f"   Total: {len(home_stats)} matches")
print(f"   Win rate: {len(home_stats[home_stats['result']=='Win'])/len(home_stats)*100:.1f}%")

print(f"\n   Away Performance:")
print(f"   Total: {len(away_stats)} matches")
print(f"   Win rate: {len(away_stats[away_stats['result']=='Win'])/len(away_stats)*100:.1f}%")

# 3.3 Create visualization folder
import os
os.makedirs('visualizations', exist_ok=True)

# Plot 1: Results distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart
axes[0].pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#f39c12', '#e74c3c'])
axes[0].set_title('Morocco Match Results Distribution')

# Bar chart
axes[1].bar(result_counts.index, result_counts.values, 
            color=['#2ecc71', '#f39c12', '#e74c3c'])
axes[1].set_title('Number of Wins/Draws/Losses')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('visualizations/results_distribution.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: visualizations/results_distribution.png")

# Plot 2: Results over time
morocco_matches['year'] = morocco_matches['date'].dt.year
results_by_year = morocco_matches.groupby('year')['result'].value_counts().unstack().fillna(0)

plt.figure(figsize=(12, 6))
results_by_year.plot(kind='line', marker='o', linewidth=2)
plt.title('Morocco Results Over Time (2010-2014)')
plt.xlabel('Year')
plt.ylabel('Number of Matches')
plt.grid(True, alpha=0.3)
plt.legend(title='Result')
plt.tight_layout()
plt.savefig('visualizations/results_over_time.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: visualizations/results_over_time.png")

# Plot 3: Results by tournament
plt.figure(figsize=(12, 6))
tournament_counts = morocco_matches['tournament'].value_counts().head(10)
sns.barplot(x=tournament_counts.values, y=tournament_counts.index)
plt.title('Top 10 Tournaments Morocco Played In')
plt.xlabel('Number of Matches')
plt.tight_layout()
plt.savefig('visualizations/tournament_distribution.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: visualizations/tournament_distribution.png")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
print("\n4️⃣ Engineering features...")

def prepare_features(df):
    features_df = df.copy()
    
    # 1. Opponent team
    features_df['opponent'] = features_df.apply(
        lambda x: x['away_team'] if x['home_team'] == 'Morocco' else x['home_team'], 
        axis=1
    )
    
    # 2. Match location (binary: 1 for Home, 0 for Away)
    features_df['is_home'] = features_df['location'].apply(lambda x: 1 if x == 'Home' else 0)
    
    # 3. Tournament type (encoded)
    tournament_encoder = LabelEncoder()
    features_df['tournament_encoded'] = tournament_encoder.fit_transform(features_df['tournament'])
    
    # 4. Opponent strength (frequency as proxy)
    opponent_freq = features_df['opponent'].value_counts().to_dict()
    features_df['opponent_strength'] = features_df['opponent'].map(opponent_freq)
    
    # 5. Year and month features
    features_df['year'] = features_df['date'].dt.year
    features_df['month'] = features_df['date'].dt.month
    
    # 6. Rolling performance features (last 5 matches)
    print("   Calculating rolling statistics...")
    features_df['rolling_win_rate'] = 0.5  # Default
    features_df['rolling_goals_scored'] = 1.0
    features_df['rolling_goals_conceded'] = 1.0
    
    for i in range(len(features_df)):
        if i >= 5:
            last_5 = features_df.iloc[i-5:i]
            
            # Win rate in last 5 matches
            wins = 0
            goals_scored = []
            goals_conceded = []
            
            for _, match in last_5.iterrows():
                if match['home_team'] == 'Morocco':
                    if match['home_score'] > match['away_score']:
                        wins += 1
                    goals_scored.append(match['home_score'])
                    goals_conceded.append(match['away_score'])
                else:
                    if match['away_score'] > match['home_score']:
                        wins += 1
                    goals_scored.append(match['away_score'])
                    goals_conceded.append(match['home_score'])
            
            features_df.at[i, 'rolling_win_rate'] = wins / 5
            features_df.at[i, 'rolling_goals_scored'] = np.mean(goals_scored) if goals_scored else 1.0
            features_df.at[i, 'rolling_goals_conceded'] = np.mean(goals_conceded) if goals_conceded else 1.0
    
    return features_df, tournament_encoder

# Prepare features
features_df, tournament_encoder = prepare_features(morocco_matches)

# Save tournament encoder
joblib.dump(tournament_encoder, 'models/tournament_encoder.pkl')
print("   ✓ Saved tournament encoder")

# ============================================================================
# STEP 5: ENCODE CATEGORICAL VARIABLES
# ============================================================================
print("\n5️⃣ Encoding categorical variables...")

# Encode opponent teams
opponent_encoder = LabelEncoder()
features_df['opponent_encoded'] = opponent_encoder.fit_transform(features_df['opponent'])

# Save opponent encoder
joblib.dump(opponent_encoder, 'models/opponent_encoder.pkl')
print(f"   Encoded {len(opponent_encoder.classes_)} unique opponents")

# Encode target variable
target_encoder = LabelEncoder()
features_df['result_encoded'] = target_encoder.fit_transform(features_df['result'])

# Save target encoder
joblib.dump(target_encoder, 'models/target_encoder.pkl')
print("   Target classes:", dict(zip(target_encoder.classes_, range(len(target_encoder.classes_)))))

# ============================================================================
# STEP 6: SELECT FEATURES AND SPLIT DATA
# ============================================================================
print("\n6️⃣ Preparing train/test split...")

# Define features
feature_cols = [
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

X = features_df[feature_cols]
y = features_df['result_encoded']

print(f"   Features used: {feature_cols}")
print(f"   Feature matrix shape: {X.shape}")

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train)} matches")
print(f"   Test set: {len(X_test)} matches")

# ============================================================================
# STEP 7: TRAIN MODELS
# ============================================================================
print("\n7️⃣ Training machine learning models...")

# 7.1 Logistic Regression
print("\n   Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"   Logistic Regression Accuracy: {accuracy_lr:.3f}")

# 7.2 Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"   Random Forest Accuracy: {accuracy_rf:.3f}")

# ============================================================================
# STEP 8: EVALUATE MODELS
# ============================================================================
print("\n8️⃣ Model evaluation...")

# Detailed reports
print("\n   Logistic Regression Classification Report:")
print("   " + classification_report(y_test, y_pred_lr, target_names=target_encoder.classes_).replace('\n', '\n   '))

print("\n   Random Forest Classification Report:")
print("   " + classification_report(y_test, y_pred_rf, target_names=target_encoder.classes_).replace('\n', '\n   '))

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LR Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_encoder.classes_, 
            yticklabels=target_encoder.classes_,
            ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# RF Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', 
            xticklabels=target_encoder.classes_, 
            yticklabels=target_encoder.classes_,
            ax=axes[1])
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('visualizations/confusion_matrices.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: visualizations/confusion_matrices.png")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=100, bbox_inches='tight')
print("   ✓ Saved: visualizations/feature_importance.png")

print("\n   Top 5 Most Important Features:")
for idx, row in feature_importance.head().iterrows():
    print(f"   {row['feature']}: {row['importance']:.3f}")

# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================
print("\n9️⃣ Saving models...")

# Save Random Forest (best model)
joblib.dump(rf_model, 'models/morocco_match_predictor.pkl')

# Save Logistic Regression
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')

# Create a summary file
with open('models/model_summary.txt', 'w') as f:
    f.write("Morocco Football Match Predictor - Model Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: African-Nations-results.csv\n")
    f.write(f"Total Morocco matches: {len(morocco_matches)}\n")
    f.write(f"Training set size: {len(X_train)}\n")
    f.write(f"Test set size: {len(X_test)}\n\n")
    f.write("Model Performance:\n")
    f.write(f"  Logistic Regression Accuracy: {accuracy_lr:.3f}\n")
    f.write(f"  Random Forest Accuracy: {accuracy_rf:.3f}\n\n")
    f.write("Features used:\n")
    for feature in feature_cols:
        f.write(f"  - {feature}\n")
    f.write("\nTarget classes:\n")
    for i, cls in enumerate(target_encoder.classes_):
        f.write(f"  {i}: {cls}\n")

print("   ✓ Saved: models/morocco_match_predictor.pkl")
print("   ✓ Saved: models/logistic_regression_model.pkl")
print("   ✓ Saved: models/model_summary.txt")

# ============================================================================
# STEP 10: CREATE PREDICTION FUNCTION
# ============================================================================
print("\n🔟 Creating prediction function...")

def predict_match(opponent, location, tournament, year=None, month=None):
    """
    Predict outcome for a Morocco match
    """
    # Load encoders
    opponent_encoder = joblib.load('models/opponent_encoder.pkl')
    tournament_encoder = joblib.load('models/tournament_encoder.pkl')
    target_encoder = joblib.load('models/target_encoder.pkl')
    model = joblib.load('models/morocco_match_predictor.pkl')
    
    # Prepare single match features
    features = np.zeros(len(feature_cols))
    
    # Map features
    features_dict = {
        'is_home': 1 if location == 'Home' else 0,
        'tournament_encoded': tournament_encoder.transform([tournament])[0] if tournament in tournament_encoder.classes_ else 0,
        'opponent_encoded': opponent_encoder.transform([opponent])[0] if opponent in opponent_encoder.classes_ else len(opponent_encoder.classes_)-1,
        'opponent_strength': 10,  # Default
        'year': year if year else datetime.now().year,
        'month': month if month else datetime.now().month,
        'rolling_win_rate': 0.5,  # Default
        'rolling_goals_scored': 1.5,  # Default
        'rolling_goals_conceded': 1.0  # Default
    }
    
    # Create feature array
    feature_array = np.array([features_dict[col] for col in feature_cols]).reshape(1, -1)
    
    # Make prediction
    prediction_proba = model.predict_proba(feature_array)[0]
    prediction = model.predict(feature_array)[0]
    
    result = target_encoder.inverse_transform([prediction])[0]
    
    return {
        'prediction': result,
        'probabilities': {
            'Win': float(prediction_proba[2]),
            'Draw': float(prediction_proba[0]),
            'Lose': float(prediction_proba[1])
        }
    }

# Test the prediction function
print("\n   Testing prediction function with sample matches...")

# Sample predictions
test_matches = [
    {'opponent': 'Algeria', 'location': 'Home', 'tournament': 'African Cup of Nations'},
    {'opponent': 'Spain', 'location': 'Away', 'tournament': 'Friendly'},
    {'opponent': 'Senegal', 'location': 'Home', 'tournament': 'FIFA World Cup qualification'}
]

for i, match in enumerate(test_matches, 1):
    try:
        result = predict_match(**match)
        print(f"\n   Test {i}: Morocco vs {match['opponent']} ({match['location']}, {match['tournament']})")
        print(f"   Predicted: {result['prediction']}")
        print(f"   Probabilities - Win: {result['probabilities']['Win']:.2%}, "
              f"Draw: {result['probabilities']['Draw']:.2%}, "
              f"Lose: {result['probabilities']['Lose']:.2%}")
    except Exception as e:
        print(f"   Error: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("\n📊 Summary:")
print(f"  • Total Morocco matches analyzed: {len(morocco_matches)}")
print(f"  • Win rate: {result_counts['Win']/len(morocco_matches)*100:.1f}%")
print(f"  • Best model accuracy: {accuracy_rf:.3f}")
print(f"  • Models saved in: models/")
print(f"  • Visualizations saved in: visualizations/")
print("\n📁 Files created:")
print("  1. models/morocco_match_predictor.pkl - Main model")
print("  2. models/logistic_regression_model.pkl - Alternative model")
print("  3. models/target_encoder.pkl - Result encoder")
print("  4. models/opponent_encoder.pkl - Team encoder")
print("  5. models/tournament_encoder.pkl - Tournament encoder")
print("  6. visualizations/*.png - Analysis charts")
print("\n🚀 Next step: Run 'python app.py' to start the web interface!")