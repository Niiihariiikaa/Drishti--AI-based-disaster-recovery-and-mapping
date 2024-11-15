import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Define sample synthetic data
data = {
    'Area': ['Area_1', 'Area_2', 'Area_3', 'Area_4'],
    'Damage_Type': np.random.choice(['No Damage', 'Minor Damage', 'Major Damage'], size=4),
    'Population_Density': np.random.randint(100, 1000, size=4),
    'Accessibility': np.random.choice(['High', 'Moderate', 'Low'], size=4)
}

df = pd.DataFrame(data)

# Define resources based on damage type
def estimate_resources(row):
    if row['Damage_Type'] == 'No Damage':
        return {'water': 0, 'medical': 0, 'shelter': 0}
    elif row['Damage_Type'] == 'Minor Damage':
        return {'water': row['Population_Density'] * 0.5, 'medical': 10, 'shelter': row['Population_Density'] * 0.2}
    elif row['Damage_Type'] == 'Major Damage':
        return {'water': row['Population_Density'] * 1.5, 'medical': 50, 'shelter': row['Population_Density'] * 0.8}

# Apply the function
df['Resources_Needed'] = df.apply(estimate_resources, axis=1)

print(df)

# Encode 'Damage_Type' as the target label
damage_encoder = LabelEncoder()
df['Damage_Type_Encoded'] = damage_encoder.fit_transform(df['Damage_Type'])

# Encode 'Accessibility' for model input
accessibility_encoder = LabelEncoder()
df['Accessibility_Encoded'] = accessibility_encoder.fit_transform(df['Accessibility'])

# Prepare features and target variable
X = df[['Population_Density', 'Accessibility_Encoded']]
y = df['Damage_Type_Encoded']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = clf.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=damage_encoder.classes_))

# Prediction function using the separate encoders
def predict_damage(population_density, accessibility):
    # Encode the accessibility input using the accessibility encoder
    accessibility_encoded = accessibility_encoder.transform([accessibility])[0]
    # Make a prediction
    predicted_damage_type = clf.predict([[population_density, accessibility_encoded]])
    damage_label = damage_encoder.inverse_transform(predicted_damage_type)[0]
    
    # Estimate resources based on the prediction
    estimated_resources = estimate_resources({
        'Damage_Type': damage_label,
        'Population_Density': population_density,
        'Accessibility': accessibility
    })
    return damage_label, estimated_resources

# Example usage
damage, resources = predict_damage(500, 'Moderate')
print(f"Predicted Damage: {damage}\nEstimated Resources: {resources}")
