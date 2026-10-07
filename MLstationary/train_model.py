

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


df = pd.read_csv('training_data.csv')

features = df.drop('label', axis=1)
labels = df['label']

#make train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)
#test size is 20% of the dataset, stratify ensures that the class distribution is preserved in both training and testing sets
#random state is set to 0 for reproducibility

#train model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

#Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save the trained model
joblib.dump(model, 'trained_model.pkl')
