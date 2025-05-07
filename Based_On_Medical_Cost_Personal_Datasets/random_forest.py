import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

#load the dataset
data = pd.read_csv('dataset/insurance.csv')

#Encode categorical variables
encoder = LabelEncoder()
data["sex"] = encoder.fit_transform(data["sex"])
data["smoker"] = encoder.fit_transform(data["smoker"])
data["region"] = encoder.fit_transform(data["region"])

#create a binary target variable for fraud detection
# Assuming higher charges indicate possible fraud . Here we label a data point as fraud (1) if the charges are in the top 25% of all insurance charges . otherwise, we label it as not fraud (0)
data["fraud"] = data["charges"].apply(lambda x: 1 if x > data["charges"].quantile(0.75) else 0)

# Prepare feature matrix (X) and target vector (y)
X = data.drop(["charges", "fraud"], axis=1)
y = data["fraud"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# print("Model Accuracy:", accuracy)
# print("\nConfusion Matrix:\n", conf_matrix)
# print("\nClassification Report:\n", class_report)


#------------------------------------------------#
# Load input from user or csv
input_df = pd.read_csv('dataset/Input.csv')
