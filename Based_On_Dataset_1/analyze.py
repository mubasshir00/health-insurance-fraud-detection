
#yes 0 = a clean claim and 1= a fraudulent claim
# #loads pandas for handing tabular data (CSV files)
import pandas as pd
#import function to split dataset into training and test sets
from sklearn.model_selection import train_test_split
#loads RandomForestClassifier for classification
from sklearn.ensemble import RandomForestClassifier
#Imports tools to evaluate your model’s predictions.
from sklearn.metrics import classification_report, confusion_matrix
#Helps convert text labels to numbers, which ML models require.
from sklearn.preprocessing import LabelEncoder

#load dataset
df = pd.read_csv('dataset/medical_aid_claims.csv')

#drop columns not useful for prediction
df.drop(columns=['member-name', 'email', 'patient_name', 'patient_suffix', 'patient_dob'],inplace=True)

# Step 3: Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
# le = LabelEncoder()
encoders = {}  # Store encoders for reuse
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # Save the encoder for this column

# Step 4: Define X and y
X = df.drop('label', axis=1) #Input data except the label
y = df['label'] #Label column what we try to predict


# 80% for training (X_train , y_train) and 20% for testing (X_test, y_test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------------------------#
# Load input from user or csv

input_df = pd.read_csv('dataset/input.csv')
input_df.drop(columns=['member-name', 'email', 'patient_name', 'patient_suffix', 'patient_dob'], inplace=True)

# Encode sample data using saved encoders
for col in categorical_cols:
    if col in input_df:
        le = encoders[col]
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Align columns
input_df = input_df[X.columns]  # Ensure column order matches training

# Predict on sample input
preds = model.predict(input_df)
input_df['Prediction'] = ['Fraudulent Claim 🚨' if p == 1 else 'Clean Claim ✅' for p in preds]


print("\n--- Sample Predictions ---\n")
print(input_df[['Prediction']])