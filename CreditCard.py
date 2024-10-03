import pandas as pd
from google.colab import files

# Upload the file
uploaded = files.upload()

# Load the dataset
credit_df = pd.read_csv('credit_card.csv')

# Check the columns and the first few rows
print(credit_df.columns)
print(credit_df.head())

# Convert categorical variables using one-hot encoding
credit_df = pd.get_dummies(credit_df, drop_first=True)

# Prepare features and target variable
X = credit_df.drop('CreditScore', axis=1)  # Ensure 'Class' is the correct name
y = credit_df['CreditScore']  # Ensure 'Class' is the correct name

# Check class distribution
print("Class distribution before splitting:")
print(y.value_counts())

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution after splitting
print("Class distribution in training set:")
print(y_train.value_counts())
print("Class distribution in test set:")
print(y_test.value_counts())

# Handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE only if there's more than one instance of each class in training set
if (y_train.value_counts() > 1).all():
    X_train_re, y_train_re = smote.fit_resample(X_train, y_train)
else:
    print("Not enough instances to apply SMOTE. Proceeding with original training set.")
    X_train_re, y_train_re = X_train, y_train

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_re_scaled = scaler.fit_transform(X_train_re)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_re_scaled, y_train_re)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state=42)
logistic_classifier.fit(X_train_re_scaled, y_train_re)

# Predictions
y_pred_rf = rf_classifier.predict(X_test_scaled)
y_pred_lr = logistic_classifier.predict(X_test_scaled)

# Evaluation
# Evaluation
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

print("Random Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf, average='weighted'))  # Use 'weighted' for multiclass
print("F1-Score:", f1_score(y_test, y_pred_rf, average='weighted'))  # Use 'weighted' for multiclass
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

print("Logistic Regression Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr, average='weighted'))  # Use 'weighted' for multiclass
print("F1-Score:", f1_score(y_test, y_pred_lr, average='weighted'))  # Use 'weighted' for multiclass
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
