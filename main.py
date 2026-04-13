import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv(r"C:\Users\Meghana\OneDrive\Documents\Autonomous Vehicle Project\dataset_traffic prediction.csv")

# Convert text to numbers
data = pd.get_dummies(data)

# Fix missing values (only features)
data.iloc[:, :-1] = data.iloc[:, :-1].fillna(data.iloc[:, :-1].mean())

# Fix target column
data["Accident"] = data["Accident"].round().astype(int)

# ------------------ EDA ------------------

# 1. Accident count
plt.figure(num="Accident Count")
sns.countplot(x="Accident", data=data)
plt.title("Accident Count")
plt.show()

# 2. Speed vs Accident
plt.figure(num="Speed vs Accident")
sns.boxplot(x="Accident", y="Speed_Limit", data=data)
plt.title("Speed vs Accident")
plt.show()

# 3. Traffic vs Accident
plt.figure(num="Traffic vs Accident")
sns.boxplot(x="Accident", y="Traffic_Density", data=data)
plt.title("Traffic Density vs Accident")
plt.show()
# ------------------ ML MODEL ------------------

# Features & Target
X = data.drop("Accident", axis=1)
y = data["Accident"]

# Scale features (IMPORTANT improvement)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split (fixed random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved Random Forest
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

from sklearn.metrics import classification_report

print("\nClassification Report:\n", classification_report(y_test, pred))