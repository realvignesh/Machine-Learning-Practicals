# IMPORTING THE LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

print()
print("1. LOADING THE DATASET AND PREPROCESSING".center(60), "\n")

# LOADING THE DATASET
dataset = pd.read_csv("Datasets\\P5-Iris.csv")
print("Example Records From The Dataset")
print(dataset.head(5))
print()

# DATA PREPROCESSING
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

# ======================================================================================================================


print("2. PREDICTIONS".center(60), "\n")

predictions = classifier.predict(X_test)

for i in range(0, 5):
    print(f"{X_test[i]} - {predictions[i]}")
print()

# ======================================================================================================================


print("3. EVALUATION".center(60), "\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))
