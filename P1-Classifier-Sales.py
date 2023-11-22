# 1. IMPORTING LIBRARIES AND LOADING DATASET
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

print()
print("1. LOADING THE DATASET".center(60), "\n")
df = pd.read_csv("Datasets\\P1-Churn-Modelling.csv")

print("Example Records From The Dataset")
print(df.head(5))
print("\n")

# ======================================================================================================================


# 2. DATA PREPROCESSING
print("2. DATA PREPROCESSING".center(60), "\n")

print("Selecting the Label and Features")
print("Chosen Label - 'Exited'")
y = df.iloc[:, 13]
print(y.head(5), "\n")

print("Chosen Features - 'CreditScore' to 'EstimatedSalary'")
X = df.iloc[:, 3:13]
print(X.head(5), "\n")

label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])
print("Encoding Gender Feature")
print(X['Gender'].head(5), "\n")

X['Geography'] = label.fit_transform(X['Geography'])
print("Encoding Geography Feature")
print(X['Geography'].head(5), "\n")

print("Splitting the Training and Testing Data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print('Shape of Train Set:', X_train.shape, y_train.shape)
print('Shape of Test Set:', X_test.shape, y_test.shape)
print("\n")

# ======================================================================================================================


# 3. BUILDING AND IMPLEMENTING A RANDOM FOREST CLASSIFIER
print("3. BUILDING AND IMPLEMENTING A RANDOM FOREST CLASSIFIER".center(60), "\n")

classifier = RandomForestClassifier(n_estimators=100)  # warning 10 to 100
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
df_prediction = pd.DataFrame(prediction)

print("Churn Prediction")
print(df_prediction.head(5))
print("\n")

# ======================================================================================================================


# 4. CLASSIFICATION METRICS
print("4. CLASSIFICATION METRICS".center(60), "\n")

accuracy = accuracy_score(y_test, df_prediction)
print("Model Accuracy: ", accuracy)

cm = confusion_matrix(y_test, prediction)
print("Confusion Matrix:\n", cm)

f1 = f1_score(y_test, prediction, average='weighted')
print("F1 Score: ", f1)
