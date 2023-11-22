# 1. IMPORTING LIBRARIES AND LOADING DATASET
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import graphviz
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

print()
print("1. LOADING THE DATASET".center(60), "\n")
df = pd.read_csv('Datasets\\P4-Churn-RawData.csv', encoding='latin-1')

print("Example Records From The Dataset")
print(df.head(5))
print("\n")

# ======================================================================================================================


# 2. DATA PREPROCESSING
print("2. DATA PREPROCESSING".center(60), "\n")

print("CHECKING MISSING VALUES:")
print(df.isnull().sum())
print()

print("LIMITING THE DATA")
limited_data = df.iloc[:, 3:]
print("Columns:")
print(limited_data.columns)
print("\nExample Records:")
print(limited_data.head(5), "\n")

print("CONVERTING CATEGORICAL VARIABLES INTO NUMERIC REPRESENTATION")
processed_data = pd.get_dummies(limited_data, columns=['Geography', 'Gender', 'HasCrCard', 'IsActiveMember'], dtype=int)
print("Example Records:")
print(processed_data.head(5), "\n")

print("SCALING THE COLUMNS")
scale_vars = ['CreditScore', 'EstimatedSalary', 'Balance', 'Age']
scaler = MinMaxScaler()
processed_data[scale_vars] = scaler.fit_transform(processed_data[scale_vars])
print("Example Records:")
print(processed_data.head(5), "\n")

print("SELECTING FEATURES AND LABELS")
X = processed_data.drop('Exited', axis=1).values  # Input features (attributes)
y = processed_data['Exited'].values  # Target vector
print('X shape: {}'.format(np.shape(X)))
print('y shape: {}'.format(np.shape(y)))
print()

print("SPLITTING THE TRAINING AND TESTING DATA")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print('Shape of Train Set:', X_train.shape, y_train.shape)
print('Shape of Test Set:', X_test.shape, y_test.shape)
print("\n")

# ======================================================================================================================


# 3. BUILDING AND IMPLEMENTING THE DECISION TREE CLASSIFIER
print("3. BUILDING AND IMPLEMENTING A DECISION TREE CLASSIFIER".center(60), "\n")

dt = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=1)
dt.fit(X_train, y_train)

dot_data = tree.export_graphviz(dt, out_file=None,
                                feature_names=processed_data.drop('Exited', axis=1).columns,
                                class_names=processed_data['Exited'].unique().astype(str),
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()

prediction = dt.predict(X_test)
df_prediction = pd.DataFrame(prediction)

print("Churn Prediction")
print(df_prediction.head(5))
print("\n")

print("Accuracy: ", accuracy_score(y_test, df_prediction))
print("F1 Score: ", f1_score(y_test, prediction, average='weighted'))

cm = confusion_matrix(y_test, prediction)
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
if dt.classes_ is not None:
    sns.heatmap(cm_norm, xticklabels=dt.classes_, yticklabels=dt.classes_, vmin=0., vmax=1., annot=True,
                annot_kws={'size': 50})
else:
    sns.heatmap(cm_norm, vmin=0., vmax=1.)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
