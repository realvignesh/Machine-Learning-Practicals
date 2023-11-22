# 1. IMPORTING THE LIBRARIES AND LOADING THE DATASET
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

print()
print("1. LOADING THE DATASET".center(60), "\n")
df = pd.read_csv("Datasets\\P7-Insurance-Claims.csv")

print("Example Records From The Dataset")
print(df.head(5))
print("\n")

# ======================================================================================================================


# 2. DATA PREPROCESSING
print("2. DATA PREPROCESSING".center(60), "\n")

print("Checking Missing Values:")
print(df.isnull().sum())
print()

print("Dropping Unnecessary Columns...")
df.drop(['policy_number', 'policy_bind_date', 'policy_csl', 'insured_hobbies', 'incident_date',
         'incident_location', 'auto_model', 'authorities_contacted', '_c39'], axis=1, inplace=True)
print()

print("Converting Categorical Values into Numeric Representation")
processed_data = pd.get_dummies(df, columns=['policy_state', 'insured_sex', 'insured_education_level',
                                             'insured_occupation', 'insured_relationship', 'incident_type',
                                             'collision_type', 'incident_severity', 'incident_state', 'incident_city',
                                             'property_damage', 'police_report_available', 'auto_make', 'auto_year'],
                                dtype=int)
print("Example Records:")
print(processed_data.iloc[:, 3:].head(5), "\n")

# SPLIT THE FEATURES AND THE TARGET
X = processed_data.drop(['fraud_reported'], axis=1)
y = processed_data['fraud_reported']

# ======================================================================================================================


kmeans = KMeans(n_clusters=2, n_init=10)  # CREATE A KMEANS CLUSTERING MODEL
kmeans.fit(X)  # FIT THE MODEL TO THE DATA
y_pred = kmeans.predict(X)  # PREDICT THE CLUSTER LABELS FOR EACH DATA POINT

fraudulent_claims = X[y_pred == 1]  # IDENTIFY THE FRAUDULENT CLAIMS

# VISUALIZE THE RESULTS
plt.figure(figsize=(10, 7))
cluster_labels = y_pred
cluster_colors = ["Green" if label == 0 else "Red" for label in cluster_labels]
plt.scatter(X['months_as_customer'], X['policy_annual_premium'], c=cluster_colors, s=50, alpha=0.7)
plt.xlabel('Months as Customer')
plt.ylabel('Policy Annual Premium')
plt.title('Insurance Fraud Detection using KMeans Clustering')
plt.legend(['Legitimate', 'Fraudulent'], loc='upper left')
plt.show()
