import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# 1. Load Data (FIXED: Changed read_excel to read_csv and added sep=";")
data = pd.read_csv(r"C:\Users\Student\Downloads\wine+quality\winequality-red.csv", sep=";")

print("Dataset Head:")
print(data.head())

# 2. Inspect Data
print("\nDataset Shape:", data.shape)
print("\nColumn Names:")
print(data.columns)
print("\nDataset information:")
data.info()
print("\nDataset Summary Statistics:")
print(data.describe())

# 3. Check and Handle Missing Values
print("\nMissing values:")
print(data.isnull().sum())

X = data.drop("quality", axis=1)
Y = data["quality"]

mean_imputer = SimpleImputer(strategy="mean")
X_mean = mean_imputer.fit_transform(X)

print("\nMissing values after imputation:")
print(pd.DataFrame(X_mean, columns=X.columns).isnull().sum())

# 4. Handle Duplicate Rows
duplicates = data.duplicated().sum()
print("\nDuplicate Rows =", duplicates)
data = data.drop_duplicates()
print("New Shape =", data.shape)

# 5. Outlier Detection Visualization
plt.figure(figsize=(14, 8))
sns.boxplot(data=data)
plt.xticks(rotation=90)
plt.title("Outlier Detection using Box Plot")
plt.show()

# 6. Remove Outliers using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data_clean=data[~((data<(Q1-1.5*IQR))|
                  (data>(Q3+1.5*IQR))).any(axis=1)]
print("Original Shape: ",data.shape)
print("After Removing Outliers : ",data_clean.shape)
# Fixed outlier filtering logic to handle data frame indexing correctly
data_clean = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\nOriginal shape:", data.shape)
print("After Removing outliers:", data_clean.shape)

# 7. Final Visualization
data_clean.hist(figsize=(16, 12), bins=20)
plt.suptitle("Histogram of all features", fontsize=16)
plt.tight_layout()
plt.show()

