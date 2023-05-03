import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv("poems.csv")

# specify the features and target variable columns
X = df.drop("Poet", axis=1)
y = df["Poet"]

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save the training and testing sets as separate CSV files
X_train.to_csv("train.csv", index=False)
y_train.to_csv("train_labels.csv", index=False)
X_test.to_csv("test.csv", index=False)
y_test.to_csv("test_labels.csv", index=False)
