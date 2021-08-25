import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# import data and remove ID column
df = pd.read_csv('loan_prediction_dataset.csv')
df.drop('Loan_ID', axis=1, inplace=True)

null_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed',
             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Each null value is replaced by the column's mode.
for col in null_cols:
    df[col] = df[col].fillna(
        df[col].dropna().mode().values[0])

# collecting all numeric columns from train dataset
num = df.select_dtypes('number').columns.to_list()
df_num = df[num]

# collecting all catergorical value columns from train dataset
cat = df.select_dtypes('object').columns.to_list()
df_cat = df[cat]

# conversions from categorical to numeric

to_convert = {'Male': 1, 'Female': 0,
              'Yes': 1, 'No': 0,
              'Graduate': 1, 'Not Graduate': 0,
              'Urban': 1, 'Semiurban': 2, 'Rural': 3,
              'Y': 1, 'N': 0,
              '0': 0, '1': 1, '2': 2, '3+': 3}


def convert_to_numeric(word):
    return to_convert[word]

# conversion from categorical to numeric


df = df.applymap(
    lambda x: convert_to_numeric(x) if x in to_convert else x)

# Preparing training and testing sets
y = df["Loan_Status"]
X = df.drop(["Loan_Status"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model fitting with decision trees

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model to pkl
pickle.dump(model, open("model.pkl", "wb"))