import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# import data
train_dataset = pd.read_csv('loan_prediction_dataset_train.csv')
train_dataset.drop('Loan_ID', axis=1, inplace=True)

null_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed',
             'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

# Each null value is replaced by the column's mode.
for col in null_cols:
    train_dataset[col] = train_dataset[col].fillna(
        train_dataset[col].dropna().mode().values[0])

# collecting all numeric columns from train dataset
num = train_dataset.select_dtypes('number').columns.to_list()
train_dataset_num = train_dataset[num]

# collecting all catergorical value columns from train dataset
cat = train_dataset.select_dtypes('object').columns.to_list()
train_dataset_cat = train_dataset[cat]

# convert categorical to numeric

to_convert = {'Male': 1, 'Female': 0,
              'Yes': 1, 'No': 0,
              'Graduate': 1, 'Not Graduate': 0,
              'Urban': 1, 'Semiurban': 2, 'Rural': 3,
              'Y': 1, 'N': 0,
              '0': 0, '1': 1, '2': 2, '3+': 3}


def convert_to_numeric(word):
    return to_convert[word]


train_dataset = train_dataset.applymap(
    lambda x: convert_to_numeric(x) if x in to_convert else x)
train_dataset.head(n=614)

y = train_dataset["Loan_Status"]
X = train_dataset.drop(["Loan_Status"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Accuracy score (remove for prod)
print(accuracy_score(y_pred, y_test))
