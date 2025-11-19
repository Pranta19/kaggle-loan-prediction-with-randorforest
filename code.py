import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# 1️⃣ Load data
df_train = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\train.csv")
df_test  = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\test.csv")
# 2️⃣ Separate target and test ids
X_train = df_train.drop(columns=['loan_paid_back'])
y_train = df_train['loan_paid_back']

if 'id' in df_test.columns:
    test_ids = df_test['id']
    df_test = df_test.drop(columns=['id'])
else:
    test_ids = df_test.index