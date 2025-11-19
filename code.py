import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# 1️⃣ Load data
df_train = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\train.csv")
df_test  = pd.read_csv(r"D:\ml 15 days\kaggle compitition2\test.csv")