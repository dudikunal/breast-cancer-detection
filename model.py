import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

class BreastCancerDetectionModel():
    df = pd.DataFrame()
    model = None
    X = pd.Series()
    y = pd.Series()

    def __init__(self, data_path : str):
        self.load_dataframe(data_path)


    def load_dataframe(self, data_path : str):
        data_path = 'dataset/data.csv'
        self.df = pd.read_csv(data_path)

    
    def fit():
        pass

    def predict():
        pass



    


df.head()


df = df.iloc[:, 1:-1]

diagnosis_col = df['diagnosis']
lb = LabelEncoder()
lb.fit(diagnosis_col)
diagnosis_col = lb.transform(diagnosis_col)
df['diagnosis'] = diagnosis_col


X = df.iloc[:, 1:] # Features
y = df['diagnosis'] # Target


print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, " " , y_train.shape, y_test.shape)


gnb = GaussianNB()

gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)



results = {'y' : y_test, 'y_pred' : y_pred}
results = pd.DataFrame(results)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))




report = classification_report(y_test, y_pred, target_names=['Diseased', 'Normal'], output_dict=True)
report = pd.DataFrame(report)



