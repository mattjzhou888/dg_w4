import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

df = pd.read_csv("Salary_Data.csv")

X = df.loc[:, df.columns != "Salary"]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {mean_squared_error(y_test, y_pred)}")
pickle.dump(model, open("model.pickle", "wb"))