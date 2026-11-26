import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Automobile MPG data (Auto MPG) from OpenML.
data = fetch_openml(name="autoMpg", version=1, as_frame=True)
df = data.frame.dropna()
target = "mpg"
X = df.drop(columns=[target])
y = df[target].astype(float)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
print("R2:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("Feature coefficients:")
print(pd.Series(model.coef_, index=X.columns).sort_values())
