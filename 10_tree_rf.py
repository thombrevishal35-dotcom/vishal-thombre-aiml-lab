from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Heart disease dataset; fetched from OpenML.
data = fetch_openml(name="heart-statlog", version=1, as_frame=True)
df = data.frame.dropna()
target = data.target.name
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
print("Decision Tree accuracy:", accuracy_score(y_test, tree.predict(X_test)))
print("Random Forest accuracy:", accuracy_score(y_test, forest.predict(X_test)))
print("Tree training accuracy:", tree.score(X_train, y_train))
print("Tree testing accuracy:", tree.score(X_test, y_test))
print("Random Forest training accuracy:", forest.score(X_train, y_train))
print("Random Forest testing accuracy:", forest.score(X_test, y_test))
print("Random Forest feature importance:")
print(pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
