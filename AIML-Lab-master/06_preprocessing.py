import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Example dataset; replace with a downloaded CSV if required.
df = pd.DataFrame({
    "Age": [20, 25, None, 30, 35],
    "City": ["Pune", "Mumbai", "Pune", None, "Nashik"],
    "Score": [50, 60, 70, None, 90]
})
print("Original data:\n", df)

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Score"] = df["Score"].fillna(df["Score"].mean())
df["City"] = df["City"].fillna(df["City"].mode()[0])

df["City_encoded"] = LabelEncoder().fit_transform(df["City"])
df[["Age", "Score"]] = MinMaxScaler().fit_transform(df[["Age", "Score"]])
print("\nPreprocessed data:\n", df)
