import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Math square root:", math.sqrt(25))
arr = np.array([10, 20, 30, 40, 50])
print("NumPy mean:", np.mean(arr))
print("NumPy standard deviation:", np.std(arr))
print("SciPy mode:", stats.mode(arr, keepdims=True))

df = pd.DataFrame({"Marks": arr, "Hours": [1, 2, 3, 4, 5]})
print(df)
print(df.describe())

df.plot(x="Hours", y="Marks", kind="line", marker="o")
plt.title("Marks vs Study Hours")
plt.show()

sns.histplot(data=df, x="Marks")
plt.title("Marks Distribution")
plt.show()
