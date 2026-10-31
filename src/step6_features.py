import pandas as pd
from src.dataset import make_df

f = make_df().drop_duplicates().copy()

f["age"] = f["age"].fillna(f["age"].median())

f["salary_z"] = (f.salary - f.salary.mean()) / f.salary.std()

f = pd.get_dummies(f, columns=["city"], prefix="c", dtype=int)

X = f.drop(columns=["name", "salary"]).round(2)

print(X.shape)
print(X.head(3))

X.to_csv("results/features.csv", index=False)