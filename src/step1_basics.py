
from src.dataset import make_df

df = make_df()
s = df["age"]

print(df)
print(type(s).__name__, s.shape)
