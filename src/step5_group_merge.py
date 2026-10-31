from src.dataset import make_df, make_depts

df = make_df()
depts = make_depts()

# Grouping
print(df.groupby("city")["salary"].mean())

res = df.groupby("city").agg(
    n=("name", "count"),
    avg=("salary", "mean")
)
print(res)

# Merging
m = df.merge(depts, on="city", how="left")
print(m[["name", "city", "region"]])

inner = df.merge(depts, on="city", how="inner")
print(len(inner), "rows in inner join")

# Applying functions
df["tier"] = df["salary"].apply(
    lambda v: "high" if v >= 60 else "low"
)
print(df[["salary", "tier"]].head(3))

print(df["name"].str.upper().tolist())

print(df[["age", "salary"]].apply(
    lambda c: c.max() - c.min()
))