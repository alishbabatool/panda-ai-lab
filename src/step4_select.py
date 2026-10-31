from src.dataset import make_df

df = make_df()

# Select columns
print(df[["name", "salary"]].head(2))

# Select a value using loc
print(df.loc[1, "city"])

# Select rows and columns using iloc
print(df.iloc[0:2, 0:2])

# Filter rows
print(df[(df.salary > 50) & (df.city == "Karachi")])

# Sort by salary in descending order
top = df.sort_values("salary", ascending=False)
print(top[["name", "salary"]].head(3))

# Sort by multiple columns
by = df.sort_values(["city", "age"])
print(by[["city", "age"]])