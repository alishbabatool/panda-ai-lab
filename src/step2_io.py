import pandas as pd
from src.dataset import make_df

df = make_df()

# Save the DataFrame in three formats
df.to_csv("data/data.csv", index=False)
df.to_excel("data/data.xlsx", index=False)
df.to_json("data/data.json", orient="records")

# Read the files back
print(pd.read_csv("data/data.csv").shape)
print(pd.read_excel("data/data.xlsx").shape)
print(pd.read_json("data/data.json").shape)

# Display the first 60 characters of the JSON file
print(open("data/data.json").read()[:60])