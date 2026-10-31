import pandas as pd
# ===== Step 1: Series and DataFrames =====


def t1_1_weather_series(df):
    """Return the temp_c column as a Series whose index is the day names."""
    return df.set_index("day")["temp_c"]


def t1_2_hottest_day(df):
    """Return the name of the day with the highest temp_c."""
    hottest_index = df["temp_c"].idxmax()
    return df.loc[hottest_index, "day"]


def t1_3_build_frame():
    """Build a DataFrame from a dict."""
    return pd.DataFrame({
        "item": ["pen", "book", "bag"],
        "price": [10, 120, 800]
    })


def t1_4_series_vs_frame(df):
    """Return the type names of single and double bracket selections."""
    a = type(df["temp_c"]).__name__
    b = type(df[["temp_c"]]).__name__
    return (a, b)


def t2_1_csv_roundtrip(df, path):
    """Save to CSV and return the shape after reading it back."""
    df.to_csv(path, index=False)
    result = pd.read_csv(path)
    return result.shape


def t2_2_json_columns(df, path):
    """Save to JSON records and return the column names."""
    df.to_json(path, orient="records")
    result = pd.read_json(path)
    return result.columns.tolist()


def t2_3_excel_roundtrip(df, path):
    """Save to Excel and return the shape after reading it back."""
    df.to_excel(path, index=False)
    result = pd.read_excel(path)
    return result.shape


def t2_4_index_trap(df, path):
    """Save df to CSV with the default index and return the number of columns."""
    df.to_csv(path)
    result = pd.read_csv(path)
    return result.shape[1]


def t3_1_count_missing(df):
    """Return missing-value counts for each column."""
    return df.isna().sum().to_dict()


def t3_2_math_numeric(df):
    """Convert math marks to numeric values."""
    return pd.to_numeric(df["math"], errors="coerce").tolist()


def t3_3_remove_duplicates(df):
    """Remove duplicate rows and return the new shape."""
    return df.drop_duplicates().shape


def t3_4_fill_english_median(df):
    """Remove duplicates and fill missing English marks with the median."""
    clean = df.drop_duplicates().copy()
    clean["english"] = clean["english"].fillna(clean["english"].median())
    return clean["english"].tolist()


def t3_5_rows_after_dropna(df):
    """Convert math to numeric, drop incomplete rows, and return row count."""
    clean = df.drop_duplicates().copy()
    clean["math"] = pd.to_numeric(clean["math"], errors="coerce")
    clean = clean.dropna()
    return len(clean)


def t3_6_section_category(df):
    """Convert section to category and return its dtype name."""
    clean = df.copy()
    clean["section"] = clean["section"].astype("category")
    return clean["section"].dtype.name


def t3_7_clean_students(df):
    """Clean student data by removing duplicates and filling missing marks."""
    clean = df.drop_duplicates().copy()
    clean["math"] = pd.to_numeric(clean["math"], errors="coerce")
    clean["math"] = clean["math"].fillna(clean["math"].mean())
    clean["english"] = clean["english"].fillna(clean["english"].median())
    return clean


def t4_1_cheap_products(df):
    """Return product names with price below 50."""
    return df.loc[df["price"] < 50, "product"].tolist()


def t4_2_accessories_in_stock(df):
    """Return Accessories product names that are in stock."""
    mask = (df["category"] == "Accessories") & (df["stock"] > 0)
    return df.loc[mask, "product"].tolist()


def t4_3_top_two_by_price(df):
    """Return names of the two most expensive products."""
    result = df.sort_values("price", ascending=False).head(2)
    return result["product"].tolist()


def t4_4_loc_price(df):
    """Return the price at index label 3 using loc."""
    return df.loc[3, "price"]


def t4_5_iloc_block(df):
    """Return the first 2 rows and first 2 columns as a list of lists."""
    return df.iloc[0:2, 0:2].values.tolist()


def t4_6_low_stock_or_premium(df):
    """Return products with stock below 6 OR price above 500."""
    mask = (df["stock"] < 6) | (df["price"] > 500)
    return df.loc[mask, "product"].tolist()


def t4_7_sort_two_keys(df):
    """Sort category A to Z and price highest first."""
    result = df.sort_values(
        ["category", "price"],
        ascending=[True, False]
    )
    return result["product"].tolist()


def t5_1_stock_by_category(products):
    """Return total stock for each category."""
    return products.groupby("category")["stock"].sum().to_dict()


def t5_2_avg_price_by_category(products):
    """Return average price per category, rounded to 2 decimals."""
    return products.groupby("category")["price"].mean().round(2).to_dict()


def t5_3_category_summary(products):
    """Return product count and maximum price for each category."""
    return products.groupby("category").agg(
        n=("product", "count"),
        max_price=("price", "max")
    ).to_dict("index")


def t5_4_left_merge_names(orders, products):
    """Left merge and return product names."""
    merged = orders.merge(products, on="product_id", how="left")
    return merged["product"].tolist()


def t5_5_inner_merge_rows(orders, products):
    """Return number of rows after an inner merge."""
    merged = orders.merge(products, on="product_id", how="inner")
    return len(merged)


def t5_6_revenue_list(orders, products):
    """Inner-merge, then return the revenue of each order (qty * price) as a list."""
    merged = orders.merge(products, on="product_id", how="inner")
    merged["revenue"] = merged["qty"] * merged["price"]
    return merged["revenue"].tolist()


def t5_7_revenue_by_customer(orders, products):
    """Calculate total revenue for each customer."""
    merged = orders.merge(products, on="product_id", how="left")
    merged["revenue"] = merged["qty"] * merged["price"]
    return merged.groupby("customer")["revenue"].sum().to_dict()


def t5_8_price_label(products):
    """Use apply to label each product cheap, mid or premium."""
    def label(price):
        if price < 50:
            return "cheap"
        elif price < 500:
            return "mid"
        else:
            return "premium"

    return products["price"].apply(label).tolist()


def t5_9_upper_names(products):
    """Return product names in uppercase."""
    return products["product"].str.upper().tolist()

# ===== Step 6: Feature Preparation =====

def t6_1_price_zscore(df):
    """Return price z-scores rounded to 2 decimals as a list."""
    z = (df["price"] - df["price"].mean()) / df["price"].std()
    return z.round(2).tolist()


def t6_2_dummy_columns(df):
    """Return one-hot encoded category column names."""
    result = pd.get_dummies(df, columns=["category"], prefix="cat", dtype=int)
    return [col for col in result.columns if col.startswith("cat_")]


def t6_3_feature_matrix(df):
    """Create the feature matrix and return the resulting DataFrame."""
    result = df.copy()
    result["price_z"] = (
        (result["price"] - result["price"].mean()) / result["price"].std()
    ).round(2)

    result = pd.get_dummies(
        result, columns=["category"], prefix="cat", dtype=int
    )

    result = result.drop(columns=["product_id", "product", "price"])
    return result


def t6_4_minmax_stock(df):
    """Scale stock to the 0-1 range and round to 2 decimals."""
    scaled = (df["stock"] - df["stock"].min()) / (
        df["stock"].max() - df["stock"].min()
    )
    return scaled.round(2).tolist()


def t6_5_train_test_sizes(df):
    """Return training and testing row counts using an 80/20 split."""
    train_rows = int(len(df) * 0.8)
    test_rows = len(df) - train_rows
    return (train_rows, test_rows)

# ===== Step 7: Capstone =====


def t7_1_top_customer(orders, products):
    """Return the customer with the highest total revenue."""
    merged = orders.merge(products, on="product_id", how="inner")
    merged["revenue"] = merged["qty"] * merged["price"]

    totals = merged.groupby("customer")["revenue"].sum()
    return totals.idxmax()


def t7_2_best_selling_product(orders, products):
    """Return the product with the highest total quantity sold."""
    merged = orders.merge(products, on="product_id", how="inner")

    totals = merged.groupby("product")["qty"].sum()
    return totals.idxmax()


def t7_3_customer_ranking(orders, products):
    """Return customers ranked by total revenue, highest first."""
    merged = orders.merge(products, on="product_id", how="inner")
    merged["revenue"] = merged["qty"] * merged["price"]

    totals = merged.groupby("customer")["revenue"].sum()
    return totals.sort_values(ascending=False).index.tolist()


def t7_4_unmatched_orders(orders, products):
    """Return order IDs whose product is not in the catalogue."""
    mask = ~orders["product_id"].isin(products["product_id"])
    return orders.loc[mask, "order_id"].tolist()


def t7_5_save_report(orders, products, path):
    """Save total revenue per customer to CSV, highest first."""
    merged = orders.merge(products, on="product_id", how="inner")
    merged["revenue"] = merged["qty"] * merged["price"]

    report = (
        merged.groupby("customer", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    report.to_csv(path, index=False)