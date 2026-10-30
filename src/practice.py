"""Practice tasks for Pandas for AI Lab."""

from src import dataset as d
import pandas as pd


# ===== Step 1: Series and DataFrames =====

def t1_1_weather_series(df):
    """Set day as the index and return temp_c as a Series."""
    return df.set_index("day")["temp_c"]


def t1_2_hottest_day(df):
    """Return the day with the highest temperature."""
    return df.loc[df["temp_c"].idxmax(), "day"]


def t1_3_build_frame():
    """Build a DataFrame with item and price columns."""
    return pd.DataFrame({
        "item": ["pen", "book", "bag"],
        "price": [10, 120, 800]
    })


def t1_4_series_vs_frame(df):
    """Return the type names for single and double bracket selection."""
    return type(df["temp_c"]).__name__, type(df[["temp_c"]]).__name__


def t2_1_series_arithmetic(df):
    """Return temperature in Fahrenheit as a Series."""
    return df["temp_c"] * 9 / 5 + 32


def t2_2_filter_hot_days(df):
    """Return rows where temperature is greater than 25."""
    return df[df["temp_c"] > 25]


def t2_3_missing_values():
    """Create a Series and return its missing-value count."""
    s = pd.Series([10, None, 30, None, 50])
    return s.isna().sum()

def t4_1_cheap_products():
    """Return products with price below 100."""
    from src.dataset import make_products
    df = make_products()
    return df[df["price"] < 100]


def t4_2_accessories_in_stock():
    """Return accessories that have stock greater than zero."""
    from src.dataset import make_products
    df = make_products()
    return df[(df["category"] == "Accessories") & (df["stock"] > 0)]



def t4_3_top_two_by_price():
    """Return the two most expensive products."""
    from src.dataset import make_products
    df = make_products()
    return df.sort_values("price", ascending=False).head(2)

def t4_4_loc_price():
    """Select product prices using loc."""
    from src.dataset import make_products
    df = make_products()
    return df.loc[0:2, ["product", "price"]]


def t4_5_iloc_block():
    """Select rows and columns by position using iloc."""
    from src.dataset import make_products
    df = make_products()
    return df.iloc[0:3, 1:3]


def t4_6_low_stock_or_premium():
    """Return products with low stock or price above 200."""
    from src.dataset import make_products
    df = make_products()
    return df[(df["stock"] < 10) | (df["price"] > 200)]


def t4_7_sort_two_keys():
    """Sort products by category and price."""
    from src.dataset import make_products
    df = make_products()
    return df.sort_values(
        by=["category", "price"],
        ascending=[True, False]
    )


def t5_1_stock_by_category():
    """Return total stock for each category."""
    from src.dataset import make_products
    df = make_products()
    return df.groupby("category")["stock"].sum()


def t5_2_avg_price_by_category():
    """Return average price by category, rounded to 2 decimals."""
    from src.dataset import make_products
    df = make_products()
    return df.groupby("category")["price"].mean().round(2)


def t5_3_category_summary():
    """Return product count and average price for each category."""
    from src.dataset import make_products
    df = make_products()
    return df.groupby("category").agg(
        n=("product", "count"),
        avg_price=("price", "mean")
    )


def t5_4_left_merge_names():
    """Merge orders with products using a left join."""
    from src.dataset import make_products, make_orders
    products = make_products()
    orders = make_orders()
    return orders.merge(products, on="product_id", how="left")


def t5_5_inner_merge_rows():
    """Merge orders with products using an inner join."""
    from src.dataset import make_products, make_orders
    products = make_products()
    orders = make_orders()
    return orders.merge(products, on="product_id", how="inner")


def t5_6_revenue_list():
    """Return revenue for each order as a list."""
    from src.dataset import make_products, make_orders
    products = make_products()
    orders = make_orders()
    merged = orders.merge(products, on="product_id", how="left")
    merged["revenue"] = merged["price"] * merged["qty"]
    return merged["revenue"].tolist()


def t5_7_revenue_by_customer():
    """Return total revenue for each customer."""
    from src.dataset import make_products, make_orders
    products = make_products()
    orders = make_orders()
    merged = orders.merge(products, on="product_id", how="left")
    merged["revenue"] = merged["price"] * merged["qty"]
    return merged.groupby("customer")["revenue"].sum()


def t5_8_price_label():
    """Label products as budget or premium using apply."""
    from src.dataset import make_products
    df = make_products()
    return df["price"].apply(
        lambda price: "premium" if price >= 100 else "budget"
    )


def t5_9_upper_names():
    """Return product names in uppercase."""
    from src.dataset import make_products
    df = make_products()
    return df["product"].str.upper().tolist()


def t6_1_price_zscore(df):
    """Return price z-scores rounded to 2 decimals as a list."""
    return ((df["price"] - df["price"].mean()) / df["price"].std()).round(2).tolist()


def t6_2_dummy_columns(df):
    """One-hot encode category and return the new column names."""
    encoded = pd.get_dummies(df, columns=["category"], prefix="cat", dtype=int)
    return [col for col in encoded.columns if col.startswith("cat_")]


def t6_3_feature_matrix(df):
    """Create the feature matrix with scaled price and encoded category."""
    f = df.copy()
    f["price_z"] = (
        (f["price"] - f["price"].mean()) / f["price"].std()
    ).round(2)

    f = pd.get_dummies(f, columns=["category"], prefix="cat", dtype=int)
    return f.drop(columns=["product_id", "product", "price"])


def t6_4_minmax_stock(df):
    """Scale stock to the 0-1 range and return a rounded list."""
    stock = df["stock"]
    scaled = (stock - stock.min()) / (stock.max() - stock.min())
    return scaled.round(2).tolist()


def t6_5_train_test_sizes(df):
    """Return the training and testing row counts using an 80/20 split."""
    train_rows = int(len(df) * 0.8)
    test_rows = len(df) - train_rows
    return train_rows, test_rows


def t7_1_top_customer(orders, products):
    """Return the customer with the highest total revenue."""
    m = orders.merge(products, on="product_id", how="inner")
    m["revenue"] = m["qty"] * m["price"]
    totals = m.groupby("customer")["revenue"].sum()
    return totals.idxmax()


def t7_2_best_selling_product(orders, products):
    """Return the product with the highest total quantity sold."""
    m = orders.merge(products, on="product_id", how="inner")
    totals = m.groupby("product")["qty"].sum()
    return totals.idxmax()


def t7_3_customer_ranking(orders, products):
    """Return customers ranked by total revenue, highest first."""
    m = orders.merge(products, on="product_id", how="inner")
    m["revenue"] = m["qty"] * m["price"]
    return (
        m.groupby("customer")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )


def t7_4_unmatched_orders(orders, products):
    """Return orders whose product IDs are not in the catalogue."""
    return orders[~orders["product_id"].isin(products["product_id"])]


def t7_5_save_report(orders, products, path):
    """Save the customer revenue ranking to a CSV file."""
    ranking = t7_3_customer_ranking(orders, products)
    ranking.to_csv(path, header=["revenue"])
    return ranking