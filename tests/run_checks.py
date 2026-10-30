import pandas as pd


def t7_2_best_selling_product(orders, products):
    """Return the name of the product with the highest total quantity sold."""
    merged = orders.merge(
        products[["product_id", "product_name"]],
        on="product_id",
        how="inner"
    )

    sales = merged.groupby("product_name")["quantity"].sum()

    if sales.empty:
        return None

    return sales.idxmax()


def t7_3_customer_ranking(orders, products):
    """Return customer names from highest to lowest total revenue."""
    merged = orders.merge(
        products[["product_id", "price"]],
        on="product_id",
        how="inner"
    )

    merged["revenue"] = merged["quantity"] * merged["price"]

    ranking = (
        merged.groupby("customer")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    return ranking.index.tolist()


def t7_4_unmatched_orders(orders, products):
    """Return order_id values whose product_id is not in the catalogue."""
    unmatched = orders.loc[
        ~orders["product_id"].isin(products["product_id"]),
        "order_id"
    ]

    return unmatched.tolist()


def t7_5_save_report(orders, products, path):
    """Save a CSV with customer and total revenue, sorted highest first."""
    merged = orders.merge(
        products[["product_id", "price"]],
        on="product_id",
        how="inner"
    )

    merged["revenue"] = merged["quantity"] * merged["price"]

    report = (
        merged.groupby("customer", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    report.to_csv(path, index=False)

    return report
