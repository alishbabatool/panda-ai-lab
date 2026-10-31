from src.dataset import make_orders, make_products

orders = make_orders()
products = make_products()

print(orders.shape, products.shape)

m = orders.merge(
    products,
    on="product_id",
    how="left",
    indicator=True
)

print(m["_merge"].value_counts())

print(
    m[m["_merge"] == "left_only"][
        ["order_id", "customer", "product_id"]
    ]
)