"""Shared datasets for the pandas lab."""

import pandas as pd


def make_df():
    """Customers (worked examples, Steps 1-6)."""
    return pd.DataFrame({
        "name": ["Ali", "Sara", "Omar", "Sara", "Zoya"],
        "city": [
            "Lahore",
            "Karachi",
            "Lahore",
            "Karachi",
            "Multan"
        ],
        "age": [25, 31, None, 31, 40],
        "salary": [50, 62, 58, 62, 45]
    })


def make_depts():
    """City to region lookup."""
    return pd.DataFrame({
        "city": ["Lahore", "Karachi", "Quetta"],
        "region": ["North", "South", "West"]
    })


def make_weather():
    """Daily temperatures."""
    return pd.DataFrame({
        "day": [
            "Mon", "Tue", "Wed", "Thu",
            "Fri", "Sat", "Sun"
        ],
        "temp_c": [21, 25, 23, 28, 30, 27, 24]
    })


def make_products():
    """Product catalogue."""
    return pd.DataFrame({
        "product_id": [101, 102, 103, 104, 105, 106],
        "product": [
            "Laptop",
            "Mouse",
            "Keyboard",
            "Monitor",
            "Webcam",
            "Desk"
        ],
        "category": [
            "Computers",
            "Accessories",
            "Accessories",
            "Computers",
            "Accessories",
            "Furniture"
        ],
        "price": [900, 15, 30, 250, 45, 180],
        "stock": [5, 120, 80, 12, 0, 7]
    })


def make_orders():
    """Customer orders; product 107 is not in the catalogue."""
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "customer": [
            "Ali", "Sara", "Ali", "Omar",
            "Zoya", "Sara", "Ali", "Hina"
        ],
        "product_id": [101, 102, 103, 102, 104, 105, 107, 101],
        "qty": [1, 2, 1, 3, 1, 2, 1, 1]
    })


def make_students():
    """Messy exam scores."""
    return pd.DataFrame({
        "student": [
            "Hira", "Bilal", "Areeba", "Usman",
            "Noor", "Fahad", "Noor"
        ],
        "section": ["A", "B", "A", "B", "A", "B", "A"],
        "math": ["78", "91", "n/a", "65", "88", "72", "88"],
        "english": [82, None, 74, 69, None, 90, None]
    })