"""Auto-checker. Usage: python -m tests.run_checks [step_number]"""

import math
import sys
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd
from src import dataset as d
from src import practice as p


NAN = float("nan")
ZSCORES = [1.96, -0.66, -0.61, 0.04, -0.57, -0.17]
TMP = "results/_tmp_"


def feature_summary():
    X = p.t6_3_feature_matrix(d.make_products())
    return (
        X.shape,
        sorted(X.columns.tolist()),
        X["price_z"].tolist()
    )


def cleaned_summary():
    c = p.t3_7_clean_students(d.make_students())
    return (
        c.shape,
        round(float(c["math"].mean()), 2),
        int(c.isna().sum().sum())
    )


def saved_report():
    p.t7_5_save_report(
        d.make_orders(),
        d.make_products(),
        TMP + "report.csv"
    )
    r = pd.read_csv(TMP + "report.csv")
    return (r.shape, r.columns.tolist())


CHECKS = {
    1: [
        ("1.1 weather_series",
         lambda: p.t1_1_weather_series(d.make_weather()).to_dict(),
         {"Mon": 21, "Tue": 25, "Wed": 23, "Thu": 28,
          "Fri": 30, "Sat": 27, "Sun": 24}),

        ("1.2 hottest_day",
         lambda: p.t1_2_hottest_day(d.make_weather()),
         "Fri"),

        ("1.3 build_frame",
         lambda: p.t1_3_build_frame().to_dict("list"),
         {"item": ["pen", "book", "bag"], "price": [10, 120, 800]}),

        ("1.4 series_vs_frame",
         lambda: p.t1_4_series_vs_frame(d.make_weather()),
         ("Series", "DataFrame")),
    ],

    2: [
        ("2.1 csv_roundtrip",
         lambda: p.t2_1_csv_roundtrip(d.make_products(), TMP + "a.csv"),
         (6, 5)),

        ("2.2 json_columns",
         lambda: p.t2_2_json_columns(d.make_products(), TMP + "a.json"),
         ["product_id", "product", "category", "price", "stock"]),

        ("2.3 excel_roundtrip",
         lambda: p.t2_3_excel_roundtrip(d.make_products(), TMP + "a.xlsx"),
         (6, 5)),

        ("2.4 index_trap",
         lambda: p.t2_4_index_trap(d.make_products(), TMP + "b.csv"),
         6),
    ],

    3: [
        ("3.1 count_missing",
         lambda: p.t3_1_count_missing(d.make_students()),
         {"student": 0, "section": 0, "math": 0, "english": 3}),

        ("3.2 math_numeric",
         lambda: p.t3_2_math_numeric(d.make_students()),
         [78.0, 91.0, NAN, 65.0, 88.0, 72.0, 88.0]),

        ("3.3 remove_duplicates",
         lambda: p.t3_3_remove_duplicates(d.make_students()),
         (6, 4)),

        ("3.4 fill_english_median",
         lambda: p.t3_4_fill_english_median(d.make_students()),
         [82.0, 78.0, 74.0, 69.0, 78.0, 90.0]),

        ("3.5 rows_after_dropna",
         lambda: p.t3_5_rows_after_dropna(d.make_students()),
         3),

        ("3.6 section_category",
         lambda: p.t3_6_section_category(d.make_students()),
         "category"),

        ("3.7 clean_students",
         cleaned_summary,
         ((6, 4), 78.8, 0)),
    ],

    4: [
        ("4.1 cheap_products",
         lambda: p.t4_1_cheap_products(d.make_products()),
         ["Mouse", "Keyboard", "Webcam"]),

        ("4.2 accessories_in_stock",
         lambda: p.t4_2_accessories_in_stock(d.make_products()),
         ["Mouse", "Keyboard"]),

        ("4.3 top_two_by_price",
         lambda: p.t4_3_top_two_by_price(d.make_products()),
         ["Laptop", "Monitor"]),

        ("4.4 loc_price",
         lambda: p.t4_4_loc_price(d.make_products()),
         250),

        ("4.5 iloc_block",
         lambda: p.t4_5_iloc_block(d.make_products()),
         [[101, "Laptop"], [102, "Mouse"]]),

        ("4.6 low_stock_or_premium",
         lambda: p.t4_6_low_stock_or_premium(d.make_products()),
         ["Laptop", "Webcam"]),

        ("4.7 sort_two_keys",
         lambda: p.t4_7_sort_two_keys(d.make_products()),
         ["Webcam", "Keyboard", "Mouse", "Laptop", "Monitor", "Desk"]),
    ],

    5: [
        ("5.1 stock_by_category",
         lambda: p.t5_1_stock_by_category(d.make_products()),
         {"Accessories": 200, "Computers": 17, "Furniture": 7}),

        ("5.2 avg_price_by_category",
         lambda: p.t5_2_avg_price_by_category(d.make_products()),
         {"Accessories": 30.0, "Computers": 575.0, "Furniture": 180.0}),

        ("5.3 category_summary",
         lambda: p.t5_3_category_summary(d.make_products()),
         {
             "Accessories": {"n": 3, "max_price": 45},
             "Computers": {"n": 2, "max_price": 900},
             "Furniture": {"n": 1, "max_price": 180}
         }),

        ("5.4 left_merge_names",
         lambda: p.t5_4_left_merge_names(d.make_orders(), d.make_products()),
         ["Laptop", "Mouse", "Keyboard", "Mouse", "Monitor",
          "Webcam", NAN, "Laptop"]),

        ("5.5 inner_merge_rows",
         lambda: p.t5_5_inner_merge_rows(d.make_orders(), d.make_products()),
         7),

        ("5.6 revenue_list",
         lambda: p.t5_6_revenue_list(d.make_orders(), d.make_products()),
         [900, 30, 30, 45, 250, 90, 900]),

        ("5.7 revenue_by_customer",
         lambda: p.t5_7_revenue_by_customer(d.make_orders(), d.make_products()),
         {"Ali": 930, "Hina": 900, "Omar": 45, "Sara": 120, "Zoya": 250}),

        ("5.8 price_label",
         lambda: p.t5_8_price_label(d.make_products()),
         ["premium", "cheap", "cheap", "mid", "cheap", "mid"]),

        ("5.9 upper_names",
         lambda: p.t5_9_upper_names(d.make_products()),
         ["LAPTOP", "MOUSE", "KEYBOARD", "MONITOR", "WEBCAM", "DESK"]),
    ],

    6: [
        ("6.1 price_zscore",
         lambda: p.t6_1_price_zscore(d.make_products()),
         ZSCORES),

        ("6.2 dummy_columns",
         lambda: p.t6_2_dummy_columns(d.make_products()),
         ["cat_Accessories", "cat_Computers", "cat_Furniture"]),

        ("6.3 feature_matrix",
         feature_summary,
         (
             (6, 5),
             ["cat_Accessories", "cat_Computers", "cat_Furniture",
              "price_z", "stock"],
             ZSCORES
         )),

        ("6.4 minmax_stock",
         lambda: p.t6_4_minmax_stock(d.make_products()),
         [0.04, 1.0, 0.67, 0.1, 0.0, 0.06]),

        ("6.5 train_test_sizes",
         lambda: p.t6_5_train_test_sizes(d.make_products()),
         (4, 2)),
    ],

    7: [
        ("7.1 top_customer",
         lambda: p.t7_1_top_customer(d.make_orders(), d.make_products()),
         "Ali"),

        ("7.2 best_selling_product",
         lambda: p.t7_2_best_selling_product(d.make_orders(), d.make_products()),
         "Mouse"),

        ("7.3 customer_ranking",
         lambda: p.t7_3_customer_ranking(d.make_orders(), d.make_products()),
         ["Ali", "Hina", "Zoya", "Sara", "Omar"]),

        ("7.4 unmatched_orders",
         lambda: p.t7_4_unmatched_orders(d.make_orders(), d.make_products()),
         [7]),

        ("7.5 save_report",
         saved_report,
         ((5, 2), ["customer", "revenue"])),
    ],
}


def same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            same(x, y) for x, y in zip(a, b)
        )

    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True

    return a == b


def run(step):
    passed = 0

    for name, fn, expected in CHECKS[step]:
        try:
            got = fn()
            ok = bool(same(got, expected))

            print(
                f" {'PASS' if ok else 'FAIL'} {name}"
                + ("" if ok else f" -> you returned {got!r}")
            )

            passed += ok

        except NotImplementedError:
            print(f" TODO {name}")

        except Exception as e:
            print(f" ERROR {name} -> {type(e).__name__}: {e}")

    return passed, len(CHECKS[step])


if __name__ == "__main__":

    # Create results folder if it does not exist
    Path("results").mkdir(parents=True, exist_ok=True)

    # Capture checker output
    output = StringIO()

    steps = [int(sys.argv[1])] if len(sys.argv) > 1 else sorted(CHECKS)

    got = total = 0

    with redirect_stdout(output):
        for s in steps:
            print(f"Step {s}")
            a, b = run(s)
            got += a
            total += b

        print(f"\nScore: {got}/{total}")

    # Display output in terminal
    report_text = output.getvalue()
    print(report_text, end="")

    # Save output to check_report.txt
    report_path = Path("results/check_report.txt")
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\nChecker report saved to: {report_path}")