from src.dataset import make_orders, make_products
from src.practice import t7_5_save_report

t7_5_save_report(
    make_orders(),
    make_products(),
    "results/report.csv"
)

print("Saved results/report.csv")