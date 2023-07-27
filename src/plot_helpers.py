# general

import matplotlib.pyplot as plt
# ML
import pandas as pd
# visual
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

cols = ["precision", "recall", "f1-score", "support"]

def plot_results(reports, report_filename="results"):
    for name, report in reports.items():
        df = pd.DataFrame(report).transpose()
        df = df[cols]

        print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))

        # Save report to CSV
        df.to_csv(f"{report_filename} : {name}")

        print(f"Evaluation report saved to {report_filename}")
