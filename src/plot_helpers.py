# general

import os
import matplotlib.pyplot as plt
# ML
import pandas as pd
# visual
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate

cols = ["precision", "recall", "f1-score", "support"]


def plot_results(logger, reports, report_filename="results"):
    if reports is None:
        return

    logger.debug(f"plot report : {report_filename}")
    for name, report in reports.items():
        df = pd.DataFrame(report).transpose()
        df = df[cols]

        msg = "\n" + tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f")
        logger.debug(msg)
        # Save report to CSV
        # df.to_csv(os.path.join(debug_folder, f"{report_filename}_{name}"))

        # logger.debug(f"Evaluation report saved to {report_filename}")


def plot_steps_info(loss_train_values, loss_dev_values, accuracy_dev_values):
    pass
    # for name in ["nikud", "dagesh", "sin"]:
    #     df = pd.DataFrame(report).transpose()
    #     df = df[cols]
    #
    #     print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".4f"))
    #
    #     # Save report to CSV
    #     df.to_csv(f"{report_filename} : {name}")
    #
    #     print(f"Evaluation report saved to {report_filename}")


def generate_plot_by_nikud_dagesh_sin_dict(nikud_dagesh_sin_dict, title, y_axis, plot_folder=None):
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    plt.title(title)

    ax = plt.gca()
    indexes = list(range(1, len(nikud_dagesh_sin_dict["nikud"]) + 1))

    # Plot data series with different colors and labels
    ax.plot(indexes, nikud_dagesh_sin_dict["nikud"], color='blue', label='Nikud')
    ax.plot(indexes, nikud_dagesh_sin_dict["dagesh"], color='green', label='Dagesh')
    ax.plot(indexes, nikud_dagesh_sin_dict["sin"], color='red', label='Sin')

    # Add legend
    ax.legend()

    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel(y_axis)

    if plot_folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_folder, f'{title.replace(" ", "_")}_plot.jpg'))


def generate_word_and_letter_accuracy_plot(word_and_letter_accuracy_dict, title, plot_folder=None):
    # Create a figure and axis
    plt.figure(figsize=(8, 6))
    plt.title(title)

    ax = plt.gca()
    indexes = list(range(1, len(word_and_letter_accuracy_dict["all_nikud_word"]) + 1))

    # Plot data series with different colors and labels
    ax.plot(indexes, word_and_letter_accuracy_dict["all_nikud_letter"], color='blue', label='Letter')
    ax.plot(indexes, word_and_letter_accuracy_dict["all_nikud_word"], color='green', label='Word')

    # Add legend
    ax.legend()

    # Set labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Accuracy")

    if plot_folder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(plot_folder, 'word_and_letter_accuracy_plot.jpg'))
