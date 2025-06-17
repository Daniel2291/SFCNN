import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_error_vs_rotation(file_label_pairs):
    plt.figure(figsize=(10, 6))

    for file_path, label in file_label_pairs:
        df = pd.read_csv(file_path)
        df['Error'] = 1 - df['Test Accuracy']

        plt.plot(
            df['Rotation Angle (deg)'],
            df['Error'],
            label=label,
            linewidth = 2.5
        )

    plt.yscale('log')
    plt.xlabel('Rotation Angle (deg)')
    plt.ylabel('Test Error (log scale)')
    plt.title('Test Error vs Rotation Angle')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  Define file paths and custom labels
file_label_pairs = [
    ("EXP5_45_regular_16_False_0_None_False_True_None_None_2_rot_accuracy.csv", "SFCNN 45 aug"),
    ("EXP5_90_regular_16_False_0_None_False_True_None_None_2_rot_accuracy.csv", "SFCNN 90 aug"),
    ("EXP5_0_regular_16_False_0_None_False_True_None_None_2_rot_accuracy.csv", "SFCNN no aug"),
    ("CNN_regular_16_False_0_None_False_True_None_None_2_rot_accuracy.csv", "CNN no aug"),

]

plot_error_vs_rotation(file_label_pairs)

