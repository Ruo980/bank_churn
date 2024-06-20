import matplotlib.pyplot as plt
import numpy as np

# Data for each model
models = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]

# Metrics for each class
precision_0 = [0.88, 0.88, 0.89, 0.89]
precision_1 = [0.72, 0.78, 0.75, 0.75]

recall_0 = [0.95, 0.96, 0.95, 0.95]
recall_1 = [0.51, 0.49, 0.55, 0.55]

f1_0 = [0.91, 0.92, 0.92, 0.92]
f1_1 = [0.60, 0.60, 0.63, 0.63]

accuracy = [0.86, 0.86, 0.87, 0.87]
macro_avg_precision = [0.80, 0.83, 0.82, 0.82]
macro_avg_recall = [0.73, 0.73, 0.75, 0.75]
macro_avg_f1 = [0.76, 0.76, 0.78, 0.78]
weighted_avg_precision = [0.85, 0.85, 0.86, 0.86]
weighted_avg_recall = [0.86, 0.86, 0.87, 0.87]
weighted_avg_f1 = [0.85, 0.85, 0.86, 0.86]

x = np.arange(len(models))

# Plotting the metrics
fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # Increase the figure size

# Precision for each class (Bar Chart)
axs[0].bar(x - 0.2, precision_0, 0.4, label="Class 0")
axs[0].bar(x + 0.2, precision_1, 0.4, label="Class 1")
axs[0].set_title("Precision")
axs[0].set_xticks(x)
axs[0].set_xticklabels(models, rotation=45, ha="right")  # Rotate the x-labels
axs[0].legend()

# Recall for each class (Bar Chart)
axs[1].bar(x - 0.2, recall_0, 0.4, label="Class 0")
axs[1].bar(x + 0.2, recall_1, 0.4, label="Class 1")
axs[1].set_title("Recall")
axs[1].set_xticks(x)
axs[1].set_xticklabels(models, rotation=45, ha="right")  # Rotate the x-labels
axs[1].legend()

# F1-Score for each class (Bar Chart)
axs[2].bar(x - 0.2, f1_0, 0.4, label="Class 0")
axs[2].bar(x + 0.2, f1_1, 0.4, label="Class 1")
axs[2].set_title("F1-Score")
axs[2].set_xticks(x)
axs[2].set_xticklabels(models, rotation=45, ha="right")  # Rotate the x-labels
axs[2].legend()

# Accuracy & Macro Averages (Line Chart)
axs[3].plot(models, accuracy, marker="o", label="Accuracy", linestyle="-", color="blue")
axs[3].plot(
    models,
    macro_avg_precision,
    marker="o",
    label="Macro Avg Precision",
    linestyle="--",
    color="orange",
)
axs[3].plot(
    models,
    macro_avg_recall,
    marker="o",
    label="Macro Avg Recall",
    linestyle=":",
    color="green",
)
axs[3].plot(
    models,
    macro_avg_f1,
    marker="o",
    label="Macro Avg F1-Score",
    linestyle="-.",
    color="red",
)
axs[3].set_title("Accuracy & Macro Averages")
axs[3].legend()

# Weighted Averages (Line Chart)
axs[4].plot(
    models,
    weighted_avg_precision,
    marker="o",
    label="Weighted Avg Precision",
    linestyle="-",
    color="purple",
)
axs[4].plot(
    models,
    weighted_avg_recall,
    marker="o",
    label="Weighted Avg Recall",
    linestyle="--",
    color="brown",
)
axs[4].plot(
    models,
    weighted_avg_f1,
    marker="o",
    label="Weighted Avg F1-Score",
    linestyle=":",
    color="pink",
)
axs[4].set_title("Weighted Averages")
axs[4].legend()

plt.tight_layout()
plt.show()
