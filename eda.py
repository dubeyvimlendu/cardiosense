import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/heart.csv")

# ── Basic inspection ─────────────────────────────────────
print("=" * 50)
print("SHAPE:", df.shape)
print("=" * 50)
print("\nFIRST 5 ROWS:")
print(df.head())
print("\nDATA TYPES:")
print(df.dtypes)
print("\nBASIC STATISTICS:")
print(df.describe())
print("\nMISSING VALUES:")
print(df.isnull().sum())
print("\nTARGET DISTRIBUTION:")
print(df["target"].value_counts())
print(f"\nDisease    : {df['target'].sum()} patients")
print(f"No Disease : {(df['target'] == 0).sum()} patients")
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Plot 1 — Target distribution
plt.figure()
colors = ["#4C9BE8", "#E8694C"]
df["target"].value_counts().plot(
    kind="bar",
    color=colors,
    edgecolor="white",
    width=0.5
)
plt.title("Heart Disease Distribution", fontsize=14, fontweight="bold")
plt.xticks([0, 1], ["No Disease (0)", "Disease (1)"], rotation=0)
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("data/plot_target_distribution.png")
plt.show()
print("Plot 1 saved!")
# Plot 2 — Age distribution by target
plt.figure()
df[df["target"] == 1]["age"].hist(
    alpha=0.7, color="#E8694C", label="Disease", bins=20
)
df[df["target"] == 0]["age"].hist(
    alpha=0.7, color="#4C9BE8", label="No Disease", bins=20
)
plt.title("Age Distribution by Heart Disease", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("data/plot_age_distribution.png")
plt.show()
print("Plot 2 saved!")

# Plot 3 — Gender vs Disease
plt.figure()
gender_disease = df.groupby(["sex", "target"]).size().unstack()
gender_disease.plot(
    kind="bar",
    color=["#4C9BE8", "#E8694C"],
    edgecolor="white",
    width=0.5
)
plt.title("Gender vs Heart Disease", fontsize=14, fontweight="bold")
plt.xticks([0, 1], ["Female", "Male"], rotation=0)
plt.ylabel("Count")
plt.legend(["No Disease", "Disease"])
plt.tight_layout()
plt.savefig("data/plot_gender_disease.png")
plt.show()
print("Plot 3 saved!")
# Plot 4 — Correlation heatmap
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("data/plot_correlation_heatmap.png")
plt.show()
print("Plot 4 saved!")

# Print top features correlated with target
print("\nTOP FEATURES CORRELATED WITH TARGET:")
print("=" * 40)
corr_target = correlation["target"].drop("target").sort_values(
    key=abs, ascending=False
)
for feat, val in corr_target.items():
    bar = "█" * int(abs(val) * 20)
    print(f"{feat:<12} {val:+.3f}  {bar}")
