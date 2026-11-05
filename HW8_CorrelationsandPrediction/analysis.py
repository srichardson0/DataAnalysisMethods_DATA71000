import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILE = "HW8_CorrelationsandPrediction/State Crime Rate Data_1993.xls"
df = pd.read_excel(DATA_FILE)
df = df.rename(columns={
    "Violent Crime Rate": "Violent crime", "Murder Rate": "Murder",
    "% Urban ": "% Urban", "% H.S Graduates": "% HS grads",
    "% Poverty": "% Poverty", "% Single Parent Household": "% Single parent"})
numeric = df.drop(columns="State")
print(f"n = {len(df)} cases, missing values = {int(df.isna().sum().sum())}")

# ---- 1. the full correlation matrix (pandas) ----
corr = numeric.corr(method="pearson")
print("\nPearson correlation matrix (all 51 cases)")
print(corr.round(2))

# two-tailed p-values for H0: rho = 0, df = n - 2
n = len(numeric)
t = corr * np.sqrt((n - 2) / (1 - corr ** 2 + 1e-12))
pvals = pd.DataFrame(2 * stats.t.sf(np.abs(t), n - 2), index=corr.index, columns=corr.columns)
print("\np-values")
print(pvals.round(3))

# ---- 2. the heatmap of the matrix ----
fig, ax = plt.subplots(figsize=(7.6, 6.2))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        val = corr.values[i, j]
        ax.text(j, i, (f"{val:.2f}" if abs(val) >= 0.005 else "0.00").replace("-", "−"), ha="center", va="center",
                fontsize=9, color="white" if abs(val) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
ax.set_title("Correlation matrix, 1993 state crime data (n = 51)")
plt.tight_layout()
plt.savefig("HW8_CorrelationsandPrediction/correlation_heatmap.png", dpi=200)

# ---- 3. partial correlations used in the write-up ----
def partial(x, y, z):
    return (corr.loc[x, y] - corr.loc[x, z] * corr.loc[y, z]) / np.sqrt(
        (1 - corr.loc[x, z] ** 2) * (1 - corr.loc[y, z] ** 2))

print("\nPoverty with Murder, controlling for single parent:",
      round(partial("% Poverty", "Murder", "% Single parent"), 2))
print("Urban with Violent crime, controlling for Murder:",
      round(partial("% Urban", "Violent crime", "Murder"), 2))