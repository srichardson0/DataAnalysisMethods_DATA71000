import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

df = pd.read_excel("HW10_RegressionAnalysisContinued/Wage2.xls")
n = len(df)
print(f"n = {n}, missing = {int(df.isna().sum().sum())}")
print(df[["Wage2", "educ", "exper", "tenure"]].describe().round(2))
print("\nCorrelations\n", df[["Wage2", "educ", "exper", "tenure"]].corr().round(3))

formulas = {"Model 1": "Wage2 ~ educ + exper + tenure",
            "Model 2": "Wage2 ~ educ + exper",
            "Model 3": "Wage2 ~ educ + tenure"}
fits = {}
for name, f in formulas.items():
    m = smf.ols(f, data=df).fit()
    fits[name] = m
    print(f"\n===== {name}: {f}")
    print(pd.DataFrame({"b": m.params, "SE": m.bse, "t": m.tvalues, "p": m.pvalues,
                        "CI low": m.conf_int()[0], "CI high": m.conf_int()[1]}).round(4))
    print(f"R2 = {m.rsquared:.4f}, adjusted R2 = {m.rsquared_adj:.4f}, s = {np.sqrt(m.scale):.3f}, "
          f"F({int(m.df_model)}, {int(m.df_resid)}) = {m.fvalue:.2f}, p = {m.f_pvalue:.2e}")

# does adding the third predictor help? (nested F tests)
m1, m2, m3 = fits["Model 1"], fits["Model 2"], fits["Model 3"]
for small, label in [(m2, "tenure added to Model 2"), (m3, "exper added to Model 3")]:
    f_val, p_val, df_diff = m1.compare_f_test(small)
    print(f"\n{label}: F = {f_val:.2f}, p = {p_val:.2e}, R2 gain = {m1.rsquared - small.rsquared:.4f}")

# influence and residual checks for the full model (Model 1)
inf = OLSInfluence(m1)
cooks = inf.cooks_distance[0]
print(f"\nModel 1 Cook's D: max = {cooks.max():.3f}, cases above 4/n = {(cooks > 4 / n).sum()}")
top = np.argsort(-cooks)[:5]
print(df.iloc[top][["record_id", "Wage2", "educ", "exper", "tenure"]].assign(cooks_d=cooks[top].round(3)))
keep = cooks <= 4 / n
trim = smf.ols(formulas["Model 1"], data=df[keep]).fit()
print(f"Model 1 without {(~keep).sum()} high-influence cases:", trim.params.round(4).to_dict(),
      f"adj R2 = {trim.rsquared_adj:.4f}")
print("Negative fitted values in Model 1:", int((m1.fittedvalues < 0).sum()))

# figures: residuals vs fitted for each model, plus Cook's distance for Model 1
fig, axs = plt.subplots(1, 4, figsize=(13, 3.3))
for ax, (name, m) in zip(axs[:3], fits.items()):
    ax.scatter(m.fittedvalues, m.resid, s=9, alpha=.45, color="#3b6ea5", linewidths=0)
    ax.axhline(0, color="#c0504d", lw=1.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted wage ($)")
axs[0].set_ylabel("Residual ($)")
axs[3].stem(np.arange(n), cooks, markerfmt=" ", basefmt=" ", linefmt="#3b6ea5")
axs[3].set_title("How much each worker\naffects Model 1", fontsize=10)
axs[3].set_xlabel("Worker")
for ax in axs:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig("HW10_RegressionAnalysisContinued/fig_residuals.png", dpi=200)
print("\nSaved fig_residuals.png")