"""
DATA 71000 HW #9: hourly earnings, education and experience (Wage2, 1976, n = 526)
Requires: pip install pandas numpy matplotlib statsmodels xlrd
Run:      python hw9_analysis.py   (keep Wage2.xls in the same folder)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import OLSInfluence

df = pd.read_excel("HW9_StatisticalModelingUsingRegressionAnalyses/Wage2.xls")
n = len(df)
print(f"n = {n}, missing = {int(df.isna().sum().sum())}")
print(df[["Wage2", "educ", "exper"]].describe().round(2))
print("\nCorrelations\n", df[["Wage2", "educ", "exper"]].corr().round(3))

# ---- the three models ----
formulas = {"Model 1": "Wage2 ~ educ",
            "Model 2": "Wage2 ~ exper",
            "Model 3": "Wage2 ~ educ + exper"}
fits = {}
for name, f in formulas.items():
    m = smf.ols(f, data=df).fit()
    fits[name] = m
    print(f"\n===== {name}: {f}")
    table = pd.DataFrame({"b": m.params, "SE": m.bse, "t": m.tvalues, "p": m.pvalues,
                          "CI low": m.conf_int()[0], "CI high": m.conf_int()[1]})
    print(table.round(4))
    print(f"R2 = {m.rsquared:.4f}, adjusted R2 = {m.rsquared_adj:.4f}, "
          f"s = {np.sqrt(m.scale):.3f}, F({int(m.df_model)}, {int(m.df_resid)}) = {m.fvalue:.2f}, "
          f"p = {m.f_pvalue:.2e}")
    print(f"SSE = {m.ssr:.1f}, total SS = {m.centered_tss:.1f}")

# ---- which predictor matters more? ----
m1, m2, m3 = fits["Model 1"], fits["Model 2"], fits["Model 3"]
z = df[["Wage2", "educ", "exper"]].apply(lambda c: (c - c.mean()) / c.std())
print("\nStandardized coefficients, Model 3:",
      smf.ols("Wage2 ~ educ + exper", data=z).fit().params.round(3).to_dict())
print(f"R2 added by educ beyond exper: {m3.rsquared - m2.rsquared:.4f}")
print(f"R2 added by exper beyond educ: {m3.rsquared - m1.rsquared:.4f}")

# ---- diagnostics for Model 3 ----
inf = OLSInfluence(m3)
cooks = inf.cooks_distance[0]
stud = inf.resid_studentized_external
lev = inf.hat_matrix_diag
print(f"\nCook's D: max = {cooks.max():.3f}, cases above 4/n ({4 / n:.4f}) = {(cooks > 4 / n).sum()}")
print(f"Studentized residuals beyond +/-3: {(np.abs(stud) > 3).sum()}")
top = np.argsort(-cooks)[:5]
print(df.iloc[top][["record_id", "Wage2", "educ", "exper"]]
      .assign(cooks_d=cooks[top].round(3), stud_resid=stud[top].round(2), leverage=lev[top].round(3)))

# refit without the high-influence cases to see how much they matter
keep = cooks <= 4 / n
m3_trim = smf.ols("Wage2 ~ educ + exper", data=df[keep]).fit()
print(f"\nModel 3 without {(~keep).sum()} high-Cook's-D cases:",
      m3_trim.params.round(4).to_dict(), f"adj R2 = {m3_trim.rsquared_adj:.4f}")

# check for a bend in experience and for unequal spread
quad = smf.ols("Wage2 ~ educ + exper + I(exper**2)", data=df).fit()
print(f"\nExperience squared term: b = {quad.params['I(exper ** 2)']:.5f}, "
      f"p = {quad.pvalues['I(exper ** 2)']:.2e}, adj R2 = {quad.rsquared_adj:.4f}")
hc3 = m3.get_robustcov_results(cov_type="HC3")
print("Model 3 robust (HC3) standard errors:", np.round(hc3.bse, 4))
print(f"Skewness of Wage2: {df['Wage2'].skew():.2f}")

# ---- figures ----
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, x, m, lab in [(axs[0], "educ", m1, "Education (years)"),
                      (axs[1], "exper", m2, "Potential experience (years)")]:
    ax.scatter(df[x], df["Wage2"], s=10, alpha=.45, color="#3b6ea5", linewidths=0)
    xs = np.linspace(df[x].min(), df[x].max(), 50)
    ax.plot(xs, m.params["Intercept"] + m.params[x] * xs, color="#c0504d", lw=2)
    ax.set_xlabel(lab)
    ax.set_title(f"r = {df[[x, 'Wage2']].corr().iloc[0, 1]:.2f}", fontsize=10)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axs[0].set_ylabel("Hourly wage ($)")
plt.tight_layout()
plt.savefig("HW9_StatisticalModelingUsingRegressionAnalyses/fig1_scatter.png", dpi=200)
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
axs[0].scatter(m3.fittedvalues, m3.resid, s=10, alpha=.45, color="#3b6ea5", linewidths=0)
axs[0].axhline(0, color="#c0504d", lw=1.5)
axs[0].set_xlabel("Fitted wage ($)"); axs[0].set_ylabel("Residual ($)")
axs[0].set_title("Residuals vs fitted", fontsize=10)
axs[1].scatter(df["exper"], m3.resid, s=10, alpha=.45, color="#3b6ea5", linewidths=0)
axs[1].axhline(0, color="#c0504d", lw=1.5)
axs[1].set_xlabel("Potential experience (years)")
axs[1].set_title("Residuals vs experience", fontsize=10)
axs[2].stem(np.arange(n), cooks, markerfmt=" ", basefmt=" ", linefmt="#3b6ea5")
axs[2].axhline(4 / n, color="#c0504d", lw=1, ls="--")
axs[2].set_xlabel("Observation"); axs[2].set_ylabel("Cook's distance")
axs[2].set_title("Influence (dashed line = 4/n)", fontsize=10)
for ax in axs:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig("HW9_StatisticalModelingUsingRegressionAnalyses/fig2_diagnostics.png", dpi=200)
plt.close()
print("\nSaved fig1_scatter.png and fig2_diagnostics.png")