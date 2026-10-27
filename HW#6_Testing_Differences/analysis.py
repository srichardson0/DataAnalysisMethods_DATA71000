
import numpy as np
import pandas as pd
from scipy import stats

DATA_FILE = "HW#6_Testing_Differences/Student_Performance_Data_V3.xls"
SCORES = ["MathScore", "ReadingScore", "WritingScore"]

# ---------------------------------------------------------------------------
# Load and check the data
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA_FILE)
print(f"Rows: {len(df)}   Missing values per column: {df.isna().sum().to_dict()}")
print("Sex counts (0 = female, 1 = male):", df["Sex"].value_counts().sort_index().to_dict())
print("TestPrep counts (0 = none, 1 = completed):", df["TestPrep"].value_counts().sort_index().to_dict())

ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(SCORES)  # three tests per part -> .0167 (six tests overall -> .0083)


def pooled_sd(a, b):
    """Pooled standard deviation of two groups."""
    n1, n2 = len(a), len(b)
    return np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))


# ---------------------------------------------------------------------------
# Part (a): t-tests by Sex
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART (a)  Independent-samples t-tests by Sex (difference = male - female)")
print("=" * 70)

female = df[df["Sex"] == 0]
male = df[df["Sex"] == 1]

for score in SCORES:
    f, m = female[score], male[score]
    n1, n2 = len(f), len(m)
    dfree = n1 + n2 - 2

    # Levene's test checks whether the two groups have similar variances
    levene_p = stats.levene(f, m).pvalue

    # Pooled-variance (Student) t-test, plus Welch as a check
    t_stat, p_val = stats.ttest_ind(m, f, equal_var=True)
    t_welch, p_welch = stats.ttest_ind(m, f, equal_var=False)

    # Mean difference with a 95% confidence interval
    diff = m.mean() - f.mean()
    sp = pooled_sd(f, m)
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    t_crit = stats.t.ppf(0.975, dfree)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    # Cohen's d (effect size)
    d = diff / sp

    print(f"\n{score}")
    print(f"  Female: M = {f.mean():.2f}, SD = {f.std():.2f}, n = {n1}")
    print(f"  Male:   M = {m.mean():.2f}, SD = {m.std():.2f}, n = {n2}")
    print(f"  Difference = {diff:.2f}, 95% CI [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  t({dfree}) = {t_stat:.2f}, p = {p_val:.2e}, Cohen's d = {d:.2f}")
    print(f"  Levene p = {levene_p:.3f};  Welch t = {t_welch:.2f}, p = {p_welch:.2e}")
    print(f"  Significant after Bonferroni (alpha = {BONFERRONI_ALPHA:.4f})? {p_val < BONFERRONI_ALPHA}")

# ---------------------------------------------------------------------------
# Part (b): one-way ANOVA by TestPrep
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PART (b)  One-way ANOVA by TestPrep")
print("=" * 70)

for score in SCORES:
    groups = [df[df["TestPrep"] == g][score] for g in (0, 1)]
    none, done = groups
    k = len(groups)
    N = sum(len(g) for g in groups)
    grand_mean = df[score].mean()

    # Sums of squares, computed by hand to match the course notes
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_total = ss_between + ss_within

    df_between, df_within = k - 1, N - k
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within
    p_val = stats.f.sf(F, df_between, df_within)
    eta_sq = ss_between / ss_total

    # Cross-check against scipy's built-in ANOVA
    F_check, p_check = stats.f_oneway(*groups)

    # Same comparison as a t-test, to show F = t^2
    t_stat, _ = stats.ttest_ind(done, none, equal_var=True)
    levene_p = stats.levene(none, done).pvalue
    t_welch, p_welch = stats.ttest_ind(done, none, equal_var=False)

    print(f"\n{score}")
    print(f"  None:      M = {none.mean():.2f}, SD = {none.std():.2f}, n = {len(none)}")
    print(f"  Completed: M = {done.mean():.2f}, SD = {done.std():.2f}, n = {len(done)}")
    print(f"  Difference (completed - none) = {done.mean() - none.mean():.2f}")
    print(f"  {'Source':<10}{'SS':>14}{'df':>6}{'MS':>14}{'F':>10}")
    print(f"  {'Between':<10}{ss_between:>14,.1f}{df_between:>6}{ms_between:>14,.2f}{F:>10.2f}")
    print(f"  {'Within':<10}{ss_within:>14,.1f}{df_within:>6}{ms_within:>14,.2f}")
    print(f"  {'Total':<10}{ss_total:>14,.1f}{N - 1:>6}")
    print(f"  F({df_between}, {df_within}) = {F:.2f}, p = {p_val:.2e}, eta-squared = {eta_sq:.3f}")
    print(f"  scipy check: F = {F_check:.2f}, p = {p_check:.2e}")
    print(f"  t = {t_stat:.3f}, t^2 = {t_stat ** 2:.2f} (equals F)")
    print(f"  Levene p = {levene_p:.3f};  Welch t = {t_welch:.2f}, p = {p_welch:.2e}")

# ---------------------------------------------------------------------------
# Permutation tests (Lecture 5.1): build the chance model by shuffling labels
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PERMUTATION TESTS  (10,000 shuffles of the group labels, two-way)")
print("=" * 70)

rng = np.random.default_rng(71000)  # fixed seed so results reproduce
R = 10_000

for group in ["Sex", "TestPrep"]:
    labels = df[group].to_numpy()
    n_group1 = (labels == 1).sum()
    for score in SCORES:
        x = df[score].to_numpy()
        observed = x[labels == 1].mean() - x[labels == 0].mean()
        perm_diffs = np.empty(R)
        for i in range(R):
            shuffled = rng.permutation(x)          # combine, then reshuffle
            perm_diffs[i] = shuffled[:n_group1].mean() - shuffled[n_group1:].mean()
        count = (np.abs(perm_diffs) >= abs(observed)).sum()
        p_perm = count / R
        p_text = f"{p_perm:.4f}" if count > 0 else f"< {1 / R:.4f}"
        print(f"{group:<9}{score:<14} observed diff = {observed:6.2f}   "
              f"largest |shuffled diff| = {np.abs(perm_diffs).max():.2f}   "
              f"shuffles as extreme = {count}   permutation p {p_text}")

# ---------------------------------------------------------------------------
# Multiple testing and a design check
# ---------------------------------------------------------------------------
n_tests = 6
print(f"\nChance of at least one false positive in {n_tests} tests at alpha = .05: "
      f"{1 - 0.95 ** n_tests:.2f}")
print(f"Bonferroni-adjusted alpha for {n_tests} tests: {ALPHA / n_tests:.4f}")

print("\nTest prep completion rate by Sex (0 = female, 1 = male):")
print(df.groupby("Sex")["TestPrep"].mean().round(3).to_dict())
