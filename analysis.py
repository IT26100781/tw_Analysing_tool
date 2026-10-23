"""
IT1150 Assignment II - Survey Data Analysis
Topic: Time Spent on Social Media During Study Hours and Its Perceived
       Effect on Academic Performance

This program performs the analysis described in Section 2.5 (Method of
Analysis) of the report:
  1. Loads and cleans the survey responses exported from Google Forms (CSV).
  2. Calculates frequencies and percentages for every closed question
     (percentages are based on the full sample).
  3. Handles multiple-response questions (Q5, Q9, Q22) by counting each
     selected option separately.
  4. Builds cross-tabulations between key variables.
  5. Tests associations between ordered variables with Spearman's rank
     correlation.
  6. Produces the figures used in the report (Matplotlib).
  7. Exports the open-ended answers (Q23-Q25) so they can be grouped into
     themes by hand.

Requirements:  pip install pandas matplotlib scipy
Usage:         python analyze_survey.py "path/to/responses.csv"
               (if no path is given, the first .csv in the current folder is used)
Output:        an "output" folder containing figures/, tables/ and summary.txt
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# 1. Settings
# ----------------------------------------------------------------------------
OUT_DIR = "output"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

# Short names for the 25 questions (the Google Form headers are very long)
QUESTION_NAMES = {
    "q1": "Age group", "q2": "Gender", "q3": "Education level", "q4": "Field of study",
    "q5": "Platforms used most", "q6": "Daily hours on social media",
    "q7": "Check social media while studying", "q8": "Primary reason for use while studying",
    "q9": "Triggers for checking", "q10": "Daily study hours",
    "q11": "Phone placement while studying", "q12": "Phone checking frequency",
    "q13": "Time to return to full focus", "q14": "Self-rated academic performance",
    "q15": "Perceived effect on performance", "q16": "Results/GPA suffered",
    "q17": "Use social media for education", "q18": "Stay up late on social media",
    "q19": "Statement agreed with most", "q20": "Reducing use would improve performance",
    "q21": "Tried to limit use", "q22": "Strategies to manage distractions",
    "q23": "Biggest disadvantage (open)", "q24": "Advice to others (open)",
    "q25": "Best focus technique (open)",
}
MULTI_RESPONSE = ["q5", "q9", "q22"]      # respondents could tick several options
OPEN_ENDED = ["q23", "q24", "q25"]

# Logical order of answer categories (used for tables and charts)
ORDER = {
    "q6": ["Less than 1 hour", "1 - 2 hours", "3 - 4 hours", "5 - 6 hours", "More than 6 hours"],
    "q7": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q10": ["Less than 1 hour", "1 – 2 hours", "3 – 4 hours", "More than 4 hours"],
    "q11": ["On my study desk within arm's reach (screen facing up)",
            "On my study desk (screen facing down)",
            "In the same room, but out of reach",
            "In another room / Completely turned off / Silent mode"],
    "q12": ["Every few minutes (1–10 mins)", "Every 15–30 minutes",
            "Every hour / Only during scheduled breaks", "Rarely / Almost never"],
    "q13": ["Immediately (No loss of focus)", "1–5 minutes", "5–15 minutes",
            "More than 15 minutes / I get completely sidetracked"],
    "q14": ["Excellent", "Good", "Fair", "Poor"],
    "q15": ["Positive impact (helps my studies)", "No noticeable effect",
            "Slight negative impact", "Significant negative impact"],
    "q16": ["Yes", "Maybe", "No"],
    "q18": ["Never", "Rarely", "Sometimes", "Often", "Always"],
    "q20": ["Strongly agree", "Agree", "Neutral", "Disagree", "Strongly disagree"],
}

# Numeric codes for ordered variables (needed for Spearman's correlation)
CODES = {
    "q6": {"Less than 1 hour": 1, "1 - 2 hours": 2, "3 - 4 hours": 3, "5 - 6 hours": 4, "More than 6 hours": 5},
    "q7": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4},
    "q13": {"Immediately (No loss of focus)": 0, "1–5 minutes": 1, "5–15 minutes": 2,
            "More than 15 minutes / I get completely sidetracked": 3},
    "q15": {"Positive impact (helps my studies)": 1, "No noticeable effect": 2,
            "Slight negative impact": 3, "Significant negative impact": 4},
    "q16": {"No": 0, "Maybe": 1, "Yes": 2},
}

# Shorter labels for chart axes only (the data itself is not changed)
SHORT_LABELS = {
    "More than 15 minutes / I get completely sidetracked": "More than 15 min / sidetracked",
    "Immediately (No loss of focus)": "Immediately (no loss of focus)",
}

# Chart colours
BLUE = "#2a78d6"
IMPACT_COLOURS = ["#2a78d6", "#a8a8a3", "#f4a07a", "#c4471a"]   # positive -> significant negative
YES_MAYBE_NO = ["#c4471a", "#f4a07a", "#2a78d6"]


# ----------------------------------------------------------------------------
# 2. Load and clean the data
# ----------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df.columns = ["timestamp"] + [f"q{i}" for i in range(1, 26)]

    # Remove extra spaces from every text answer
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()

    # Treat empty open-ended answers as missing
    df[OPEN_ENDED] = df[OPEN_ENDED].replace("", pd.NA)

    # Convert the Google Forms timestamp, e.g. "2026/09/17 9:55:17 PM GMT+5:30"
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.replace(" GMT+5:30", "", regex=False),
                                     format="%Y/%m/%d %I:%M:%S %p", errors="coerce")
    return df


def split_multi(series):
    """Split a multiple-response answer ('A;B;C') into one row per option."""
    options = series.dropna().str.split(";").explode().str.strip()
    return options[options != ""]


# ----------------------------------------------------------------------------
# 3. Frequencies and percentages
# ----------------------------------------------------------------------------
def frequency_table(df, q):
    n = len(df)
    values = split_multi(df[q]) if q in MULTI_RESPONSE else df[q].dropna()
    counts = values.value_counts()
    if q in ORDER:  # put categories in their logical order, keep any extra ones at the end
        order = [c for c in ORDER[q] if c in counts.index] + [c for c in counts.index if c not in ORDER[q]]
        counts = counts.reindex(order)
    table = pd.DataFrame({"Frequency": counts, "Percentage (%)": (counts / n * 100).round(1)})
    table.index.name = QUESTION_NAMES[q]
    return table


# ----------------------------------------------------------------------------
# 4. Cross-tabulation and Spearman's rank correlation
# ----------------------------------------------------------------------------
def cross_tab(df, row_q, col_q):
    table = pd.crosstab(df[row_q], df[col_q], margins=True, margins_name="Total")
    rows = [c for c in ORDER.get(row_q, []) if c in table.index] or [i for i in table.index if i != "Total"]
    cols = [c for c in ORDER.get(col_q, []) if c in table.columns] or [c for c in table.columns if c != "Total"]
    table = table.reindex(index=rows + ["Total"], columns=cols + ["Total"]).fillna(0).astype(int)
    table.index.name, table.columns.name = QUESTION_NAMES[row_q], QUESTION_NAMES[col_q]
    return table


def spearman(df, q_a, q_b):
    a, b = df[q_a].map(CODES[q_a]), df[q_b].map(CODES[q_b])
    valid = a.notna() & b.notna()
    rho, p = stats.spearmanr(a[valid], b[valid])
    return rho, p, int(valid.sum())


# ----------------------------------------------------------------------------
# 5. Charts
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def bar_chart(table, n, filename, colours=None):
    """Horizontal bar chart showing count and percentage for each answer."""
    counts = table["Frequency"]
    labels = [SHORT_LABELS.get(i, i) for i in counts.index]
    fig, ax = plt.subplots(figsize=(6.5, 0.36 * len(counts) + 0.8))
    bars = ax.barh(labels[::-1], counts.values[::-1], color=(colours[::-1] if colours else BLUE), height=0.6)
    for bar, v in zip(bars, counts.values[::-1]):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2, f"{int(v)} ({v / n * 100:.0f}%)", va="center", fontsize=10)
    ax.set_xlabel("Number of respondents")
    ax.set_xlim(0, counts.max() * 1.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", color="#e5e5e5"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, filename), dpi=200); plt.close(fig)


def stacked_chart(df, row_q, col_q, colours, filename, ylabel, legend_labels=None):
    """Stacked bar chart of a cross-tabulation (every respondent is counted once)."""
    table = cross_tab(df, row_q, col_q).drop(index="Total", columns="Total")
    table = table.iloc[::-1]  # first category at the top
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(table) + 1.4))
    left = pd.Series(0, index=table.index)
    y_labels = [SHORT_LABELS.get(i, i) for i in table.index]
    for col, colour in zip(table.columns, colours):
        ax.barh(y_labels, table[col], left=left, color=colour, height=0.6, edgecolor="white", linewidth=1.5,
                label=(legend_labels or {}).get(col, col))
        for i, (start, v) in enumerate(zip(left, table[col])):
            if v > 0:
                ax.text(start + v / 2, i, int(v), ha="center", va="center", fontsize=9,
                        color="#222" if colour in ("#a8a8a3", "#f4a07a") else "white")
        left = left + table[col]
    for i, total in enumerate(left):
        ax.text(total + 0.2, i, f"n={int(total)}", va="center", fontsize=9, color="#555")
    ax.set_xlabel("Number of respondents"); ax.set_ylabel(ylabel)
    ax.set_xlim(0, left.max() * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.25))
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, filename), dpi=200, bbox_inches="tight"); plt.close(fig)
    assert int(left.sum()) == len(df), "stacked chart does not include every respondent"


# ----------------------------------------------------------------------------
# 6. Main program
# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = glob.glob("*.csv")
        if not files:
            sys.exit("No CSV file found. Usage: python analyze_survey.py responses.csv")
        path = files[0]

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

    df = load_data(path)
    n = len(df)
    lines = [f"Survey analysis of: {os.path.basename(path)}",
             f"Number of responses: {n}",
             f"Collection period: {df['timestamp'].min():%d %B %Y} to {df['timestamp'].max():%d %B %Y}", ""]

    # --- Frequencies for every closed question --------------------------------
    lines.append("=" * 70 + "\nFREQUENCIES AND PERCENTAGES (base = all respondents)\n" + "=" * 70)
    tables = {}
    for i in range(1, 23):
        q = f"q{i}"
        t = frequency_table(df, q)
        tables[q] = t
        t.to_csv(os.path.join(TAB_DIR, f"Q{i:02d}_frequencies.csv"), encoding="utf-8-sig")
        note = "  (multiple responses - percentages may exceed 100%)" if q in MULTI_RESPONSE else ""
        lines += [f"\nQ{i}. {QUESTION_NAMES[q]}{note}", t.to_string()]

    # --- Key combined figures quoted in the report ----------------------------
    def share(q, answers):
        k = df[q].isin(answers).sum()
        return f"{k} ({k / n * 100:.0f}%)"
    lines += ["", "=" * 70, "KEY COMBINED FIGURES", "=" * 70,
              f"Use social media 3+ hours a day:          {share('q6', ['3 - 4 hours', '5 - 6 hours', 'More than 6 hours'])}",
              f"Use social media 5+ hours a day:          {share('q6', ['5 - 6 hours', 'More than 6 hours'])}",
              f"Study 3+ hours a day:                     {share('q10', ['3 – 4 hours', 'More than 4 hours'])}",
              f"Check social media often/always:          {share('q7', ['Often', 'Always'])}",
              f"Keep phone on study desk:                 {share('q11', ORDER['q11'][:2])}",
              f"Check phone at least every 30 minutes:    {share('q12', ORDER['q12'][:2])}",
              f"Need 5+ minutes to refocus:               {share('q13', ORDER['q13'][2:])}",
              f"Perceive a negative effect:               {share('q15', ORDER['q15'][2:])}",
              f"Stay up late often/always:                {share('q18', ['Often', 'Always'])}",
              f"Agree reducing use would help:            {share('q20', ['Strongly agree', 'Agree'])}"]

    # --- Cross-tabulations ----------------------------------------------------
    lines += ["", "=" * 70, "CROSS-TABULATIONS", "=" * 70]
    for row_q, col_q in [("q6", "q15"), ("q13", "q16"), ("q11", "q12"), ("q19", "q15")]:
        t = cross_tab(df, row_q, col_q)
        t.to_csv(os.path.join(TAB_DIR, f"crosstab_{row_q}_by_{col_q}.csv"), encoding="utf-8-sig")
        lines += [f"\n{QUESTION_NAMES[row_q]}  x  {QUESTION_NAMES[col_q]}", t.to_string()]

    # --- Spearman's rank correlation ------------------------------------------
    lines += ["", "=" * 70, "SPEARMAN'S RANK CORRELATION (p < 0.05 = statistically significant)", "=" * 70]
    for q_a, q_b in [("q6", "q15"), ("q13", "q16"), ("q7", "q15"), ("q13", "q15")]:
        rho, p, k = spearman(df, q_a, q_b)
        verdict = "significant" if p < 0.05 else "not significant"
        lines.append(f"{QUESTION_NAMES[q_a]} vs {QUESTION_NAMES[q_b]}: rho = {rho:.2f}, p = {p:.3f}, n = {k}  -> {verdict}")

    # --- Figures used in the report -------------------------------------------
    bar_chart(tables["q5"], n, "fig01_platforms.png")
    bar_chart(tables["q6"], n, "fig02_daily_hours.png")
    bar_chart(tables["q7"], n, "fig03_check_frequency.png")
    bar_chart(tables["q8"].sort_values("Frequency", ascending=False), n, "fig04_reasons.png")
    bar_chart(tables["q9"].sort_values("Frequency", ascending=False), n, "fig05_triggers.png")
    bar_chart(tables["q11"], n, "fig06_phone_placement.png")
    bar_chart(tables["q13"], n, "fig07_time_to_refocus.png")
    bar_chart(tables["q15"], n, "fig08_perceived_effect.png", colours=IMPACT_COLOURS)
    stacked_chart(df, "q6", "q15", IMPACT_COLOURS, "fig09_hours_vs_effect.png", "Daily social media use")
    stacked_chart(df, "q13", "q16", YES_MAYBE_NO, "fig10_refocus_vs_results.png", "Time to refocus",
                  legend_labels={"Yes": "Yes, results have suffered"})
    bar_chart(tables["q22"].sort_values("Frequency", ascending=False), n, "fig11_strategies.png")

    # --- Open-ended answers (grouped into themes by hand) ---------------------
    with open(os.path.join(OUT_DIR, "open_ended_answers.txt"), "w", encoding="utf-8") as f:
        for q in OPEN_ENDED:
            answers = df[q].dropna()
            f.write(f"{q.upper()}. {QUESTION_NAMES[q]}  ({len(answers)} answers)\n" + "-" * 60 + "\n")
            f.writelines(f"- {a}\n" for a in answers)
            f.write("\n")

    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nDone. Results saved in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
