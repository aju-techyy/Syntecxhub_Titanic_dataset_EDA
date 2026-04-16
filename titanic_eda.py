import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load titanic dataset directly from seaborn's built-in datasets
df = sns.load_dataset("titanic")

# ------------------------------------------------------------------
# 1. basic inspection
# ------------------------------------------------------------------
print("shape:", df.shape)
print("\ndtypes:\n", df.dtypes)
print("\nmissing values:\n", df.isnull().sum())
print("\nbasic stats:\n", df.describe())

# ------------------------------------------------------------------
# 2. survival rates by sex, class, and age bucket
# ------------------------------------------------------------------

# survival rate by sex
survival_sex = df.groupby("sex")["survived"].mean().reset_index()
survival_sex.columns = ["sex", "survival_rate"]
print("\nsurvival rate by sex:\n", survival_sex)

# survival rate by passenger class
survival_class = df.groupby("pclass")["survived"].mean().reset_index()
survival_class.columns = ["pclass", "survival_rate"]
print("\nsurvival rate by class:\n", survival_class)

# age buckets
df["age_bucket"] = pd.cut(
    df["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["child", "teen", "young adult", "adult", "senior"]
)
survival_age = df.groupby("age_bucket", observed=True)["survived"].mean().reset_index()
survival_age.columns = ["age_bucket", "survival_rate"]
print("\nsurvival rate by age bucket:\n", survival_age)

# ------------------------------------------------------------------
# 3. visualizations
# ------------------------------------------------------------------

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic EDA - Survival Analysis", fontsize=15)

# bar chart: survival rate by sex
axes[0, 0].bar(survival_sex["sex"], survival_sex["survival_rate"], color=["steelblue", "salmon"])
axes[0, 0].set_title("survival rate by sex")
axes[0, 0].set_ylabel("survival rate")
axes[0, 0].set_ylim(0, 1)

# bar chart: survival rate by class
axes[0, 1].bar(
    survival_class["pclass"].astype(str),
    survival_class["survival_rate"],
    color=["gold", "silver", "#cd7f32"]
)
axes[0, 1].set_title("survival rate by passenger class")
axes[0, 1].set_xlabel("class")
axes[0, 1].set_ylabel("survival rate")
axes[0, 1].set_ylim(0, 1)

# bar chart: survival rate by age bucket
axes[0, 2].bar(
    survival_age["age_bucket"].astype(str),
    survival_age["survival_rate"],
    color="mediumpurple"
)
axes[0, 2].set_title("survival rate by age bucket")
axes[0, 2].set_xlabel("age group")
axes[0, 2].set_ylabel("survival rate")
axes[0, 2].set_ylim(0, 1)
axes[0, 2].tick_params(axis="x", rotation=20)

# boxplot: age distribution by survival status
sns.boxplot(
    data=df, x="survived", y="age", palette="Set2", ax=axes[1, 0]
)
axes[1, 0].set_title("age distribution by survival")
axes[1, 0].set_xticklabels(["did not survive", "survived"])

# violin plot: age by class and survival
sns.violinplot(
    data=df, x="pclass", y="age", hue="survived",
    split=True, palette="muted", ax=axes[1, 1]
)
axes[1, 1].set_title("age distribution by class and survival")
axes[1, 1].set_xlabel("passenger class")
axes[1, 1].legend(title="survived", labels=["no", "yes"])

# count plot: survival count by sex and class
sns.countplot(
    data=df, x="pclass", hue="survived", palette="pastel", ax=axes[1, 2]
)
axes[1, 2].set_title("survival count by class")
axes[1, 2].set_xlabel("passenger class")
axes[1, 2].legend(title="survived", labels=["no", "yes"])

plt.tight_layout()
plt.savefig("titanic_eda_plots.png", dpi=150)
plt.show()
print("\nplots saved to titanic_eda_plots.png")

# ------------------------------------------------------------------
# 4. insight report
# ------------------------------------------------------------------
print("\n--- insight report ---")
print("- females had a much higher survival rate than males (~74% vs ~19%)")
print("- 1st class passengers survived at nearly double the rate of 3rd class passengers")
print("- children had the highest survival rate among age groups, seniors the lowest")
print("- across all classes, age distributions of survivors skew slightly younger")
print("- 3rd class had the largest passenger count but the lowest survival proportion")