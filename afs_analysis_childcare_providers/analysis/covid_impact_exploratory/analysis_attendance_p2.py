# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import afs_analysis_childcare_providers

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# %%
project_directory = afs_analysis_childcare_providers.PROJECT_DIR

# %%
# Weekly attendance
wa = pd.read_csv(
    f"{project_directory}/inputs/data/table_5_weekly_attendance_in_early_years_settings_during_the_covid-19_outbreak_at_local_authority_level_.csv"
)

# Population projections
uk_pop = pd.read_excel(
    f"{project_directory}/inputs/data/ukpopestimatesmid2020on2021geography.xls",
    skiprows=7,
    sheet_name="MYE2 - Persons",
)

# IMD
imd = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="IMD",
)


# IMD IDACI
idaci = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="IDACI",
)

# Health
health = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="Health",
)

# Income
income = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="Income",
)

# Covid cases - Specimen Date
new_cases = pd.read_csv(
    "https://api.coronavirus.data.gov.uk/v2/data?areaType=utla&metric=newCasesBySpecimenDate&format=csv"
)

# %% [markdown]
# Clean weekly attendance and population indicators

# %%
wa["total_children_in_early_years_settings"].replace(
    ["c"], [0], inplace=True
)  # recode c to 0
wa["total_children_in_early_years_settings"] = wa[
    "total_children_in_early_years_settings"
].astype(int)

# Aged 0 to 5 population
uk_pop["EY_pop"] = (
    uk_pop["0"] + uk_pop["1"] + uk_pop["2"] + uk_pop["3"] + uk_pop["4"] + uk_pop["5"]
)

wa.rename(columns={"new_la_code": "Code"}, inplace=True)  # Rename to match uk_pop

# %%
uk_pop["Name"] = uk_pop["Name"].str.strip()
uk_pop["Geography"] = uk_pop["Geography"].str.strip()

# Combining North and West Northamptonshire
uk_pop["Code"] = uk_pop["Code"].replace({"E06000062": "E06000061"})
uk_pop["Name"] = uk_pop["Name"].replace(
    {"West Northamptonshire": "North Northamptonshire"}
)
uk_pop = uk_pop.groupby(["Code", "Name", "Geography"]).sum().reset_index()
uk_pop["Name"] = uk_pop["Name"].replace({"North Northamptonshire": "Northamptonshire"})
uk_pop["Code"] = uk_pop["Code"].replace({"E06000061": "E10000021"})

wa["la_name"] = wa["la_name"].replace(
    {"West Northamptonshire": "North Northamptonshire"}
)
wa["Code"] = wa["Code"].replace({"E06000062": "E06000061"})
wa.drop(["old_la_code"], axis=1, inplace=True)
wa = (
    wa.groupby(["date", "la_name", "Code"])
    .agg(
        {
            "time_period": "first",
            "time_identifier": "first",
            "geographic_level": "first",
            "country_code": "first",
            "region_code": "first",
            "region_name": "first",
            "total_early_years_settings_reported_by_la": "sum",
            "count_of_open_early_years_settings": "sum",
            "count_of_closed_early_years_settings": "sum",
            "total_children_in_early_years_settings": "sum",
            "total_vulnerable_children_in_early_years_settings": "sum",
        }
    )
    .reset_index()
)
wa["la_name"] = wa["la_name"].replace({"North Northamptonshire": "Northamptonshire"})
wa["Code"] = wa["Code"].replace({"E06000061": "E10000021"})

# Change Buckinghamshire from E06000060 to E10000002
wa["Code"] = wa["Code"].replace({"E06000060": "E10000002"})
uk_pop["Code"] = uk_pop["Code"].replace({"E06000060": "E10000002"})

# %% [markdown]
# Clean IMD datasets

# %%
# Remove leading and trailing whitespace in headers
idaci.rename(columns=lambda x: x.strip(), inplace=True)
health.rename(columns=lambda x: x.strip(), inplace=True)
income.rename(columns=lambda x: x.strip(), inplace=True)
imd.rename(columns=lambda x: x.strip(), inplace=True)

# Rename code to match wa df
idaci.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)
health.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)
income.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)
imd.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)

# %% [markdown]
# Group by la-month and merge

# %%
wa["date"] = pd.to_datetime(wa["date"], format="%d/%m/%Y")
wa.set_index("date", inplace=True, drop=True)

# %%
wa_m = (
    wa[wa.columns.difference(["time_period"])]
    .groupby([pd.Grouper(freq="M"), "la_name", "Code", "region_name"])
    .mean()
    .reset_index()
)
# Merge with wa df
wa_m = wa_m.merge(
    idaci[
        idaci.columns.difference(["Upper Tier Local Authority District name (2019)"])
    ],
    how="left",
    on="Code",
)
wa_m = wa_m.merge(
    health[
        health.columns.difference(["Upper Tier Local Authority District name (2019)"])
    ],
    how="left",
    on="Code",
)
wa_m = wa_m.merge(
    income[
        income.columns.difference(["Upper Tier Local Authority District name (2019)"])
    ],
    how="left",
    on="Code",
)
wa_m = wa_m.merge(
    imd[imd.columns.difference(["Upper Tier Local Authority District name (2019)"])],
    how="left",
    on="Code",
)


# %%
def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


# %%
wa_m["IMD_tiers_10"] = pct_rank_qcut(wa_m["IMD - Rank of average score"], 10)
wa_m["IDACI_tiers_10"] = pct_rank_qcut(wa_m["IDACI - Rank of average score"], 10)

wa_m["lsoa_tiers_10"] = pct_rank_qcut(
    wa_m["IDACI - Rank of proportion of LSOAs in most deprived 10% nationally"], 10
)

wa_m["Income_tiers_10"] = pct_rank_qcut(wa_m["Income - Rank of average score"], 10)
wa_m["Health_tiers_10"] = pct_rank_qcut(
    wa_m["Health Deprivation and Disability - Rank of average score"], 10
)

# %%
# Merge pop and wa datasets
wa_m = wa_m.merge(uk_pop[["Code", "EY_pop", "Geography"]], how="left", on="Code")

# %%
area_df = wa_m[
    [
        "la_name",
        "Code",
        "region_name",
        "IMD - Rank of average score",
        "Geography",
        "EY_pop",
    ]
].copy()

# %%
area_df.shape

# %%
area_df.drop_duplicates(inplace=True)

# %%
area_df.shape

# %%
area_df.to_excel("area_df.xlsx", index=False)

# %%
wa_m.head(1)

# %%
wa_m.set_index("date", inplace=True, drop=True)

# %% [markdown]
# ### Avg total EYS

# %%
totalEYS = wa_m.groupby([pd.Grouper(freq="M")])[
    ["total_early_years_settings_reported_by_la"]
].mean()

# %%
plt.plot(
    totalEYS.index,
    totalEYS["total_early_years_settings_reported_by_la"],
)

plt.title("Average number of EYS monthly reported - England")
plt.xlabel("Month")
plt.ylabel("Avg EYS")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/avg_eys.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# ### EYS open

# %%
total = wa_m.groupby([pd.Grouper(freq="M")])[
    ["count_of_open_early_years_settings", "total_early_years_settings_reported_by_la"]
].sum()
total["Percent open"] = (
    total["count_of_open_early_years_settings"]
    / total["total_early_years_settings_reported_by_la"]
) * 100

# %%
plt.plot(
    total.index,
    total["Percent open"],
)

plt.title("Percent open EYS - England")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_eys.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
# IMD
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "IMD_tiers_10"])[
        [
            "count_of_open_early_years_settings",
            "total_early_years_settings_reported_by_la",
        ]
    ]
    .sum()
    .unstack("IMD_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent open", col] = (
        tier10["count_of_open_early_years_settings", col]
        / tier10["total_early_years_settings_reported_by_la", col]
    ) * 100

plt.plot(
    tier10.index,
    tier10["Percent open"][1],
)
plt.plot(
    total.index,
    total["Percent open"],
)
plt.plot(
    tier10.index,
    tier10["Percent open"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent open - IMD top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
# IDACI
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "IDACI_tiers_10"])[
        [
            "count_of_open_early_years_settings",
            "total_early_years_settings_reported_by_la",
        ]
    ]
    .sum()
    .unstack("IDACI_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent open", col] = (
        tier10["count_of_open_early_years_settings", col]
        / tier10["total_early_years_settings_reported_by_la", col]
    ) * 100

plt.plot(
    tier10.index,
    tier10["Percent open"][1],
)
plt.plot(
    total.index,
    total["Percent open"],
)
plt.plot(
    tier10.index,
    tier10["Percent open"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent open - IDACI top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_idaci_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
# Health
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "Health_tiers_10"])[
        [
            "count_of_open_early_years_settings",
            "total_early_years_settings_reported_by_la",
        ]
    ]
    .sum()
    .unstack("Health_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent open", col] = (
        tier10["count_of_open_early_years_settings", col]
        / tier10["total_early_years_settings_reported_by_la", col]
    ) * 100

plt.plot(
    tier10.index,
    tier10["Percent open"][1],
)
plt.plot(
    total.index,
    total["Percent open"],
)
plt.plot(
    tier10.index,
    tier10["Percent open"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent open - Health top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_health_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
# Income
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "Income_tiers_10"])[
        [
            "count_of_open_early_years_settings",
            "total_early_years_settings_reported_by_la",
        ]
    ]
    .sum()
    .unstack("Income_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent open", col] = (
        tier10["count_of_open_early_years_settings", col]
        / tier10["total_early_years_settings_reported_by_la", col]
    ) * 100

plt.plot(
    tier10.index,
    tier10["Percent open"][1],
)
plt.plot(
    total.index,
    total["Percent open"],
)
plt.plot(
    tier10.index,
    tier10["Percent open"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent open - Income top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_income_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
# LSOAs
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "lsoa_tiers_10"])[
        [
            "count_of_open_early_years_settings",
            "total_early_years_settings_reported_by_la",
        ]
    ]
    .sum()
    .unstack("lsoa_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent open", col] = (
        tier10["count_of_open_early_years_settings", col]
        / tier10["total_early_years_settings_reported_by_la", col]
    ) * 100

plt.plot(
    tier10.index,
    tier10["Percent open"][1],
)
plt.plot(
    total.index,
    total["Percent open"],
)
plt.plot(
    tier10.index,
    tier10["Percent open"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent open - LSOAs top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent open")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/open_lsoa_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# ### Attendance

# %% [markdown]
# IMD 10 tiers

# %%
total = wa_m.groupby([pd.Grouper(freq="M")])[
    ["total_children_in_early_years_settings", "EY_pop"]
].sum()
total["Percent attending"] = (
    total["total_children_in_early_years_settings"] / total["EY_pop"]
) * 100

# %%
plt.plot(
    total.index,
    total["Percent attending"],
)

plt.title("Avg attendance EYS - England")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attend_eys.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "IMD_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IMD_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
heat_imd = tier10["Percent attending"].transpose()
fig, ax = plt.subplots(figsize=(14, 8))  # Sample figsize in inches

ax = sns.heatmap(
    heat_imd,
    cmap="YlGnBu_r",
    vmin=1,
    vmax=30,
    annot=True,
    annot_kws={"style": "italic", "weight": "bold"},
    xticklabels=tier10.index.strftime("%Y-%m-%d"),
)

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/imd10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent attending - IMD top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attending_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# IDACI 10 tiers

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "IDACI_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IDACI_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
tier10["Percent attending"].head(2)

# %%
heat_idaci = tier10["Percent attending"].transpose()

# %%
fig, ax = plt.subplots(figsize=(14, 8))  # Sample figsize in inches

ax = sns.heatmap(
    heat_idaci,
    cmap="YlGnBu_r",
    vmin=1,
    vmax=30,
    annot=True,
    annot_kws={"style": "italic", "weight": "bold"},
    xticklabels=tier10.index.strftime("%Y-%m-%d"),
)

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/idaci10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent attending - IDACI top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attending_idaci_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# lsoa 10 tiers

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "lsoa_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("lsoa_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
tier10["Percent attending"].head(2)

# %%
heat_idaci_lsoa = tier10["Percent attending"].transpose()

# %%
fig, ax = plt.subplots(figsize=(14, 8))  # Sample figsize in inches

ax = sns.heatmap(
    heat_idaci_lsoa,
    cmap="YlGnBu_r",
    vmin=1,
    vmax=30,
    annot=True,
    annot_kws={"style": "italic", "weight": "bold"},
    xticklabels=tier10.index.strftime("%Y-%m-%d"),
)

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/idacilsoa10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent attending - LSOA's deprived top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attending_isoa_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# Income 10 tiers

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "Income_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("Income_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
tier10["Percent attending"].head(2)

# %%
heat_income = tier10["Percent attending"].transpose()

# %%
fig, ax = plt.subplots(figsize=(14, 8))  # Sample figsize in inches

ax = sns.heatmap(
    heat_income,
    cmap="YlGnBu_r",
    vmin=1,
    vmax=30,
    annot=True,
    annot_kws={"style": "italic", "weight": "bold"},
    xticklabels=tier10.index.strftime("%Y-%m-%d"),
)


ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/income10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent attending - Income top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attending_income_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# Health 10 tiers

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "Health_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("Health_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
tier10["Percent attending"].head(2)

# %%
heat_income = tier10["Percent attending"].transpose()

# %%
fig, ax = plt.subplots(figsize=(14, 8))  # Sample figsize in inches

ax = sns.heatmap(
    heat_income,
    cmap="YlGnBu_r",
    vmin=1,
    vmax=30,
    annot=True,
    annot_kws={"style": "italic", "weight": "bold"},
    xticklabels=tier10.index.strftime("%Y-%m-%d"),
)


ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/health10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("Percent attending - Health top, bottom and average")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/attending_health_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# London - IMD 10 tiers

# %%
london = wa_m[wa_m["region_name"] == "London"].copy()

# %%
london["LDN_IMD_tiers_10"] = pct_rank_qcut(london["IMD - Rank of average score"], 10)

# %%
total = london.groupby([pd.Grouper(freq="M")])[
    ["total_children_in_early_years_settings", "EY_pop"]
].sum()
total["Percent attending"] = (
    total["total_children_in_early_years_settings"] / total["EY_pop"]
) * 100

# %%
tier10 = (
    london.groupby([pd.Grouper(freq="M"), "LDN_IMD_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("LDN_IMD_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("Most deprived 10%", "Avg London", "Least deprived 10%"))

plt.title("London: Percent attending EYS")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/London_attending_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# East of England - IMD 10 tiers

# %%
eoe = wa_m[wa_m["region_name"] == "East of England"].copy()

# %%
eoe["EOE_IMD_tiers_10"] = pct_rank_qcut(eoe["IMD - Rank of average score"], 10)

# %%
total = eoe.groupby([pd.Grouper(freq="M")])[
    ["total_children_in_early_years_settings", "EY_pop"]
].sum()
total["Percent attending"] = (
    total["total_children_in_early_years_settings"] / total["EY_pop"]
) * 100

# %%
tier10 = (
    eoe.groupby([pd.Grouper(freq="M"), "EOE_IMD_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("EOE_IMD_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("Most deprived 10%", "Avg East of England", "Least deprived 10%"))

plt.title("East of England: Percent attending EYS")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/EOE_attending_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# North East - IMD 10 tiers

# %%
eoe = wa_m[wa_m["region_name"] == "Yorkshire and The Humber"].copy()

# %%
eoe["EOE_IMD_tiers_10"] = pct_rank_qcut(eoe["IMD - Rank of average score"], 10)

# %%
total = eoe.groupby([pd.Grouper(freq="M")])[
    ["total_children_in_early_years_settings", "EY_pop"]
].sum()
total["Percent attending"] = (
    total["total_children_in_early_years_settings"] / total["EY_pop"]
) * 100

# %%
tier10 = (
    eoe.groupby([pd.Grouper(freq="M"), "EOE_IMD_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("EOE_IMD_tiers_10")
)

tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    total.index,
    total["Percent attending"],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(
    ("Most deprived 10%", "Avg Yorkshire and The Humber", "Least deprived 10%")
)

plt.title("Yorkshire and The Humber: Percent attending EYS")
plt.xlabel("Month")
plt.ylabel("Percent attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/YH_attending_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%

# %%
wa_m["avg_daily_perc_att"] = (
    wa_m["total_children_in_early_years_settings"] / wa_m["EY_pop"]
) * 100

# %%
wa_m.reset_index(inplace=True)

# %% [markdown]
# ### Covid cases

# %%
new_cases = pd.read_csv(
    "https://api.coronavirus.data.gov.uk/v2/data?areaType=utla&metric=newCasesBySpecimenDate&format=csv"
)

# %%
new_cases["date"] = pd.to_datetime(new_cases["date"], format="%Y-%m-%d")
new_cases.set_index("date", inplace=True, drop=True)
new_cases_m = (
    new_cases.groupby([pd.Grouper(freq="M"), "areaCode", "areaName"])
    .mean()
    .reset_index()
)
new_cases_m.rename(columns={"areaCode": "Code"}, inplace=True)  # Rename to match uk_pop
wa_mc = wa_m[
    [
        "la_name",
        "date",
        "Code",
        "IMD_tiers_10",
        "IDACI_tiers_10",
        "lsoa_tiers_10",
        "Income_tiers_10",
        "Health_tiers_10",
        "EY_pop",
        "total_children_in_early_years_settings",
        "avg_daily_perc_att",
    ]
].merge(
    new_cases_m[["date", "Code", "newCasesBySpecimenDate"]],
    how="left",
    on=["date", "Code"],
)

# %%
temp_drop = ["Cornwall", "Isles Of Scilly", "Hackney", "City of London"]
wa_mc = wa_mc[~wa_mc.la_name.isin(temp_drop)]

# %%
# Rate per 100,000 population
wa_mc["New cases per 100,000"] = wa_mc["newCasesBySpecimenDate"] / (
    wa_mc["EY_pop"] / 100000
)

# %%
wa_mc.set_index("date", inplace=True, drop=True)

# %%
wa_mc.head(1)

# %%

# %% [markdown]
# IMD tiers - covid rates

# %%
total = wa_mc.groupby([pd.Grouper(freq="M")])[
    ["newCasesBySpecimenDate", "EY_pop"]
].sum()
total["New cases per 100,000"] = total["newCasesBySpecimenDate"] / (
    total["EY_pop"] / 100000
)

# %%
tier10 = (
    wa_mc.groupby([pd.Grouper(freq="M"), "IMD_tiers_10"])[
        [
            "newCasesBySpecimenDate",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IMD_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["New cases per 100,000", col] = tier10["newCasesBySpecimenDate", col] / (
        tier10["EY_pop", col] / 100000
    )

# %%
tier10["New cases per 100,000"].head(2)

# %%
tier10["New cases per 100,000"].to_excel("tempcovid.xlsx")

# %%
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][1],
)
plt.plot(
    total.index,
    total["New cases per 100,000"],
)
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("New covid cases - IMD top, bottom and average")
plt.xlabel("Month")
plt.ylabel("New covid cases")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_imd_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# IMD income tiers - covid rates

# %%
total = wa_mc.groupby([pd.Grouper(freq="M")])[
    ["newCasesBySpecimenDate", "EY_pop"]
].sum()
total["New cases per 100,000"] = total["newCasesBySpecimenDate"] / (
    total["EY_pop"] / 100000
)

# %%
tier10 = (
    wa_mc.groupby([pd.Grouper(freq="M"), "Income_tiers_10"])[
        [
            "newCasesBySpecimenDate",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("Income_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["New cases per 100,000", col] = tier10["newCasesBySpecimenDate", col] / (
        tier10["EY_pop", col] / 100000
    )

# %%
tier10["New cases per 100,000"].head(2)

# %%
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][1],
)
plt.plot(
    total.index,
    total["New cases per 100,000"],
)
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("New covid cases - IMD income top, bottom and average")
plt.xlabel("Month")
plt.ylabel("New covid cases")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_income_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%

# %% [markdown]
# IMD IDACI tiers - covid rates

# %%
tier10 = (
    wa_mc.groupby([pd.Grouper(freq="M"), "IDACI_tiers_10"])[
        [
            "newCasesBySpecimenDate",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IDACI_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["New cases per 100,000", col] = tier10["newCasesBySpecimenDate", col] / (
        tier10["EY_pop", col] / 100000
    )

# %%
tier10["New cases per 100,000"].head(2)

# %%
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][1],
)
plt.plot(
    total.index,
    total["New cases per 100,000"],
)
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("New covid cases - IMD IDACI top, bottom and average")
plt.xlabel("Month")
plt.ylabel("New covid cases")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_idaci_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%

# %% [markdown]
# IMD lsoa tiers - covid rates

# %%
tier10 = (
    wa_mc.groupby([pd.Grouper(freq="M"), "lsoa_tiers_10"])[
        [
            "newCasesBySpecimenDate",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("lsoa_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["New cases per 100,000", col] = tier10["newCasesBySpecimenDate", col] / (
        tier10["EY_pop", col] / 100000
    )

# %%
tier10["New cases per 100,000"].head(2)

# %%
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][1],
)
plt.plot(
    total.index,
    total["New cases per 100,000"],
)
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("New covid cases - LSOA's deprived top, bottom and average")
plt.xlabel("Month")
plt.ylabel("New covid cases")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_lsoa_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%

# %% [markdown]
# IMD health tiers - covid rates

# %%
tier10 = (
    wa_mc.groupby([pd.Grouper(freq="M"), "Health_tiers_10"])[
        [
            "newCasesBySpecimenDate",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("Health_tiers_10")
)

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["New cases per 100,000", col] = tier10["newCasesBySpecimenDate", col] / (
        tier10["EY_pop", col] / 100000
    )

# %%
tier10["New cases per 100,000"].head(1)

# %%
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][1],
)
plt.plot(
    total.index,
    total["New cases per 100,000"],
)
plt.plot(
    tier10.index,
    tier10["New cases per 100,000"][10],
)

plt.gca().legend(("IMD 1", "Avg", "IMD 10"))

plt.title("New covid cases - IMD health top, bottom and average")
plt.xlabel("Month")
plt.ylabel("New covid cases")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_health_avg.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%

# %%
fig = plt.figure(figsize=(6, 6))
cases_att = plt.scatter(
    wa_mc["New cases per 100,000"], wa_mc["avg_daily_perc_att"], color="#ff0000"
)
plt.xlabel("New cases per 100,000")
plt.ylabel("% EYS attendance")

plt.title("New cases per 100,000 population verses attendance rate", fontsize=12)

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_v_attend_scatter.jpg"
)

plt.show()

# %%

# %%

# %%
wa_mc.head(2)

# %%
total = wa_mc.groupby([pd.Grouper(freq="M")])[
    ["newCasesBySpecimenDate", "EY_pop", "total_children_in_early_years_settings"]
].sum()

# %%
total["New cases per 100,000"] = total["newCasesBySpecimenDate"] / (
    total["EY_pop"] / 100000
)
total["Percent attending"] = (
    total["total_children_in_early_years_settings"] / total["EY_pop"]
) * 100

# %%
total.drop(
    ["newCasesBySpecimenDate", "EY_pop", "total_children_in_early_years_settings"],
    axis=1,
    inplace=True,
)

# %%
total.head(1)

# %%
from sklearn.preprocessing import MinMaxScaler

total_scaled = total.copy()

scaler = MinMaxScaler()

for col in total_scaled.columns:
    scaler = MinMaxScaler()
    total_scaled[col] = scaler.fit_transform(total_scaled[col].values.reshape(-1, 1))

# %%
total_scaled.head(1)

# %%
total_scaled.plot()

plt.title("New covid cases verses Percent attending - scaled")


plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/new_cases_attend_scaled.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
wa_mc.head(1)

# %%
wa_m.set_index("date", inplace=True, drop=True)
wa_2021 = wa_m["2021-01-31":"2021-02-28"]

# %%
wa_m.boxplot(column="avg_daily_perc_att", by="IMD_tiers_10", figsize=(12, 8))

# %%
# Creating plot
wa_2021.boxplot(column="avg_daily_perc_att", by="IMD_tiers_10", figsize=(12, 8))

# show plot
plt.show()

# %%
