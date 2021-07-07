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
# Group by month and merge

# %%
wa["date"] = pd.to_datetime(wa["date"], format="%d/%m/%Y")
wa.set_index("date", inplace=True, drop=True)

# %%
wa_m = (
    wa[wa.columns.difference(["time_period"])]
    .groupby([pd.Grouper(freq="M"), "la_name", "Code"])
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
wa_m.columns

# %%
wa_m["IMD_tiers_10"] = (
    pd.qcut(wa_m["IMD - Rank of average score"], 10, labels=False) + 1
)
wa_m["IDACI_tiers_10"] = (
    pd.qcut(wa_m["IDACI - Rank of average score"], 10, labels=False) + 1
)
wa_m["IDACI_lsoa_tiers_10"] = (
    pd.qcut(
        wa_m["IDACI - Rank of proportion of LSOAs in most deprived 10% nationally"],
        10,
        labels=False,
    )
    + 1
)
wa_m["Income_tiers_10"] = (
    pd.qcut(wa_m["Income - Rank of average score"], 10, labels=False) + 1
)

# %%
# Merge pop and wa datasets
wa_m = wa_m.merge(uk_pop[["Code", "EY_pop", "Geography"]], how="left", on="Code")

# %%
wa_m.set_index("date", inplace=True, drop=True)

# %%
wa_m.head(5)

# %% [markdown]
# IMD 10 tiers

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

# %%
tier10_list = list(tier10.columns.levels[1])
for col in tier10_list:
    tier10["Percent attending", col] = (
        tier10["total_children_in_early_years_settings", col] / tier10["EY_pop", col]
    ) * 100

# %%
tier10["Percent attending"].head(2)

# %%
heat_imd = tier10["Percent attending"].transpose()

# %%
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

# %% [markdown]
# IDACI lsoa 10 tiers

# %%
tier10 = (
    wa_m.groupby([pd.Grouper(freq="M"), "IDACI_lsoa_tiers_10"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IDACI_lsoa_tiers_10")
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

ax.invert_yaxis()

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance_p2/income10-attend-heatmap.jpg",
    bbox_inches="tight",
)

# %%
wa_m.head(1)

# %%
wa_m["avg_daily_perc_att"] = (
    wa_m["total_children_in_early_years_settings"] / wa_m["EY_pop"]
) * 100

# %%
wa_m.reset_index(inplace=True)

# %%
wa_m.head(1)

# %% [markdown]
# Covid cases

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
        "Income_tiers_10",
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
wa_mc["avg_daily_perc_cases"] = (
    wa_mc["newCasesBySpecimenDate"] / wa_mc["EY_pop"]
) * 100

# %%
wa_mc.head(1)

# %%
fig = plt.figure(figsize=(8, 8))
cases_att = plt.scatter(
    wa_mc["avg_daily_perc_cases"], wa_mc["avg_daily_perc_att"], color="#ff0000"
)
plt.xlabel("% avg daily cases")
plt.ylabel("% EYS attendance")

# %%
