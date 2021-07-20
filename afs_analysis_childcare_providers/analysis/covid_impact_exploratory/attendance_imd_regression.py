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

# %% [markdown]
# Read in datasets

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
# Merge weekly attendance and population indicators

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

# %%
# Merge datasets
wa = wa.merge(uk_pop[["Code", "EY_pop", "Geography"]], how="left", on="Code")

# %%
# Change Buckinghamshire from E06000060 to E10000002
wa["Code"] = wa["Code"].replace({"E06000060": "E10000002"})

# %% [markdown]
# Clean IMD datasets

# %%
# Remove leading and trailing whitespace in headers
idaci.rename(columns=lambda x: x.strip(), inplace=True)
health.rename(columns=lambda x: x.strip(), inplace=True)
income.rename(columns=lambda x: x.strip(), inplace=True)
imd.rename(columns=lambda x: x.strip(), inplace=True)

# %%
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
    .groupby([pd.Grouper(freq="M"), "la_name", "Code", "Geography"])
    .mean()
    .reset_index()
)

# %%
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
wa_m["avg_daily_perc_att"] = (
    wa_m["total_children_in_early_years_settings"] / wa_m["EY_pop"]
) * 100

# %% [markdown]
# Covid cases

# %%
new_cases["date"] = pd.to_datetime(new_cases["date"], format="%Y-%m-%d")
new_cases.set_index("date", inplace=True, drop=True)

# %%
new_cases_m = (
    new_cases.groupby([pd.Grouper(freq="M"), "areaCode", "areaName"])
    .mean()
    .reset_index()
)
new_cases_m.rename(columns={"areaCode": "Code"}, inplace=True)  # Rename to match uk_pop

# %%
wa_m = wa_m.merge(
    new_cases_m[["date", "Code", "newCasesBySpecimenDate"]],
    how="left",
    on=["date", "Code"],
)

# %%
temp_drop = ["Cornwall", "Isles Of Scilly", "Hackney", "City of London"]
wa_m = wa_m[~wa_m.la_name.isin(temp_drop)]

# %%
wa_m["avg_daily_perc_cases"] = (wa_m["newCasesBySpecimenDate"] / wa_m["EY_pop"]) * 100

# %%
wa_m.set_index("date", inplace=True, drop=True)
wa_Nov = wa_m["2020-11-30":"2020-11-30"]

# %%
wa_m.head(1)

# %% [markdown]
# ### Regression

# %%
wa_m.reset_index(inplace=True, drop=True)
wa_Nov.reset_index(inplace=True, drop=True)

# %%
wa_m.head(1)

# %%
features = wa_m.columns.difference(
    [
        "la_name",
        "Geography",
        "EY_pop",
        "count_of_closed_early_years_settings",
        "count_of_open_early_years_settings",
        "total_children_in_early_years_settings",
        "Code",
        "total_early_years_settings_reported_by_la",
        "avg_daily_perc_att",
        "avg_daily_perc_cases",
    ]
)

target = "avg_daily_perc_att"

# %%
X = wa_Nov[
    [
        "Health Deprivation and Disability - Average score",
        "Health Deprivation and Disability - Proportion  of LSOAs in most deprived 10% nationally",
        "IDACI - Average score",
        "IDACI - Proportion  of LSOAs in most deprived 10% nationally",
        "Income - Average score",
        "Income - Proportion  of LSOAs in most deprived 10% nationally",
        "IMD - Average score",
        "IMD - Proportion of LSOAs in most deprived 10% nationally",
        "newCasesBySpecimenDate",
    ]
]

y = wa_Nov["avg_daily_perc_att"]

# %% [markdown]
# Statsmodel

# %%
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

# %%
X = wa_Nov[features]

# %%
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

# %%
X = wa_Nov[
    [
        "IDACI - Rank of proportion of LSOAs in most deprived 10% nationally",
        "Rank of Income Scale",
        "newCasesBySpecimenDate",
        "IMD - Average score",
        "IMD - Proportion of LSOAs in most deprived 10% nationally",
    ]
]

# %%
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()
est.summary()

# %% [markdown]
# Standardise and scale

# %%
s_scaler = StandardScaler()

# %%
Nov_data = wa_Nov[
    [
        "avg_daily_perc_att",
        "Health Deprivation and Disability - Average score",
        "Health Deprivation and Disability - Proportion  of LSOAs in most deprived 10% nationally",
        "IDACI - Average score",
        "IDACI - Proportion  of LSOAs in most deprived 10% nationally",
        "Income - Average score",
        "Income - Proportion  of LSOAs in most deprived 10% nationally",
        "newCasesBySpecimenDate",
        "IMD - Average score",
        "IMD - Proportion of LSOAs in most deprived 10% nationally",
    ]
]

# %%
Nov_data_scaled = s_scaler.fit_transform(Nov_data)
Nov_data_scaled = pd.DataFrame(Nov_data_scaled, columns=Nov_data.columns)

# %%
Nov_data_scaled.describe().round(1)  # round the numbers for dispaly

# %%
from statsmodels.regression import linear_model

# %%
X_scaled = Nov_data_scaled[
    [
        "Health Deprivation and Disability - Average score",
        "Health Deprivation and Disability - Proportion  of LSOAs in most deprived 10% nationally",
        "IDACI - Average score",
        "IDACI - Proportion  of LSOAs in most deprived 10% nationally",
        "Income - Average score",
        "Income - Proportion  of LSOAs in most deprived 10% nationally",
        "newCasesBySpecimenDate",
    ]
]
y_scaled = Nov_data_scaled[["avg_daily_perc_att"]]

# %%
Xscaled_train, Xscaled_test, yscaled_train, yscaled_test = train_test_split(
    X_scaled, y_scaled, test_size=0.33
)

# %%
Xscaled_train = sm.add_constant(Xscaled_train)
sm_ols = linear_model.OLS(yscaled_train, Xscaled_train)
sm_model = sm_ols.fit()

# %%
sm_model.summary()
