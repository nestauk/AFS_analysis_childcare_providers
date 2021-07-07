# -*- coding: utf-8 -*-
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

# %% [markdown]
# ### Analysis weekly attendance

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

# %%
project_directory = afs_analysis_childcare_providers.PROJECT_DIR

# %% [markdown]
# #### Actions
#
# -	PHE: std income deprivation
# -	Simpler, tighter – top / bottom / medium
# -	Findings such as bottom 10 40% lower than top 10…
# -	Map to covid rates also
# -	Regional cut: local plus compare to national average
# -	Dep: top, bottom, medium

# %% [markdown]
# #### Local education authority
#
# Local education authorities (LEAs) are the local councils in England and Wales that are responsible for education within their jurisdiction. The term is used to identify which council (district or county) is locally responsible for education in a system with several layers of local government.

# %% [markdown]
# #### Changes to accomadate
#
# - Buckinghamshire: Unitary authority in WA dataset but not IMD
#     - Aylesbury Vale, Chiltern, South Bucks and Wycombe district councils and Buckinghamshire County Council
#     - As combined into 1 in 2020
#     - Use upper tier Buckinghamshire for IMD?
# - Northamptonshire becomes West Northamptonshire and North Northamptonshire from April 2021
#     - Merge North and West counts and recode to 'Northamptonshire'
# - Local education authorities are made up of Unitary Authorities, Metropolitan districts, London Boroughs and County Councils
#     - Use upper tier district summaries for IMD data

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

# IMD upper tier
imd_upper = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="IMD",
)

# %%
wa.head(4)

# %%
uk_pop.head(5)

# %%
wa["total_children_in_early_years_settings"].replace(
    ["c"], [0], inplace=True
)  # recode c to 0
wa["total_children_in_early_years_settings"] = wa[
    "total_children_in_early_years_settings"
].astype(int)

# %%
# Aged 0 to 5
uk_pop["EY_pop"] = (
    uk_pop["0"] + uk_pop["1"] + uk_pop["2"] + uk_pop["3"] + uk_pop["4"] + uk_pop["5"]
)

# %%
wa.rename(columns={"new_la_code": "Code"}, inplace=True)  # Rename to match uk_pop

# %%
uk_pop["Name"] = uk_pop["Name"].str.strip()
uk_pop["Geography"] = uk_pop["Geography"].str.strip()

# %%
# Combining North and West Northamptonshire
uk_pop["Code"] = uk_pop["Code"].replace({"E06000062": "E06000061"})
uk_pop["Name"] = uk_pop["Name"].replace(
    {"West Northamptonshire": "North Northamptonshire"}
)
uk_pop = uk_pop.groupby(["Code", "Name", "Geography"]).sum().reset_index()
uk_pop["Name"] = uk_pop["Name"].replace({"North Northamptonshire": "Northamptonshire"})
uk_pop["Code"] = uk_pop["Code"].replace({"E06000061": "E10000021"})

# %%
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
wa.shape

# %%
wa = wa.merge(uk_pop[["Code", "EY_pop", "Geography"]], how="left", on="Code")

# %% [markdown]
# - Buckinghamshire is a different code
# - change from E06000060 to E10000002

# %%
wa["Code"] = wa["Code"].replace({"E06000060": "E10000002"})

# %%
imd_upper.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)

# %%
wa = wa.merge(
    imd_upper[["Code", "IMD - Rank of average score "]], how="left", on="Code"
)

# %%
wa["IMD_tiers_3"] = pd.qcut(wa["IMD - Rank of average score "], 3, labels=False) + 1
wa["IMD_tiers_10"] = pd.qcut(wa["IMD - Rank of average score "], 10, labels=False) + 1

# %%
wa["Percent_attending"] = (
    wa["total_children_in_early_years_settings"] / wa["EY_pop"]
) * 100

# %%
wa["date"] = pd.to_datetime(wa["date"], format="%d/%m/%Y")
wa.set_index("date", inplace=True, drop=True)

# %%
wa.head(1)

# %% [markdown]
# ### 3 Tier split

# %%
tier3 = (
    wa.groupby([pd.Grouper(freq="M"), "IMD_tiers_3"])[
        [
            "total_children_in_early_years_settings",
            "EY_pop",
        ]
    ]
    .sum()
    .unstack("IMD_tiers_3")
)

# %%
tier3_list = list(tier3.columns.levels[1])
for col in tier3_list:
    tier3["Percent attending", col] = (
        tier3["total_children_in_early_years_settings", col] / tier3["EY_pop", col]
    ) * 100

# %%
tier3.head(1)

# %%
plt.plot(
    tier3.index,
    tier3["Percent open"][1],
)
plt.plot(
    tier3.index,
    tier3["Percent open"][2],
)
plt.plot(
    tier3.index,
    tier3["Percent open"][3],
)

plt.gca().legend(("IMD 1", "IMD 2", "IMD 3"))

plt.title("Percentage of Children Attending Early Years Settings")
plt.xlabel("Month")
plt.ylabel("Percentage attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance/percentage_attend_imd1_3.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# ### Ten tier split

# %%
tier10 = (
    wa.groupby([pd.Grouper(freq="M"), "IMD_tiers_10"])[
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
plt.plot(
    tier10.index,
    tier10["Percent attending"][1],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][5],
)
plt.plot(
    tier10.index,
    tier10["Percent attending"][10],
)

plt.gca().legend(("IMD 1", "IMD 5", "IMD 10"))

plt.title("Percentage of Children Attending Early Years Settings")
plt.xlabel("Month")
plt.ylabel("Percentage attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance/percentage_attending_imd1_10.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %% [markdown]
# ### Percentage difference between tiers

# %%
tier10.head(1)

# %%
tier10["Percent difference 1", "1_10"] = (
    tier10["Percent attending", 10] - tier10["Percent attending", 1]
)
tier10["Percent difference 1", "1_9"] = (
    tier10["Percent attending", 9] - tier10["Percent attending", 1]
)
tier10["Percent difference 1", "1_8"] = (
    tier10["Percent attending", 8] - tier10["Percent attending", 1]
)
tier10["Percent difference 1", "1_7"] = (
    tier10["Percent attending", 7] - tier10["Percent attending", 1]
)
tier10["Percent difference 1", "1_6"] = (
    tier10["Percent attending", 6] - tier10["Percent attending", 1]
)
tier10["Percent difference 1", "1_5"] = (
    tier10["Percent attending", 5] - tier10["Percent attending", 1]
)


# %%
plt.plot(
    tier10.index,
    tier10["Percent difference 1"]["1_10"],
)

plt.plot(
    tier10.index,
    tier10["Percent difference 1"]["1_7"],
)


plt.plot(
    tier10.index,
    tier10["Percent difference 1"]["1_5"],
)

plt.gca().legend(("1 - 10", "1 - 7", "1 - 5"))


plt.title(
    "Percentage difference IMD 1 to IMD 10 - Children Attending Early Years Settings"
)
plt.xlabel("Month")
plt.ylabel("Percentage attending")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance/percentage_diff_imd1_10.jpg",
    bbox_inches="tight",
)

plt.tight_layout()

plt.show()

# %%
fig = plt.figure(figsize=(8, 8))
ten = plt.scatter(tier10.index, tier10["Percent difference 1"]["1_10"], color="#ff0000")
nine = plt.scatter(tier10.index, tier10["Percent difference 1"]["1_9"], color="#ff2d2d")
eight = plt.scatter(
    tier10.index, tier10["Percent difference 1"]["1_8"], color="#ff5252"
)
seven = plt.scatter(
    tier10.index, tier10["Percent difference 1"]["1_7"], color="#ff7b7b"
)
six = plt.scatter(tier10.index, tier10["Percent difference 1"]["1_6"], color="#ffbaba")

plt.legend(
    (ten, nine, eight, seven, six),
    ("IMD 1-10", "IMD 1-9", "IMD 1-8", "IMD 1-7", "IMD 1-6"),
    scatterpoints=1,
    loc="lower left",
    ncol=3,
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/analysis_attendance/percentage_attend_scatter_imd1_10.jpg",
    bbox_inches="tight",
)

plt.show()

# %% [markdown]
# Boxplots to show percentage difference / range

# %% [markdown]
# https://towardsdatascience.com/how-to-model-time-series-data-with-linear-regression-cd94d1d901c0

# %%
tier10.head(2)

# %% [markdown]
# ## Covid cases

# %% [markdown]
# Covid cases for upper tier LA: https://coronavirus.data.gov.uk/details/download

# %%
new_cases = pd.read_csv(
    "https://api.coronavirus.data.gov.uk/v2/data?areaType=utla&metric=newCasesBySpecimenDate&format=csv"
)

# %%
new_cases.head(5)

# %%
new_cases["date"] = pd.to_datetime(new_cases["date"], format="%Y-%m-%d")
new_cases.set_index("date", inplace=True, drop=True)

# %%
new_cases_w = new_cases.groupby(pd.Grouper(freq="W")).sum()
new_cases_m = new_cases.groupby(pd.Grouper(freq="M")).sum()

# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

ax1.plot(new_cases.index, new_cases["newCasesBySpecimenDate"])
ax2.plot(new_cases_w.index, new_cases_w["newCasesBySpecimenDate"])
ax3.plot(new_cases_m.index, new_cases_m["newCasesBySpecimenDate"])

# %%
wa.head(1)

# %%
wa_m = (
    wa[wa.columns.difference(["Percent_attending", "time_period"])]
    .groupby(
        [
            pd.Grouper(freq="M"),
            "la_name",
            "Code",
            "Geography",
            "IMD - Rank of average score ",
            "IMD_tiers_3",
            "IMD_tiers_10",
        ]
    )
    .mean()
    .reset_index()
)

# %%
new_cases_m = (
    new_cases.groupby([pd.Grouper(freq="M"), "areaCode", "areaName"])
    .mean()
    .reset_index()
)

# %%
new_cases_m.rename(columns={"areaCode": "Code"}, inplace=True)  # Rename to match uk_pop

# %%
wa_m.head(1)

# %%
new_cases_m.head(1)

# %%
print(wa_m.shape)

# %%
wa_m = wa_m.merge(
    new_cases_m[["date", "Code", "newCasesBySpecimenDate"]],
    how="left",
    on=["date", "Code"],
)

# %%
print(wa_m.shape)

# %%
# Cornwall and Isles of Scilly & Hackney and City of London combined in covid dataset
# Hackney not in for first 3 months

# %%
# E06000052 Cornwall and Isles of Scilly
# - Isles of Scilly: E06000053
# Cornwall
# E09000012 Hackney and City of London
# - Hackney: E09000001

# %% [markdown]
# Temp - dropping Hackney, City of London, Cornwall and Isles of Scilly
# - Deal with these later.

# %%
temp_drop = ["Cornwall", "Isles Of Scilly", "Hackney", "City of London"]

# %%
wa_m = wa_m[~wa_m.la_name.isin(temp_drop)]

# %%
wa_m["avg_daily_perc_att"] = (
    wa_m["total_children_in_early_years_settings"] / wa_m["EY_pop"]
) * 100
wa_m["avg_daily_perc_cases"] = (wa_m["newCasesBySpecimenDate"] / wa_m["EY_pop"]) * 100

# %%
wa_m[
    [
        "date",
        "la_name",
        "IMD - Rank of average score ",
        "avg_daily_perc_att",
        "avg_daily_perc_cases",
    ]
].head(5)

# %% [markdown]
# ### Linear regression

# %%
wa_all = wa_m[
    [
        "IMD - Rank of average score ",
        "avg_daily_perc_att",
        "avg_daily_perc_cases",
        "IMD_tiers_3",
        "IMD_tiers_10",
    ]
]

# %%
wa_m.set_index("date", inplace=True, drop=True)

# %%
wa_Nov = wa_m["2020-11-30":"2020-11-30"]

wa_2021 = wa_m["2021-02-01":"2021-04-30"]
wa_Apr = wa_m["2021-04-30":"2021-04-30"]

# %%
wa_all.head(1)

# %%
wa_all.shape

# %%
from sklearn.linear_model import LinearRegression

# %%
x = wa_all[["IMD - Rank of average score "]]
y = wa_all[["avg_daily_perc_att"]]

# %%
regressor = LinearRegression()
regressor.fit(x, y)

# %%
y_pred = regressor.predict(x)

# %%
regressor.coef_

# %%
regressor.intercept_

# %%
plt.scatter(x, y, color="red")
plt.plot(x, regressor.predict(x), color="blue")
plt.title("IMD vs attendance")
plt.xlabel("IMD rank")
plt.ylabel("% attendance EYS")
plt.show()

# %%
x = wa_2021[["IMD - Rank of average score ", "avg_daily_perc_cases"]].values.reshape(
    -1, len(["IMD - Rank of average score ", "avg_daily_perc_cases"])
)

y = wa_2021[["avg_daily_perc_att"]].values.reshape(-1, 1)

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)
df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
df.head(5)

# %%
df1 = df.head(25)
df1.plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()

# %%
x = wa_2021["IMD - Rank of average score "].values.reshape(-1, 1)
y = wa_2021["avg_daily_perc_att"].values.reshape(-1, 1)

# %%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# %%
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm

# %%
# To retrieve the intercept:
print(regressor.intercept_)

# For retrieving the slope:
print(regressor.coef_)

# %%
y_pred = regressor.predict(X_test)
df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": y_pred.flatten()})
df.head(3)

# %%
df1 = df.head(25)
df1.plot(kind="bar", figsize=(16, 10))
plt.grid(which="major", linestyle="-", linewidth="0.5", color="green")
plt.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
plt.show()

# %%
plt.scatter(X_test, y_test, color="gray")
plt.plot(X_test, y_pred, color="red", linewidth=2)
plt.show()

# %%
# IMD tiers plus covid cases Nov 2020 cut
########################################################################################

features = ["IMD_tiers_3", "IMD_tiers_10", "avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_Nov[features].values.reshape(-1, len(features))
y = wa_Nov[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")

# %%
# IMD plus covid cases Nov 2020 cut
########################################################################################

features = ["IMD - Rank of average score ", "avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_Nov[features].values.reshape(-1, len(features))
y = wa_Nov[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")

# %%
# IMD plus covid cases 2021 cut
########################################################################################

features = ["IMD - Rank of average score ", "avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_2021[features].values.reshape(-1, len(features))
y = wa_2021[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")

# %%
# Covid cases 2021 cut
########################################################################################

features = ["avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_2021[features].values.reshape(-1, len(features))
y = wa_2021[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")

# %%
# IMD plus covid cases
########################################################################################

features = ["IMD - Rank of average score ", "avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_all[features].values.reshape(-1, len(features))
y = wa_all[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")

# %%
# IMD plus covid cases
########################################################################################

features = ["avg_daily_perc_cases"]
target = "avg_daily_perc_att"

X = wa_all[features].values.reshape(-1, len(features))
y = wa_all[target].values

ols = linear_model.LinearRegression()
model = ols.fit(X, y)

print("Features                :  %s" % features)
print("Regression Coefficients : ", [round(item, 2) for item in model.coef_])
print("R-squared               :  %.2f" % model.score(X, y))
print("Y-intercept             :  %.2f" % model.intercept_)
print("")
