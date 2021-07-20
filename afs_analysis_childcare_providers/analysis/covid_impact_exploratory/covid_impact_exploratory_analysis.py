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
# ## Covid-19 impact on school readiness

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import afs_analysis_childcare_providers

# %%
project_directory = afs_analysis_childcare_providers.PROJECT_DIR

# %%
cp = pd.read_excel(
    f"{project_directory}/inputs/data/Childcare_provider_level_data_as_at_31_August_2020.xlsx",
    sheet_name="Childcare_providers",
    usecols="A:AV",
)
ple = pd.read_excel(
    f"{project_directory}/inputs/data/Childcare_provider_level_data_as_at_31_August_2020.xlsx",
    sheet_name="Providers_left_EYR",
    usecols="A:AG",
)

# %% [markdown]
# ### Childcare provider sheet (providers and inspections dataset)
#
# - Provider UPRN: unique number
# - Provider type
#     - Subtype (some missing unknown why)
# - Status (not active in left EYR sheet)
# - Postcode, LA
# - Deprivation band (LA level)
# - Places plus estimates: max number of children can attend (inc estimates where data unavailable)
# - Most recent inspection date and number (EYR)
# - Effectiveness, quality of education... (EYR)
# - 2nd / 3rd of the above...

# %% [markdown]
# Notes
# - Last inspection could be any date (not just 2020)
# - This can give a view of:
#     - Active providers (as of 2020)
#     - Numbers closed / cancelled? (as of 2020)
#         - Could compare to previous datasets...
#     - Q: Places open and children attending?
#     - Proportion of providers by type?

# %%
cp.head(1)

# %% [markdown]
# ### Weekly attendence during covid
#
# https://explore-education-statistics.service.gov.uk/find-statistics/attendance-in-education-and-early-years-settings-during-the-coronavirus-covid-19-outbreak
#
# - 'c' value in counts
# - North Northamptonshire created in 2021

# %% [markdown]
# #### Limitations

# %% [markdown]
# Not all LAs respond to the survey each week (prior to the 29 April 2021) or fortnight (from the 29 April 2021). The data returned is ‘grossed up’ to account for non-response based on either data previously submitted (within the last week prior to the 29 April 2021, or fortnight from the 29 April 2021) or data the DfE already holds to estimate the total numbers of open settings and children attending those settings.

# %% [markdown]
# The number of children in attendance is as reported by LAs, based on data they collect from Early Years providers. Depending on the data collection methodology used, estimates could be affected by the number of providers submitting their information every other week. As such there is a high degree of uncertainty around the figures. We believe actual attendance to be higher than indicated, due to not all LAs reporting data for all providers.

# %% [markdown]
# Values of less than 5 for the Total Children in Early Years Settings and  Total Vulnerable Children in Early Years Settings have been suppressed and have been replaced with a “c”.

# %%
wa = pd.read_csv(
    f"{project_directory}/inputs/data/table_5_weekly_attendance_in_early_years_settings_during_the_covid-19_outbreak_at_local_authority_level_.csv"
)

# %%
wa = wa[
    [
        "date",
        "new_la_code",
        "la_name",
        "region_name",
        "total_early_years_settings_reported_by_la",
        "count_of_open_early_years_settings",
        "count_of_closed_early_years_settings",
        "total_children_in_early_years_settings",
        "total_vulnerable_children_in_early_years_settings",
    ]
]

# %%
wa.shape

# %%
wa.info()

# %%
wa["total_children_in_early_years_settings"].replace(
    ["c"], [0], inplace=True
)  # recode c to 0
wa["total_children_in_early_years_settings"] = wa[
    "total_children_in_early_years_settings"
].astype(int)

# %%
wa["total_vulnerable_children_in_early_years_settings"].replace(
    ["c"], [0], inplace=True
)  # recode c to 0
wa["total_vulnerable_children_in_early_years_settings"] = wa[
    "total_vulnerable_children_in_early_years_settings"
].astype(int)

# %%
wa["date"] = pd.to_datetime(wa["date"], format="%d/%m/%Y")

# %%
print(wa["date"].min(), wa["date"].max())  # Start and end dates
print(wa["date"].max() - wa["date"].min())  # duration

# %%
# Average children in EYS per LA per day of the week
wa.groupby([wa["date"].dt.weekday, "la_name"])[
    "total_children_in_early_years_settings"
].mean()

# %%
wa.set_index("date", inplace=True, drop=True)

# %%
g = wa.groupby(pd.Grouper(freq="M")).sum()  # DataFrameGroupBy (grouped by Month)
gm = wa.groupby(pd.Grouper(freq="M")).mean()  # DataFrameGroupBy (grouped by Month)

# %%
g["Perc_open"] = (
    g["count_of_open_early_years_settings"]
    / g["total_early_years_settings_reported_by_la"]
) * 100

# %%
fig, axs = plt.subplots(2, 2, figsize=(18, 12))  # Size

fig.suptitle("Total monthly for early years settings", fontsize=20)  # Title
# Plot all figures
axs[0, 0].plot(g.index, g["total_early_years_settings_reported_by_la"])
axs[0, 0].set_title("Total early years settings", fontsize=16)
axs[0, 1].plot(
    g.index,
    g[["count_of_open_early_years_settings", "count_of_closed_early_years_settings"]],
)
axs[0, 1].legend(("Open", "Closed"))
axs[0, 1].set_title("Total open and closed early years settings", fontsize=16)
axs[1, 0].plot(g.index, g["total_children_in_early_years_settings"])
axs[1, 0].set_title("Total children in early years settings", fontsize=16)
axs[1, 1].plot(g.index, g["total_vulnerable_children_in_early_years_settings"])
axs[1, 1].set_title("Total vulnerable children in early years settings", fontsize=16)

# X and Y labels
for ax in axs.flat:
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Average monthly total", fontsize=14)


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.savefig(f"{project_directory}/outputs/figures/covid_impact/total_monthly_eys.jpg")

# %%
fig, axs = plt.subplots(2, 2, figsize=(18, 12))  # Size

fig.suptitle(
    "Average of LA monthly totals for early years settings", fontsize=20
)  # Title
# Plot all figures
axs[0, 0].plot(gm.index, gm["total_early_years_settings_reported_by_la"])
axs[0, 0].set_title("Total early years settings", fontsize=16)
axs[0, 1].plot(
    gm.index,
    gm[["count_of_open_early_years_settings", "count_of_closed_early_years_settings"]],
)
axs[0, 1].legend(("Open", "Closed"))
axs[0, 1].set_title("Count of open and closed early years settings", fontsize=16)
axs[1, 0].plot(gm.index, gm["total_children_in_early_years_settings"])
axs[1, 0].set_title("Total children in early years settings", fontsize=16)
axs[1, 1].plot(gm.index, gm["total_vulnerable_children_in_early_years_settings"])
axs[1, 1].set_title("Total vulnerable children in early years settings", fontsize=16)

# X and Y labels
for ax in axs.flat:
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Average monthly total", fontsize=14)


# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/average_la_monthly_eys.jpg"
)

# %%
from sklearn.preprocessing import MinMaxScaler

g_scaled = g.copy()

scaler = MinMaxScaler()

for col in g_scaled.columns[g_scaled.columns != "Perc_open"]:
    scaler = MinMaxScaler()
    g_scaled[col] = scaler.fit_transform(g_scaled[col].values.reshape(-1, 1))

# %%
plt.plot(
    g_scaled.index,
    g_scaled[
        [
            "total_children_in_early_years_settings",
            "total_vulnerable_children_in_early_years_settings",
        ]
    ],
)
plt.gca().legend(("All children", "Vulnerable children"))

plt.title("Total - All Children and Vulnerable Children")
plt.xlabel("Month")
plt.ylabel("Average monthly total")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/total_children_vulnerable.jpg"
)

plt.show()

# %%
plt.plot(g_scaled.index, g_scaled["Perc_open"])
plt.legend(("Percent open",))

plt.title("Total Monthly - Percent Open Early Years Settings")
plt.xlabel("Month")
plt.ylabel("Average monthly")

plt.savefig(f"{project_directory}/outputs/figures/covid_impact/total_percent_open.jpg")

plt.show()

# %%
plt.plot(
    g_scaled.index,
    g_scaled[
        ["total_children_in_early_years_settings", "count_of_open_early_years_settings"]
    ],
)
plt.gca().legend(("Children attending", "Open EYS"))

plt.title("Total Monthly - Children Attending and Open Early Years Settings")
plt.xlabel("Month")
plt.ylabel("Monthly total")

plt.savefig(
    f"{project_directory}/outputs/figures/covid_impact/total_children_open_eys.jpg"
)

plt.show()

# %%
g_scaled.head(1)

# %%
g_scaled.head(1)

# %%
wa.head(1)

# %% [markdown]
# #### Also think about
# - Plotting / adding key covid dates....

# %% [markdown]
# ##### Metrics
# - All children attending vs vulnerable
# - Percent open
# - Children attending vs open
#
# #### Splits
# - regional
# - inner / outer london
# - boroughs
# - deprivation deciles

# %% [markdown]
# #### Region

# %%
region = (
    wa.groupby([pd.Grouper(freq="M"), "region_name"])[
        [
            "total_early_years_settings_reported_by_la",
            "count_of_open_early_years_settings",
        ]
    ]
    .sum()
    .unstack("region_name")
)

# %%
region_list = list(region.columns.levels[1])
for col in region_list:
    region["Percent open", col] = (
        region["count_of_open_early_years_settings", col]
        / region["total_early_years_settings_reported_by_la", col]
    ) * 100

# %%
region["Percent open"].head(1)

# %%
fig = plt.figure(figsize=(17, 15))

fig.suptitle(
    "Percentage of EYS open per Region compared to England average", fontsize=20
)  # Title


for i in range(9):
    pos = 331 + i  # this is to index the position of the subplot
    ax = plt.subplot(pos)
    ax.plot(region.index, region["Percent open"][region_list[i]])
    g_scaled["Perc_open"].plot(ax=ax)
    ax.legend((region_list[i], "Avg england"))
    ax.set_xlabel("", fontsize=14)
    ax.set_title(region_list[i], fontsize=15)


fig.supxlabel("Date", fontsize=12)
fig.supylabel("Percentage", fontsize=12)

fig.tight_layout(pad=2.0)

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/percent open per region.jpg"
)

# %% [markdown]
# #### Inner / outer London

# %%
borough_profiles = pd.read_excel(
    f"{project_directory}/inputs/data/london-borough-profiles.xlsx", sheet_name="Data"
)

# %%
london = wa.loc[wa["region_name"] == "London"].copy()

# %%
london.head(1)

# %%
borough_profiles = borough_profiles[["New code", "Inner/ Outer London"]]

# %%
borough_profiles = borough_profiles.iloc[1:]
borough_profiles = borough_profiles.iloc[:-6]

# %%
borough_profiles.head(3)

# %%
london.shape

# %%
london.rename(columns={"new_la_code": "New code"}, inplace=True)

# %%
borough_profiles.shape

# %%
london = (
    london.reset_index()
    .merge(borough_profiles, how="left", on="New code")
    .set_index("date")
)

# %%
inner_outer = (
    london.groupby([pd.Grouper(freq="M"), "Inner/ Outer London"])[
        [
            "total_early_years_settings_reported_by_la",
            "count_of_open_early_years_settings",
        ]
    ]
    .sum()
    .unstack("Inner/ Outer London")
)

# %%
cols = list(inner_outer.columns.levels[1])
for col in cols:
    inner_outer["Percent open", col] = (
        inner_outer["count_of_open_early_years_settings", col]
        / inner_outer["total_early_years_settings_reported_by_la", col]
    ) * 100

# %% [markdown]
# #### London boroughs

# %%
london.head(1)

# %%
inner = london.loc[london["Inner/ Outer London"] == "Inner London"].copy()
outer = london.loc[london["Inner/ Outer London"] == "Outer London"].copy()

print(inner.shape, outer.shape)

# %%
london_bi = (
    inner.groupby([pd.Grouper(freq="M"), "la_name"])[
        [
            "total_early_years_settings_reported_by_la",
            "count_of_open_early_years_settings",
        ]
    ]
    .sum()
    .unstack("la_name")
)
london_bo = (
    outer.groupby([pd.Grouper(freq="M"), "la_name"])[
        [
            "total_early_years_settings_reported_by_la",
            "count_of_open_early_years_settings",
        ]
    ]
    .sum()
    .unstack("la_name")
)

# %%
london_bi.shape

# %%
london_bo.shape

# %%
i_cols = list(london_bi.columns.levels[1])
for col in i_cols:
    london_bi["Percent open", col] = (
        london_bi["count_of_open_early_years_settings", col]
        / london_bi["total_early_years_settings_reported_by_la", col]
    ) * 100

# %%
o_cols = list(london_bo.columns.levels[1])
for col in o_cols:
    london_bo["Percent open", col] = (
        london_bo["count_of_open_early_years_settings", col]
        / london_bo["total_early_years_settings_reported_by_la", col]
    ) * 100

# %%
print(o_cols, len(o_cols))

# %%
print(i_cols, len(i_cols))

# %%
london_bo.head(1)

# %%
fig, ax = plt.subplots(5, 4, figsize=(15, 15))
fig.delaxes(ax[4, 3])  # The indexing is zero-based here

fig.suptitle(
    "Percentage of EYS open per Outer London Borough compared to average", fontsize=17
)  # Title

n = 0
for i in range(5):
    ax[i, 0].plot(london_bo.index, london_bo["Percent open"][o_cols[0 + n]])
    ax[i, 0].plot(inner_outer.index, inner_outer["Percent open", "Outer London"])
    ax[i, 0].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 0].legend((o_cols[0 + n], "Avg Outer London", "Avg England"))
    ax[i, 0].set_title(o_cols[0 + n], fontsize=12)
    ax[i, 0].set_xlabel("", fontsize=14)
    ax[i, 0].tick_params(axis="x", rotation=90)

    ax[i, 1].plot(london_bo.index, london_bo["Percent open"][o_cols[1 + n]])
    ax[i, 1].plot(inner_outer.index, inner_outer["Percent open", "Outer London"])
    ax[i, 1].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 1].legend((o_cols[0 + n], "Avg Outer London", "Avg England"))
    ax[i, 1].set_title(o_cols[1 + n], fontsize=12)
    ax[i, 1].legend((o_cols[1 + n], "Avg Outer London", "Avg England"))
    ax[i, 1].set_xlabel("", fontsize=14)
    ax[i, 1].tick_params(axis="x", rotation=90)

    ax[i, 2].plot(london_bo.index, london_bo["Percent open"][o_cols[2 + n]])
    ax[i, 2].plot(inner_outer.index, inner_outer["Percent open", "Outer London"])
    ax[i, 2].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 2].legend((o_cols[0 + n], "Avg Outer London", "Avg England"))
    ax[i, 2].set_title(o_cols[2 + n], fontsize=12)
    ax[i, 2].legend((o_cols[2 + n], "Avg Outer London", "Avg England"))
    ax[i, 2].set_xlabel("", fontsize=14)
    ax[i, 2].tick_params(axis="x", rotation=90)

    if i < 4:
        ax[i, 3].plot(london_bo.index, london_bo["Percent open"][o_cols[3 + n]])
        ax[i, 3].plot(inner_outer.index, inner_outer["Percent open", "Outer London"])
        ax[i, 3].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
        ax[i, 3].legend((o_cols[0 + n], "Avg Outer London", "Avg England"))
        ax[i, 3].set_title(o_cols[3 + n], fontsize=12)
        ax[i, 3].legend((o_cols[3 + n], "Avg Outer London", "Avg England"))
        ax[i, 3].set_xlabel("", fontsize=14)
        ax[i, 3].tick_params(axis="x", rotation=90)
        n += 4
    else:
        continue

fig.supxlabel("Date", fontsize=12)
fig.supylabel("Percentage", fontsize=12)
fig.tight_layout(pad=2.0)

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/percent_open_per_ol_borough.jpg"
)

# %%
fig, ax = plt.subplots(4, 4, figsize=(15, 15))
fig.delaxes(ax[3, 3])
fig.delaxes(ax[3, 2])

fig.suptitle(
    "Percentage of EYS open per Inner London Borough compared to average", fontsize=17
)  # Title

n = 0
for i in range(4):
    ax[i, 0].plot(london_bi.index, london_bi["Percent open"][i_cols[0 + n]])
    ax[i, 0].plot(inner_outer.index, inner_outer["Percent open", "Inner London"])
    ax[i, 0].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 0].legend((i_cols[0 + n], "Avg Inner London", "Avg England"))
    ax[i, 0].set_title(i_cols[0 + n], fontsize=12)
    ax[i, 0].set_xlabel("", fontsize=14)
    ax[i, 0].tick_params(axis="x", rotation=90)

    ax[i, 1].plot(london_bi.index, london_bi["Percent open"][i_cols[1 + n]])
    ax[i, 1].plot(inner_outer.index, inner_outer["Percent open", "Inner London"])
    ax[i, 1].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 1].legend((i_cols[1 + n], "Avg Inner London", "Avg England"))
    ax[i, 1].set_title(i_cols[1 + n], fontsize=12)
    ax[i, 1].set_xlabel("", fontsize=14)
    ax[i, 1].tick_params(axis="x", rotation=90)

    if i < 3:
        ax[i, 2].plot(london_bi.index, london_bi["Percent open"][i_cols[2 + n]])
        ax[i, 2].plot(inner_outer.index, inner_outer["Percent open", "Inner London"])
        ax[i, 2].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
        ax[i, 2].legend((i_cols[2 + n], "Avg Inner London", "Avg England"))
        ax[i, 2].set_title(i_cols[2 + n], fontsize=12)
        ax[i, 2].set_xlabel("", fontsize=14)
        ax[i, 2].tick_params(axis="x", rotation=90)

        ax[i, 3].plot(london_bi.index, london_bi["Percent open"][i_cols[3 + n]])
        ax[i, 3].plot(inner_outer.index, inner_outer["Percent open", "Inner London"])
        ax[i, 3].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
        ax[i, 3].legend((i_cols[3 + n], "Avg Inner London", "Avg England"))
        ax[i, 3].set_title(i_cols[3 + n], fontsize=12)
        ax[i, 3].set_xlabel("", fontsize=14)
        ax[i, 3].tick_params(axis="x", rotation=90)
        n += 4
    else:
        continue

fig.supxlabel("Date", fontsize=12)
fig.supylabel("Percentage", fontsize=12)
fig.tight_layout(pad=2.0)

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/percent_open_per_il_borough.jpg"
)

# %%
london_bi["Percent open"]["Lewisham"]  # Missing data for Lewisham

# %% [markdown]
# #### East of England

# %%
region_list

# %%
east_e = wa.loc[wa["region_name"] == "East of England"].copy()

# %%
easte_b = (
    east_e.groupby([pd.Grouper(freq="M"), "la_name"])[
        [
            "total_early_years_settings_reported_by_la",
            "count_of_open_early_years_settings",
        ]
    ]
    .mean()
    .unstack("la_name")
)

# %%
ee_cols = list(easte_b.columns.levels[1])
for col in ee_cols:
    easte_b["Percent open", col] = (
        easte_b["count_of_open_early_years_settings", col]
        / easte_b["total_early_years_settings_reported_by_la", col]
    ) * 100

# %%
print(ee_cols, len(ee_cols))

# %%
easte_b.head(1)

# %%
fig, ax = plt.subplots(3, 4, figsize=(18, 15))
fig.delaxes(ax[2, 3])

fig.suptitle(
    "Percentage of EYS open in East of England Boroughs compared to average",
    fontsize=17,
)  # Title

n = 0
for i in range(3):
    ax[i, 0].plot(easte_b.index, easte_b["Percent open"][ee_cols[0 + n]])
    ax[i, 0].plot(region.index, region["Percent open"]["East of England"])
    ax[i, 0].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 0].legend((ee_cols[0 + n], "Avg East of England", "Avg England"))
    ax[i, 0].set_title(ee_cols[0 + n], fontsize=12)
    ax[i, 0].set_xlabel("", fontsize=14)
    ax[i, 0].tick_params(axis="x", rotation=90)

    ax[i, 1].plot(easte_b.index, easte_b["Percent open"][ee_cols[1 + n]])
    ax[i, 1].plot(region.index, region["Percent open"]["East of England"])
    ax[i, 1].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 1].legend((ee_cols[1 + n], "Avg East of England", "Avg England"))
    ax[i, 1].set_title(ee_cols[1 + n], fontsize=12)
    ax[i, 1].set_xlabel("", fontsize=14)
    ax[i, 1].tick_params(axis="x", rotation=90)

    ax[i, 2].plot(easte_b.index, easte_b["Percent open"][ee_cols[2 + n]])
    ax[i, 2].plot(region.index, region["Percent open"]["East of England"])
    ax[i, 2].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
    ax[i, 2].legend((ee_cols[2 + n], "Avg East of England", "Avg England"))
    ax[i, 2].set_title(ee_cols[2 + n], fontsize=12)
    ax[i, 2].set_xlabel("", fontsize=14)
    ax[i, 2].tick_params(axis="x", rotation=90)

    if i < 2:
        ax[i, 3].plot(easte_b.index, easte_b["Percent open"][ee_cols[3 + n]])
        ax[i, 3].plot(region.index, region["Percent open"]["East of England"])
        ax[i, 3].plot(g_scaled["Perc_open"].index, g_scaled["Perc_open"])
        ax[i, 3].legend((ee_cols[3 + n], "Avg East of England", "Avg England"))
        ax[i, 3].set_title(ee_cols[3 + n], fontsize=12)
        ax[i, 3].set_xlabel("", fontsize=14)
        ax[i, 3].tick_params(axis="x", rotation=90)

        n += 4
    else:
        continue

fig.supxlabel("Date", fontsize=12)
fig.supylabel("Percentage", fontsize=12)
fig.tight_layout(pad=2.0)

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/percent_open_east_of_eng_boroughs.jpg"
)

# %% [markdown]
# #### Children attending

# %%
uk_pop = pd.read_excel(
    f"{project_directory}/inputs/data/ukpopestimatesmid2020on2021geography.xls",
    skiprows=7,
    sheet_name="MYE2 - Persons",
)

# %%
ca = wa[
    [
        "new_la_code",
        "la_name",
        "region_name",
        "total_children_in_early_years_settings",
        "total_vulnerable_children_in_early_years_settings",
    ]
].copy()

# %%
uk_pop["EY_pop"] = (
    uk_pop["0"] + uk_pop["1"] + uk_pop["2"] + uk_pop["3"] + uk_pop["4"] + uk_pop["5"]
)

# %%
uk_pop.head(2)

# %%
ey_pop = uk_pop[["Code", "EY_pop"]].copy()

# %%
ca.rename(columns={"new_la_code": "Code"}, inplace=True)

# %% [markdown]
# For the purpose of this analysis I am recoding North and West Northamptonshire to Northamptonshire.
#
# Removing for now (do this later)

# %%
norths = ["West Northamptonshire", "North Northamptonshire", "Northamptonshire"]

# %%
ca = ca[~ca["la_name"].isin(norths)]

# %%
ca = ca.reset_index().merge(ey_pop, how="left", on="Code").set_index("date")

# %%
ca.head(1)

# %%
ca.shape

# %%
region_ca = (
    ca.groupby([pd.Grouper(freq="M"), "region_name"])[
        ["total_children_in_early_years_settings", "EY_pop"]
    ]
    .mean()
    .unstack("region_name")
)
england_ca = ca.groupby(pd.Grouper(freq="M")).mean()

# %%
for col in region_list:
    region_ca["Children 1000 pop", col] = region_ca[
        "total_children_in_early_years_settings", col
    ] / (region_ca["EY_pop", col] / 1000)

# %%
for col in region_list:
    region_ca["Children 1000 pop", col] = region_ca[
        "total_children_in_early_years_settings", col
    ] / (region_ca["EY_pop", col] / 1000)

# %%
england_ca["Children 1000 pop"] = england_ca[
    "total_children_in_early_years_settings"
] / (england_ca["EY_pop"] / 1000)

# %%
england_ca.head(1)

# %% [markdown]
# Children in EYS per 1,000 population of children in that borough

# %%
# England and region total monthly avgs per 1000 pop... (get England and regions pops...or totals?)
# Mean monthly children per borough

# %%
region_ca.head(1)

# %% [markdown]
# Regional monthly average - count of children

# %%
fig = plt.figure(figsize=(17, 15))

fig.suptitle(
    "Regional avg monthly children attending per 1000 population compared to England average",
    fontsize=20,
)  # Title


for i in range(9):
    pos = 331 + i  # this is to index the position of the subplot
    ax = plt.subplot(pos)
    ax.plot(region_ca.index, region_ca["Children 1000 pop"][region_list[i]])
    england_ca["Children 1000 pop"].plot(ax=ax)
    ax.legend((region_list[i], "Avg england"))
    ax.set_xlabel("", fontsize=14)
    ax.set_title(region_list[i], fontsize=15)


fig.supxlabel("Date", fontsize=12)
fig.supylabel("Percentage", fontsize=12)

fig.tight_layout(pad=2.0)

fig.savefig(
    f"{project_directory}/outputs/figures/covid_impact/children_attend_per_region.jpg"
)

# %% [markdown]
# #### Deprivation scores

# %%
imd = pd.read_excel(
    f"{project_directory}/inputs/data/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx",
    sheet_name="IMD",
)

# %%
idaci = pd.read_excel(
    f"{project_directory}/inputs/data/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx",
    sheet_name="IDACI",
)

# %%
imd.head(1)

# %%
idaci["Code"] = idaci["Local Authority District code (2019)"]

# %%
imd["Code"] = imd["Local Authority District code (2019)"]

# %%
ca.head(1)

# %%
ca_imd = (
    ca.reset_index()
    .merge(imd[["Code", "IMD - Rank of average score "]], how="left", on="Code")
    .set_index("date")
)

# %%
ca_imd.head(1)

# %%
# ca_imd.to_excel('test.xlsx') # 26 LA's not mapped

# %%
# Remove for now - sort later....

# %%
ca_imd.shape

# %%
ca_imd = ca_imd[ca_imd["IMD - Rank of average score "].notna()]

# %%
ca_imd.shape

# %%
ca_imd["Children 1000 pop"] = ca_imd["total_children_in_early_years_settings"] / (
    ca_imd["EY_pop"] / 1000
)
ca_imd["Vuln children 1000 pop"] = ca_imd[
    "total_vulnerable_children_in_early_years_settings"
] / (ca_imd["EY_pop"] / 1000)

# %%
ca_imd.head(1)

# %%
plt.scatter(ca_imd["IMD - Rank of average score "], ca_imd["Children 1000 pop"])

# %%
ca_imd[["Children 1000 pop", "Vuln children 1000 pop"]].plot()

# %%
ca_imd["Children 1000 pop"].plot()

# %%
ca_imd["Vuln children 1000 pop"].plot()

# %%
ca_imd.reset_index(inplace=True)

# %%
test = ca_imd.pivot("date", "la_name", "Children 1000 pop")
test = test.fillna(0)

# %%
ax = sns.heatmap(test, cmap="YlGnBu", vmin=-300, vmax=1000)

# %%
ca_imd.head(2)

# %%
ca_imd = ca_imd.drop_duplicates(subset="la_name", keep="first")

# %%
ca_imd.shape

# %%

# %%
ca_imd[["la_name", "IMD - Rank of average score "]].plot(kind="bar")

# %%
ca_m = ca.drop_duplicates(subset="la_name", keep="first")

# %%
imd_m = imd.merge(ca_m[["Code", "la_name"]], how="left", on="Code")

# %%
idaci = idaci.merge(ca_m[["Code", "la_name"]], how="left", on="Code")

# %%
idaci.head(1)

# %%
idaci = idaci[idaci["la_name"].notna()]

# %%
imd_m = imd_m[imd_m["la_name"].notna()]

# %%
idaci.shape

# %%
idaci.set_index("la_name", inplace=True)

# %%
imd_m.set_index("la_name", inplace=True)

# %%
idaci.drop(
    [
        "Local Authority District code (2019)",
        "Local Authority District name (2019)",
        "Code",
        "IDACI - Average rank ",
        "IDACI - Average score ",
        "IDACI - Proportion of LSOAs in most deprived 10% nationally ",
    ],
    inplace=True,
    axis=1,
)

# %%
imd_m.drop(
    [
        "Local Authority District code (2019)",
        "Local Authority District name (2019)",
        "Code",
        "IMD - Average rank ",
        "IMD - Average score ",
        "IMD - Proportion of LSOAs in most deprived 10% nationally ",
        "IMD 2019 - Extent ",
        "IMD 2019 - Local concentration ",
    ],
    inplace=True,
    axis=1,
)

# %%
idaci.columns

# %%
imd_m.head(2)

# %%
fig, ax = plt.subplots(figsize=(7, 20))  # Sample figsize in inches

ax = sns.heatmap(imd_m, cmap="YlGnBu_r", vmin=1, vmax=317)

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/imd-heatmap.jpg",
    bbox_inches="tight",
)

# %%
fig, ax = plt.subplots(figsize=(5, 20))  # Sample figsize in inches

ax = sns.heatmap(idaci, cmap="YlGnBu_r", vmin=1, vmax=317)

ax.figure.savefig(
    f"{project_directory}/outputs/figures/covid_impact/idaci-heatmap.jpg",
    bbox_inches="tight",
)

# %%
