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

# %%
import pandas as pd
import numpy as np
import afs_analysis_childcare_providers
import pandas as pd
import geopandas as gpd
import altair as alt
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# %%
project_directory = afs_analysis_childcare_providers.PROJECT_DIR

# %%
county_ua_shapefile = gpd.read_file(
    f"{project_directory}/inputs/data/shapefiles_utla/Counties_and_Unitary_Authorities_(December_2019)_Boundaries_UK_BGC.shp"
).to_crs(epsg=4326)

# %%
region_shapefile = gpd.read_file(
    f"{project_directory}/inputs/data/shapefiles_regions/Regions_(December_2019)_Boundaries_EN_BFC.shp"
).to_crs(epsg=4326)


# %% [markdown]
# ### Functions

# %%
def create_la_map(df, col_val, col_la, scheme, min_scale, max_scale):

    # Creating configs for color,selection,hovering
    geo_select = alt.selection_single(fields=["reach_area"])
    color = alt.condition(
        geo_select,
        alt.Color(
            col_val + ":Q",
            scale=alt.Scale(scheme=scheme, domain=[min_scale, max_scale]),
        ),
        alt.value("lightgray"),
    )
    # Creating an altair map layer
    choro = (
        alt.Chart(df)
        .mark_geoshape(stroke="black")
        .encode(
            color=color,
            tooltip=[
                alt.Tooltip(col_la + ":N", title="Education Authority Name"),
                alt.Tooltip(
                    col_val + ":Q",
                    title=col_val,
                    format="1.2f",
                ),
            ],
        )
        .add_selection(geo_select)
        .properties(width=650, height=800)
    ).configure_view(strokeWidth=0)
    return choro


# %% [markdown]
# ### Data read-in

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

# Income
income = pd.read_excel(
    f"{project_directory}/inputs/data/File_11_-_IoD2019_Local_Authority_District_Summaries__upper-tier__.xlsx",
    sheet_name="Income",
)

# Covid cases
new_cases = pd.read_csv(
    "https://api.coronavirus.data.gov.uk/v2/data?areaType=utla&metric=newCasesBySpecimenDate&format=csv"
)

# Labour force - industry
industry = pd.read_csv(f"{project_directory}/inputs/data/Nomis-industry-uacounty.csv")

# %% [markdown]
# ### Data cleaning

# %%
wa["total_children_in_early_years_settings"].replace(
    ["c"], [0], inplace=True
)  # recode c to 0
wa["total_children_in_early_years_settings"] = wa[
    "total_children_in_early_years_settings"
].astype(int)

# Aged 0 to 4 population
uk_pop["EY_pop"] = uk_pop["0"] + uk_pop["1"] + uk_pop["2"] + uk_pop["3"] + uk_pop["4"]

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

# %%
# Remove leading and trailing whitespace in headers
imd.rename(columns=lambda x: x.strip(), inplace=True)
idaci.rename(columns=lambda x: x.strip(), inplace=True)
income.rename(columns=lambda x: x.strip(), inplace=True)


# Rename code to match wa df
imd.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)
idaci.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)
income.rename(
    columns={"Upper Tier Local Authority District code (2019)": "Code"}, inplace=True
)


# %%
def pct_rank_qcut(series, n):
    edges = pd.Series([float(i) / n for i in range(n + 1)])
    f = lambda x: (edges >= x).argmax()
    return series.rank(pct=1).apply(f)


# %%
imd["IMD_tiers_10"] = pct_rank_qcut(imd["IMD - Rank of average score"], 10)
imd["IDACI_tiers_10"] = pct_rank_qcut(idaci["IDACI - Rank of average score"], 10)
income["Income_tiers_10"] = pct_rank_qcut(income["Income - Rank of average score"], 10)

# %% [markdown]
# ### Weekly data count

# %%
wa["la_name"].nunique()

# %%
weekly_count = wa.copy()
weekly_count["count"] = 1
weekly_count = (
    weekly_count.groupby(["time_period", "time_identifier", "la_name"])["count"]
    .sum()
    .unstack("la_name")
)
weekly_count.fillna(0, inplace=True)

# %%
weekly_total = weekly_count.astype(bool)
weekly_total["sum"] = weekly_total[list(weekly_total.columns)].sum(axis=1)
weekly_total["sum_perc"] = (weekly_total["sum"] / 151) * 100
weekly_total.head(1)

# %%
weeks_to_drop = weekly_total[weekly_total.sum_perc < 60].reset_index()[
    ["time_period", "time_identifier"]
]

# %%
fig = plt.figure(figsize=(16, 6))


weekly_total["sum_perc"].plot(kind="bar")

# %%
drop = weekly_total[weekly_total.sum_perc < 60].index

# %%
weekly_count_dropped = weekly_count.copy()
weekly_count_dropped = weekly_count_dropped.drop(drop, axis="index")

# %%
weekly_count_dropped = weekly_count_dropped.astype(bool).sum(axis=0).reset_index()
weekly_count_dropped["missing"] = 53 - weekly_count_dropped[0]
weekly_count_dropped.set_index("la_name", drop=True, inplace=True)

weekly_count = weekly_count.astype(bool).sum(axis=0).reset_index()
weekly_count["missing"] = 59 - weekly_count[0]
weekly_count.set_index("la_name", drop=True, inplace=True)

# %%
weekly_count_dropped["perc_missing"] = (weekly_count_dropped["missing"] / 53) * 100
weekly_count["perc_missing"] = (weekly_count["missing"] / 59) * 100

# %%
weekly_count_dropped.sort_values(by="perc_missing", inplace=True)
weekly_count.sort_values(by="perc_missing", inplace=True)

# %%
missing_over_20_dropped = weekly_count_dropped[
    weekly_count_dropped["perc_missing"] > 20
]
missing_over_20 = weekly_count[weekly_count["perc_missing"] > 20]

# %%
fig = plt.figure(figsize=(16, 6))

missing_over_20_dropped["perc_missing"].plot(kind="bar")

# %%
fig = plt.figure(figsize=(16, 6))

missing_over_20["perc_missing"].plot(kind="bar")

# %%
weekly_count_dropped.reset_index(drop=False, inplace=True)
weekly_count_dropped.drop([0, "missing"], axis=1, inplace=True)

# %%
la_codes = wa[["la_name", "Code"]].drop_duplicates()

# %%
weekly_count_dropped = weekly_count_dropped.merge(la_codes, how="left", on="la_name")

# %%
weekly_count_dropped.head(1)

# %%
weekly_count_dropped_geo = weekly_count_dropped.merge(
    county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
)
weekly_count_dropped_geo = gpd.GeoDataFrame(
    weekly_count_dropped_geo, geometry="geometry"
)
weekly_count_dropped_map = create_la_map(
    weekly_count_dropped_geo, "perc_missing", "la_name", "lighttealblue", 10, 90
)
weekly_count_dropped_map.save(
    f"{project_directory}/outputs/figures/covid_impact/maps/percent_missing_weekly_map.html"
)

# %%
weekly_count_dropped_map

# %% [markdown]
# ### Missing values - fill for each week

# %% [markdown]
# Remove weeks with > 60% missing

# %%
weeks_to_drop["week-year"] = (
    weeks_to_drop["time_period"].astype(str) + "_" + weeks_to_drop["time_identifier"]
)
weeks_to_drop = list(weeks_to_drop["week-year"])

# %%
wa["week-year"] = wa["time_period"].astype(str) + "_" + wa["time_identifier"]

# %%
wa.shape

# %%
wa = wa[~wa["week-year"].isin(weeks_to_drop)]

# %%
wa.shape

# %% [markdown]
# Fill missing weeks for LA's with next or previus value

# %%
time_period = wa[["week-year"]].copy()
time_period.drop_duplicates(inplace=True)
time_period.reset_index(inplace=True, drop=True)

# %%
time_cat = [
    "2020_Week 15",
    "2020_Week 16",
    "2020_Week 17",
    "2020_Week 18",
    "2020_Week 19",
    "2020_Week 20",
    "2020_Week 21",
    "2020_Week 22",
    "2020_Week 23",
    "2020_Week 24",
    "2020_Week 25",
    "2020_Week 26",
    "2020_Week 27",
    "2020_Week 28",
    "2020_Week 29",
    "2020_Week 30",
    "2020_Week 31",
    "2020_Week 32",
    "2020_Week 33",
    "2020_Week 34",
    "2020_Week 35",
    "2020_Week 37",
    "2020_Week 38",
    "2020_Week 39",
    "2020_Week 40",
    "2020_Week 41",
    "2020_Week 42",
    "2020_Week 43",
    "2020_Week 44",
    "2020_Week 45",
    "2020_Week 46",
    "2020_Week 47",
    "2020_Week 48",
    "2020_Week 49",
    "2020_Week 50",
    "2020_Week 51",
    "2021_Week 1",
    "2021_Week 2",
    "2021_Week 3",
    "2021_Week 4",
    "2021_Week 5",
    "2021_Week 6",
    "2021_Week 7",
    "2021_Week 8",
    "2021_Week 9",
    "2021_Week 10",
    "2021_Week 11",
    "2021_Week 12",
    "2021_Week 13",
    "2021_Week 16",
    "2021_Week 17",
    "2021_Week 19",
    "2021_Week 21",
]

# %%
wa_w = (
    wa[wa.columns.difference(["time_period"])]
    .groupby(["week-year", "la_name", "Code", "region_name", "region_code"])
    .mean()
    .reset_index()
)

# %%
wa_reindexed = pd.DataFrame()

for area in wa_w["la_name"].unique():
    la_wa = wa_w[wa_w["la_name"] == area].copy()
    la = time_period.copy()
    la["la_name"] = area
    law = la.merge(la_wa, how="left", on=["week-year", "la_name"])
    law["week-year"] = law["week-year"].astype("category")
    law["week-year"] = pd.Categorical(
        law["week-year"], categories=time_cat, ordered=True
    )
    law.sort_values("week-year", inplace=True)
    law.ffill(inplace=True)
    law.bfill(inplace=True)
    wa_reindexed = wa_reindexed.append(law, ignore_index=True)

# %%
wa_w = wa_reindexed.merge(
    imd[imd.columns.difference(["Upper Tier Local Authority District name (2019)"])],
    how="left",
    on="Code",
)

wa_w = wa_w.merge(
    idaci[
        idaci.columns.difference(["Upper Tier Local Authority District name (2019)"])
    ],
    how="left",
    on="Code",
)

wa_w = wa_w.merge(
    income[
        income.columns.difference(["Upper Tier Local Authority District name (2019)"])
    ],
    how="left",
    on="Code",
)

wa_w = wa_w.merge(uk_pop[["Code", "EY_pop", "Geography"]], how="left", on="Code")

# %%
wa_w["Children not attending"] = (
    wa_w["EY_pop"] - wa_w["total_children_in_early_years_settings"]
).round(0)
wa_w["Percent of pop attending"] = (
    wa_w["total_children_in_early_years_settings"] / wa_w["EY_pop"]
) * 100

# %%
wa_w.head(1)

# %% [markdown]
# ### Children not attending - weekly: maps

# %%
# Create % not attending map for every week
for week in wa_w["week-year"].unique():
    week_df = wa_w[wa_w["week-year"] == week]
    week_df_geo = week_df.merge(
        county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
    )
    week_df_geo = gpd.GeoDataFrame(week_df_geo, geometry="geometry")
    week_map = create_la_map(
        week_df_geo, "Percent of pop attending", "la_name", "plasma", 0, 80
    )
    week_map.save(
        f"{project_directory}/outputs/figures/covid_impact/maps/" + week + "_map.html"
    )

# %%
print(week)
week_map

# %% [markdown]
# ### Children not attending - total

# %% [markdown]
# #### Week-year

# %%
wa_all = (
    wa_w[
        [
            "week-year",
            "count_of_closed_early_years_settings",
            "count_of_open_early_years_settings",
            "total_children_in_early_years_settings",
            "total_early_years_settings_reported_by_la",
            "EY_pop",
        ]
    ]
    .groupby(["week-year"])
    .sum()
    .reset_index()
)

# %%
wa_all["Children not attending"] = (
    wa_all["EY_pop"] - wa_all["total_children_in_early_years_settings"]
).round(0)
wa_all["Percent of pop attending"] = (
    wa_all["total_children_in_early_years_settings"] / wa_all["EY_pop"]
) * 100

# %%
wa_all.set_index("week-year", inplace=True)

# %%
wa_all["Children not attending"].plot(figsize=(12, 6))

plt.title("Total children not attending EYS - England")
plt.xlabel("Week-year")
plt.ylabel("Total children not attending")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# %% [markdown]
# #### UTLA

# %%
utla_total = (
    wa_w[["la_name", "Code", "Children not attending"]]
    .groupby(["la_name", "Code"])
    .sum()
    .reset_index()
)

# %%
utla_total.head(1)

# %%
utla_total_geo = utla_total.merge(
    county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
)
utla_total_geo = gpd.GeoDataFrame(utla_total_geo, geometry="geometry")
utla_total_map = create_la_map(
    utla_total_geo,
    "Children not attending",
    "la_name",
    "plasma",
    utla_total["Children not attending"].min(),
    utla_total["Children not attending"].max(),
)
utla_total_map.save(
    f"{project_directory}/outputs/figures/covid_impact/maps/utla_total_map.html"
)

# %%
utla_total_map

# %% [markdown]
# #### Region

# %%
region_total = (
    wa_w[["region_name", "region_code", "Children not attending"]]
    .groupby(["region_name", "region_code"])
    .sum()
    .reset_index()
)

# %%
region_total.head(1)

# %%
region_total_geo = region_total.merge(
    region_shapefile, left_on="region_code", right_on="rgn19cd", how="left"
)
region_total_geo = gpd.GeoDataFrame(region_total_geo, geometry="geometry")
region_total_map = create_la_map(
    region_total_geo,
    "Children not attending",
    "region_name",
    "plasma",
    region_total["Children not attending"].min(),
    region_total["Children not attending"].max(),
)
region_total_map.save(
    f"{project_directory}/outputs/figures/covid_impact/maps/region_total_map.html"
)

# %%
region_total_map

# %% [markdown]
# ### Percent of children population not attending - 2020 vs 2021

# %%
wa_w["year"] = wa_w["week-year"].str[:4]

# %%
wa_year = (
    wa_w[
        ["year", "total_children_in_early_years_settings", "EY_pop", "la_name", "Code"]
    ]
    .groupby(["year", "la_name", "Code"])
    .sum()
    .reset_index()
)

# %%
wa_year["Percent of pop attending"] = (
    wa_year["total_children_in_early_years_settings"] / wa_year["EY_pop"]
) * 100

# %%
wa_year.head(1)

# %%
wa_year["Percent of pop attending"].max()

# %%
df_2020 = wa_year[wa_year["year"] == "2020"]
df_2020_geo = df_2020.merge(
    county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
)
df_2020_geo = gpd.GeoDataFrame(df_2020_geo, geometry="geometry")
map_2020 = create_la_map(
    df_2020_geo, "Percent of pop attending", "la_name", "plasma", 0, 50
)
map_2020.save(f"{project_directory}/outputs/figures/covid_impact/maps/2020_la_map.html")

# %%
df_2021 = wa_year[wa_year["year"] == "2021"]
df_2021_geo = df_2021.merge(
    county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
)
df_2021_geo = gpd.GeoDataFrame(df_2021_geo, geometry="geometry")
map_2021 = create_la_map(
    df_2021_geo, "Percent of pop attending", "la_name", "plasma", 0, 50
)
map_2021.save(f"{project_directory}/outputs/figures/covid_impact/maps/2021_la_map.html")

# %%
map_2020

# %%
map_2021

# %% [markdown]
# ### Children not attending - IMD scores

# %%
# IMD
imd_total = wa_w.groupby(["IMD_tiers_10"])[
    [
        "Children not attending",
    ]
].sum()

# %%
total = imd_total["Children not attending"].sum()

imd_total["Percent proportion"] = imd_total["Children not attending"] / total * 100

# %%
imd_total["Percent proportion"].plot(kind="bar")

plt.title("IMD - Percent of non attending")
plt.xlabel("IMD")
plt.ylabel("Percentage deciles")

plt.tight_layout()

plt.show()

# %%
# IDACI
idaci_total = wa_w.groupby(["IDACI_tiers_10"])[
    [
        "Children not attending",
    ]
].sum()

# %%
total = idaci_total["Children not attending"].sum()

idaci_total["Percent proportion"] = idaci_total["Children not attending"] / total * 100

# %%
idaci_total["Percent proportion"].plot(kind="bar")

plt.title("IDACI - Percent of non attending")
plt.xlabel("IDACI deciles")
plt.ylabel("Percentage")

plt.tight_layout()

plt.show()

# %%
# Income
income_total = wa_w.groupby(["Income_tiers_10"])[
    [
        "Children not attending",
    ]
].sum()

# %%
total = income_total["Children not attending"].sum()

income_total["Percent proportion"] = (
    income_total["Children not attending"] / total * 100
)

# %%
income_total["Percent proportion"].plot(kind="bar")

plt.title("Income - Percent of non attending")
plt.xlabel("Income deciles")
plt.ylabel("Percentage")

plt.tight_layout()

plt.show()

# %% [markdown]
# ### Notes meeting 14-07-21
#
# - Plot missing (in seperate) x2 graphs show and compare ✓
#     - Data collected over time - gaps ✓
# - Change scale: scale=alt.Scale(scheme=‘lighttealblue’,domain=[1, 100]) ✓
# - Weekly plot covid rates map next to
#     - https://geopandas.org/docs/user_guide/aggregation_with_dissolve.html
#     - For borough aggregations
# - reverse 'not attending' ✓
# - Save HTML files in bucket - nesta test
# - Jobs:
#     - Labour force survey
#     - Job by region
#     - Hypothesis test
# - Pop density - if time

# %% [markdown]
# Dissolve:
# https://stackoverflow.com/questions/40385782/make-a-union-of-polygons-in-geopandas-or-shapely-into-a-single-geometry

# %% [markdown]
# #### Covid rates

# %%
new_cases["date"] = pd.to_datetime(new_cases["date"], format="%Y-%m-%d")
new_cases.set_index("date", inplace=True, drop=True)
new_cases.rename(columns={"areaCode": "Code"}, inplace=True)  # Rename to match
new_cases = new_cases.loc["2020-04-06":"2021-05-30"]  # Remove rows before 06/04/2020

# %%
# Get total weekly cases - Marks date as week end (Sunday)
new_cases = (
    new_cases.groupby([pd.Grouper(freq="W"), "Code", "areaName"]).sum().reset_index()
)

# %%
merged_la = ["E09000012", "E06000052"]
merged_la_cases = new_cases[new_cases.Code.isin(merged_la)]
new_cases = new_cases[~new_cases.Code.isin(merged_la)]
merged_la_casesCI = merged_la_cases.copy()
merged_la_casesHC = merged_la_cases.copy()
merged_la_casesCI["Code"].replace(
    {"E06000052": "E06000053", "E09000012": "E09000001"}, inplace=True
)
merged_la_casesCI["newCasesBySpecimenDate"] = (
    merged_la_casesCI["newCasesBySpecimenDate"] / 2
)
merged_la_casesHC["newCasesBySpecimenDate"] = (
    merged_la_casesHC["newCasesBySpecimenDate"] / 2
)
new_cases = pd.concat([new_cases, merged_la_casesCI, merged_la_casesHC], axis=0)
area = wa_w.drop_duplicates("la_name")[["la_name", "Code", "EY_pop"]]

# %%
new_cases_utla = new_cases.merge(area, how="left", on=["Code"])
new_cases_utla = new_cases_utla.dropna(subset=["la_name"])

# %%
max_min = new_cases_utla.copy()
max_min["New cases per 1000"] = max_min["newCasesBySpecimenDate"] / (
    max_min["EY_pop"] / 1000
)
print(max_min["New cases per 1000"].min(), max_min["New cases per 1000"].max())

# %%
for date in new_cases_utla["date"].unique():
    week_df = new_cases_utla[new_cases_utla["date"] == date]
    week_df.set_index("date", inplace=True)
    week_df_geo = week_df.merge(
        county_ua_shapefile, left_on="Code", right_on="ctyua19cd", how="left"
    )
    week_df_geo = gpd.GeoDataFrame(week_df_geo, geometry="geometry")
    week_df_geo = week_df_geo.dissolve(by="areaName", aggfunc="sum")
    week_df_geo.reset_index(inplace=True)
    week_df_geo["New cases per 1000"] = week_df_geo["newCasesBySpecimenDate"] / (
        week_df_geo["EY_pop"] / 1000
    )
    week_df_map = create_la_map(
        week_df_geo,
        "New cases per 1000",
        "areaName",
        "plasma",
        week_df_geo["New cases per 1000"].min(),
        week_df_geo["New cases per 1000"].max(),
    )
    week_df_map.save(
        f"{project_directory}/outputs/figures/covid_impact/maps/covid_rates/"
        + str(date)[:10]
        + "_map.html"
    )

# %%
print(str(date)[:10])
week_df_map

# %% [markdown]
# #### Jobs labour force survey

# %% [markdown]
# - Jobs by type and region overview
# - Regions where attendance low - what is the job range
# - Attendance distribution by job type

# %%
# Labour force - industry
industry = pd.read_csv(
    f"{project_directory}/inputs/data/Nomis-industry-uacounty.csv", skiprows=6
)

# %%
industry = industry.iloc[1:]
industry = industry.head(205).copy()
industry.Area = industry.Area.apply(lambda x: x.split(":")[1])
industry.rename(columns={"mnemonic": "Code"}, inplace=True)  # Rename to match

# %%
area = wa_w.drop_duplicates("la_name")[["la_name", "Code", "EY_pop"]]

# %%
industry_area = area.merge(industry, how="left", on=["Code"])

# %%
industry_area = industry_area.loc[
    :, ~industry_area.columns.str.startswith("Conf")
].copy()

# %% [markdown]
# - '!' Estimate and confidence interval not available since the group sample size is zero or disclosive (0-2)
# - '*' Estimate and confidence interval unreliable since the group sample size is small (3-9)
# - '~' Estimate is less than 500
# - '-' These figures are missing.

# %% [markdown]
# Recode to zero for now?

# %%
for col in industry_area.loc[:, industry_area.columns.str.startswith("T13")].columns:
    industry_area[col] = (
        pd.to_numeric(industry_area[col], errors="coerce").fillna(0).astype("int")
    )

# %%
industry_area.head(1)

# %%
