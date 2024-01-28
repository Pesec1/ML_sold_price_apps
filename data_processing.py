# |%%--%%| <p1ddiJBQIJ|lGqkjAojdm>
r"""°°°
Imports
°°°"""
# |%%--%%| <lGqkjAojdm|3i70Gp4h3z>
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

# |%%--%%| <3i70Gp4h3z|ELzsY1bptO>
r"""°°°
Read data in CSV file
°°°"""
# |%%--%%| <ELzsY1bptO|TW2SMCMQdy>

df_main = pd.read_csv("input_data/sold_flats_2020-09-30.csv")
df_main

# |%%--%%| <TW2SMCMQdy|y4jIEeJSM3>

df_main.info()

# |%%--%%| <y4jIEeJSM3|ZnPErXgBW2>
r"""°°°
Check for duplicates if there are duplicates then clean them
°°°"""
# |%%--%%| <ZnPErXgBW2|H9gN1WWLQh>

df_main.duplicated().sum()
# |%%--%%| <H9gN1WWLQh|hLGKbk3Ou9>
r"""°°°
Check what columns are cat data, what num date to further analyze data
°°°"""
# |%%--%%| <hLGKbk3Ou9|fJzQc0zT6Z>

cat_columns = []
num_columns = []

for column_name in df_main.columns:
    if df_main[column_name].dtypes == object and column_name != "area_total":
        cat_columns += [column_name]
    elif "id" in column_name:
        cat_columns += [column_name]
    else:
        num_columns += [column_name]
print(f"Category data {cat_columns}, count columns = {len(cat_columns)} ")
print(f"Numurical date {num_columns}, count columns = {len(num_columns)} ")


# |%%--%%| <fJzQc0zT6Z|GRtA8O4DNK>
r"""°°°
Look at the statistic about num data 
°°°"""
# |%%--%%| <GRtA8O4DNK|NOE6LHTQZk>

df_main.describe()

# |%%--%%| <NOE6LHTQZk|cdvtXPiZbQ>

num = 5

fig, axes = plt.subplots(5, 5)

# Draw histograms on each subplot
for i in range(5):
    for j in range(5):
        column_name = num_columns[i * 5 + j]
        sns.histplot(data=df_main, x=column_name, bins=20, ax=axes[i, j])

plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.subplot_tool()
plt.show()
# |%%--%%| <cdvtXPiZbQ|FNQgbLWbLP>

plt.figure(figsize=(15, 6))
sns.histplot(data=df_main, x="price", bins=20)


# |%%--%%| <FNQgbLWbLP|EaPCffk4wr>
r"""°°°
Found that price contains zero, which impossible, so clean data that price is zero
°°°"""
# |%%--%%| <EaPCffk4wr|a8meKiYFFv>
zero_price_apps = df_main[(df_main.price == 0)]
df_main = df_main.drop(zero_price_apps.index)
# |%%--%%| <a8meKiYFFv|ABu7ybrp74>

plt.figure(figsize=(15, 6))
sns.histplot(data=df_main, x="price", bins=20, log_scale=True)


# |%%--%%| <ABu7ybrp74|bsUyjiYixi>
r"""°°°
Logically clean data, for example area can't be equal zero
°°°"""
# |%%--%%| <bsUyjiYixi|zpCfBe6o1B>
zero_square_area = df_main[(df_main.area_total) == 0 | (df_main.area_live == 0)]
df_main = df_main.drop(zero_square_area.index)

# Status is always sold so it doesn't matter
df_main = df_main.drop(columns=["status", "loggia"], axis=1)

zero_sold_price = df_main[df_main.sold_price == 0]
df_main = df_main.drop(zero_sold_price.index)

zero_price = df_main[df_main.price == 0]
df_main = df_main.drop(zero_price.index)

low_ceiling_height = df_main[df_main.ceiling_height < 1]
df_main = df_main.drop(low_ceiling_height.index)
df_main.reset_index(drop=True)

# plt.figure(figsize=(15, 6))
# sns.histplot(data=df_main, x="area_total", bins=20, log_scale=True)
# |%%--%%| <zpCfBe6o1B|j26yz9IiHp>
print(df_main.tail())
df_main.info()
#


# |%%--%%| <j26yz9IiHp|abM28kyNVe>
r"""°°°
Clean NAN values
°°°"""
# |%%--%%| <abM28kyNVe|ySJkefc4Ws>
df_main = df_main.dropna(
    subset=[
        "price",
        "sold_price",
        "metro_station_id",
        "floor_num",
        "floors_cnt",
        "wall_id",
    ]
)

df_main["bathrooms_cnt"] = df_main["bathrooms_cnt"].fillna(1)

df_main["closed_yard"] = df_main["closed_yard"].fillna(0)

df_main["building_year"] = df_main.groupby("series_id")["building_year"].fillna(
    method="ffill"
)
df_main["building_year"] = df_main.groupby("district_id")["building_year"].fillna(
    method="ffill"
)
df_main["building_year"] = df_main.groupby("street_id")["building_year"].fillna(
    method="ffill"
)
# to not corrupt data further, it is better to get rid of the remaining Nan values for building_year, because it is really importat stat for analyzes
# better to drop NAN in keep column because not so many values missing and easy to corrupt
df_main = df_main.dropna(subset=["building_year", "keep"])
df_main.info()

# dropping komunal_cost because not enouth data is present to fill. Could fill it by building_id.
# |%%--%%| <ySJkefc4Ws|SmUkqiSbFH>

df_main = df_main.drop(columns=["longitude", "latitude", "komunal_cost"])

# |%%--%%| <SmUkqiSbFH|mJNxDOygzj>

df_main = df_main.dropna(subset=["series_id"])
df_main.info()


# |%%--%%| <mJNxDOygzj|UXuimps0Zh
df_main["rooms_cnt"] = df_main.groupby("area_total")["rooms_cnt"].fillna(method="ffill")
df_main["bedrooms_cnt"] = df_main.groupby("area_total")["bedrooms_cnt"].fillna(
    method="ffill"
)
df_main = df_main.dropna(subset=["rooms_cnt", "bedrooms_cnt"])
df_main.info()
# |%%--%%| <|ouzvXMRdao>

df_main["plate"] = df_main.groupby("series_id")["plate"].fillna(method="ffill")
df_main = df_main.dropna(subset=["plate"])

df_main.info()


# |%%--%%| <ouzvXMRdao|RxrA4WPe1T>


df_main = df_main.reset_index(drop=True)
for i in range(len(df_main)):
    if pd.isnull(df_main.loc[i, "territory"]):
        # If the territory value is missing, skip the current iteration
        pass
    else:
        elem = df_main.loc[i, "territory"]

        # Convert the territory string to a list of words
        words = elem.split(",")

        # Count the number of words in the territory string
        num_words = len(words)

        # Add the number of words to the new territory column
        df_main.loc[i, "territory"] = num_words

# |%%--%%| <RxrA4WPe1T|OElTT0Oj0i>

df_main["territory"] = df_main.groupby("series_id")["territory"].fillna(method="ffill")
df_main = df_main.dropna(subset=["territory"])
df_main.info()
# |%%--%%| <OElTT0Oj0i|y2qTmmOL77>

df_main["area_balcony"] = df_main["area_balcony"].str.replace(r"[^\d\-+\]", "")
df_main["area_balcony"] = df_main["area_balcony"].str.replace(",", ".")

# Convert all values to float.
df_main["area_balcony"] = pd.to_numeric(df_main["area_balcony"], errors="coerce")
# |%%--%%| <y2qTmmOL77|KJ04yMcGsb>
df_main["area_balcony"].fillna(
    df_main.groupby("series_id")["area_balcony"].transform("mean"), inplace=True
)
df_main = df_main.dropna(subset=["area_balcony"])
df_main = df_main.reset_index(drop=True)
df_main.info()
# |%%--%%| <KJ04yMcGsb|F3ou8MH0am>

num = 5

fig, axes = plt.subplots(6, 6)

# Draw histograms on each subplot
for i in range(6):
    for j in range(6):
        column_name = num_columns[i * 5 + j]
        sns.histplot(data=df_main, x=column_name, bins=20, ax=axes[i, j])

plt.subplots_adjust(hspace=0.5, wspace=0.2)
plt.show()
# |%%--%%| <F3ou8MH0am|sJ68PI8JFv>

plt.figure(figsize=(15, 6))
sns.histplot(data=df_main, x="price", bins=20, log_scale=True)


# |%%--%%| <sJ68PI8JFv|jzVpsE1F1w>
# Histogram analyse
question_price = df_main[(df_main.price > 1e4)]
df_main = df_main.drop(question_price.index)


# |%%--%%| <jzVpsE1F1w|WnihlhGszj>
plt.figure(figsize=(15, 6))
sns.histplot(data=df_main, x="sold_price", bins=20, log_scale=True)
# |%%--%%| <WnihlhGszj|SKzhLNmkkv>
question_price_sold = df_main[(df_main.sold_price > 1e4)]
df_main = df_main.drop(question_price_sold.index)


# |%%--%%| <SKzhLNmkkv|vzoPZQfcZZ>
plt.figure(figsize=(15, 6))
sns.histplot(data=df_main, x="area_total", bins=20, log_scale=True)


# |%%--%%| <vzoPZQfcZZ|B09vewk4Fi>
question_area_total = df_main[(df_main.area_total < 1e1) | (df_main.area_total > 1e2)]
df_main = df_main.drop(question_area_total.index)


# |%%--%%| <B09vewk4Fi|g2bb8wjGJB
width = 6
height = int(np.ceil(len(num_columns) / width))
fig, ax = plt.subplots(nrows=height, ncols=width, figsize=(16, 8))

for idx, column_name in enumerate(num_columns):
    plt.subplot(height, width, idx + 1)
    print(column_name)
    sns.histplot(
        data=df_main,
        x=column_name,
        bins=20,
    )

# |%%--%%| <|RctrxEeFqq>
r"""°°°
Further analys of data, to see what we missed on previous steps
°°°"""
# |%%--%%| <|UYpROoL1uE
fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10, 20))

for idx, column_name in enumerate(num_columns):
    plt.subplot(5, 4, idx + 1)
    sns.boxplot(data=df_main, x=column_name)
# |%%--%%| <|usjePduczF>

question_flat_floor = df_main[df_main.flat_on_floor > 180]
df_main = df_main.drop(question_flat_floor.index)

question_floors_cnt = df_main[df_main.floors_cnt > 38]
df_main = df_main.drop(question_floors_cnt.index)

question_rooms_cnt = df_main[df_main.rooms_cnt > 20]
df_main = df_main.drop(question_rooms_cnt.index)

question_bedrooms_cnt = df_main[df_main.bedrooms_cnt > 40]
df_main = df_main.drop(question_bedrooms_cnt.index)

question_building_year = df_main[df_main.building_year < 1900]
df_main = df_main.drop(question_building_year.index)

question_area_live = df_main[df_main.area_live > 110]
df_main = df_main.drop(question_area_live.index)

question_area_kitchen = df_main[df_main.area_kitchen > 45]
df_main = df_main.drop(question_area_kitchen.index)

df_main.info()
# |%%--%%| <usjePduczF|y7JFdpFDpT>

fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(10, 20))

for idx, column_name in enumerate(num_columns):
    plt.subplot(5, 4, idx + 1)
    sns.boxplot(data=df_main, x=column_name)


# |%%--%%| <y7JFdpFDpT|otXXY24Pzf>
cat_columns = []
num_columns = []

for column_name in df_main.columns:
    if df_main[column_name].dtypes == object and column_name != "area_total":
        cat_columns += [column_name]
    elif "id" in column_name:
        cat_columns += [column_name]
    else:
        num_columns += [column_name]
cm = sns.color_palette("vlag", as_cmap=True)


df2 = df_main[num_columns].corr().style.background_gradient(cmap=cm, vmin=-1, vmax=1)
df2.to_html()
# |%%--%%| <otXXY24Pzf|0KjY3ipS40>

df_main[cat_columns].nunique()

# |%%--%%| <0KjY3ipS40|xDV8uStRh7>

counts = df_main.city_id.value_counts()
counts.median()

counts[counts < 100]

# |%%--%%| <xDV8uStRh7|wqk70DYBCA>

rare = counts[(counts.values < 25)]
df_main["city_id"] = df_main["city_id"].replace(rare.index, "Rare")
df_main.city_id.value_counts()

# |%%--%%| <wqk70DYBCA|nL3wzcTizW>

counts = df_main.district_id.value_counts()
counts.median()

counts[counts < 100]


# |%%--%%| <nL3wzcTizW|e8NQb2FUhU>

rare = counts[(counts.values < 20)]
df_main["district_id"] = df_main["district_id"].replace(rare.index, "Rare")
df_main.district_id.value_counts()


# |%%--%%| <e8NQb2FUhU|PUnPW8v2cj>

counts = df_main.street_id.value_counts()
counts.median()

counts[counts < 100]


# |%%--%%| <PUnPW8v2cj|3FTumtphaj>

rare = counts[(counts.values < 20)]
df_main["street_id"] = df_main["street_id"].replace(rare.index, "Rare")
df_main.street_id.value_counts()


# |%%--%%| <3FTumtphaj|3sLrb09cSz>
counts = df_main.metro_station_id.value_counts()
counts.median()

counts[counts < 100]

# |%%--%%| <3sLrb09cSz|WtuoHy8a7s>

rare = counts[(counts.values < 20)]
df_main["metro_staton_id"] = df_main["metro_station_id"].replace(rare.index, "Rare")
df_main.metro_station_id.value_counts()


# |%%--%%| <WtuoHy8a7s|hgtNua5fai>

counts = df_main.builder_id.value_counts()
counts.median()

counts[counts < 100]
# |%%--%%| <hgtNua5fai|1zACX9RXuW>

rare = counts[(counts.values < 5)]
df_main["builder_id"] = df_main["builder_id"].replace(rare.index, "Rare")
df_main.builder_id.value_counts()


# |%%--%%| <1zACX9RXuW|us5b8K9bjr>
r"""°°°
Saving)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))GG)
°°°"""
# |%%--%%| <us5b8K9bjr|BgRwl9tmxN>
df_main.to_csv(
    "D:\\testfolder\\test_exercise\\output_data\\sold_flats_veryyyy_clean2.csv",
    index=False,
)
# |%%--%%| <BgRwl9tmxN|hqDi5aAxsF>

df_main["closed_yard"] = df_main["closed_yard"].map({"yes": 1, "no": 0})

# |%%--%%| <hqDi5aAxsF|q4f2nmaLcK>

df_se = df_main.copy()
df_se[cat_columns] = df_se[cat_columns].astype("category")

for _, column_name in enumerate(cat_columns):
    df_se[column_name] = df_se[column_name].cat.codes

df_se.info()

# |%%--%%| <q4f2nmaLcK|PgIDxtKuIV>


df_se.head()

# |%%--%%| <PgIDxtKuIV|xz4vvOyK5H>
# takes too long to plot
# sns.pairplot(data=df_se.sample(500), hue="type")
# |%%--%%| <xz4vvOyK5H|GlfA3Ksu6B>

df_ohe = df_main.copy()
df_ohe = pd.get_dummies(df_ohe)
df_ohe.tail()

# |%%--%%| <GlfA3Ksu6B|goHKf0Js15>

df_ohe.info()

# |%%--%%| <goHKf0Js15|uEY2thQPBM>

df_se.to_csv(
    "D:\\testfolder\\test_exercise\\output_data\\sold_flats_cat_num.csv", index=False
)
df_ohe.to_csv(
    "D:\\testfolder\\test_exercise\\output_data\\sold_flats_oneshot.csv", index=False
)
