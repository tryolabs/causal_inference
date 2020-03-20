# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# Load sales
df_sales = pd.read_csv("../makridakis/sales_train_validation.csv")

# %%

cols_id = {
    "item_id": "category",
    "dept_id": "category",
    "cat_id": "category",
    "store_id": "category",
    "state_id": "category",
}

# Target columns after unpivot
cols_sales = {"d": "object", "sales_units": np.uint16, **cols_id}

# Redundant column, drop to save memory
df_sales.drop(columns=["id"], inplace=True)

# Unpivot to convert columns [d_1..d_1913] -> columns [d, sales_units]
df = df_sales.melt(
    id_vars=cols_id.keys(),
    var_name="d",  # Column "d" in calendar.csv corresponds with this
    value_name="sales_units",
)[cols_sales.keys()].astype(cols_sales)


# %%
# Load calendar and merge into df (using d, provides wm_yr_wk to merge prices later)
cols_calendar = {
    "d": "object",
    "wm_yr_wk": np.uint32,
    "event_name_1": "category",
    "event_name_2": "category",
    "event_type_1": "category",
    "event_type_2": "category",
    "snap_CA": np.uint8,
    "snap_TX": np.uint8,
    "snap_WI": np.uint8,
}

df_calendar = pd.read_csv("../makridakis/calendar.csv", parse_dates=["date"], dtype=cols_calendar)

# %%
cols_calendar_filter = ["date"] + list(
    cols_calendar.keys()
)  # Filter redundant columns in calendar
df = df.merge(df_calendar[cols_calendar_filter], on="d", how="left",)
df.drop(columns=["d"], inplace=True)  # No longer needed

# %%
# Load prices and merge into df (using wm_yr_wk from calendar)
cols_prices = {
    "store_id": "category",
    "item_id": "category",
    "wm_yr_wk": np.uint32,
    "sell_price": np.float32,
}

df_prices = pd.read_csv("../makridakis/sell_prices.csv", dtype=cols_prices)
# %%
df = df.merge(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left",)
df.drop(columns=["wm_yr_wk"], inplace=True)

# %%
# Check if there are sales with undefined price
print(f"Sales without shelf price: {len(df[df['sell_price'].isna() & (df['sales_units'] != 0)])}")

# Remove rows without prices (product was not available at that time)
print(f"Rows without shelf price and no sales (product disabled): {df['sell_price'].isna().sum()}")
df = df[~df["sell_price"].isna()]

# %%
df.to_pickle("df_full.pkl")

# %%
df.info()
# %%
# Get products grouped by amount of sales
median_sales = (
    df.groupby(["item_id", "store_id"])["sales_units"].median().sort_values(ascending=False)
)
median_sales.hist()
median_sales


# %%
def plot_sales_price(df_item_store):
    # Avoid filtering id[0] to show better if df_item_store is not properly filtered
    item_id = df_item_store["item_id"].astype(str).unique()
    store_id = df_item_store["store_id"].astype(str).unique()
    title = f"Item: {item_id} | Store: {store_id}"

    fig, ax = plt.subplots()

    ax.plot(df_item_store["date"], df_item_store["sales_units"], "r-", alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(df_item_store["date"], df_item_store["sell_price"], "b-", alpha=0.5)

    ax2.set_ylabel("Shelf price")
    ax.set_ylabel("Sales units")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90)


# %%
for item_id, store_id in median_sales[:30].index:
    # Filter df by item/location
    df_item_store = df[(df["item_id"] == item_id) & (df["store_id"] == store_id)]

    # Plot sales/price as a function of time
    plot_sales_price(df_item_store)

# %%
# Subset of the dataframe to save/train model
mask_X = np.zeros(shape=len(df)).astype(bool)

# E.g: pick products with greater amount of sales
for item_id, store_id in median_sales[:30].index:
    mask_X = mask_X | ((df["item_id"] == item_id) & (df["store_id"] == store_id))

# E.g: remove rows previous to 2014
mask_X = mask_X & (df["date"] >= pd.Timestamp(2014, 1, 1))

# Filter and show statistics of selected data
df_X = df[mask_X]
print(f"Rows to use in X: {mask_X.sum()} / {len(df)} ({100 * mask_X.sum()/len(df):.2f}%)")
print(f"Dates: {df_X['date'].min()}/{df_X['date'].max()} | # items: {df_X['item_id'].nunique()}")


# %%
df_X.to_pickle("df_max_sales.pkl")

# %%
