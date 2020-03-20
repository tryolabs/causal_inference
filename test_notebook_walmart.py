#############################################################
#
# Robust Synthetic Control Tests (based on SVD)
#
# Adapted to run as VSCode-style notebook from:
#   - testScriptSynthControlSVD.py
#
#############################################################
# %%
import os
import numpy as np
import pandas as pd
import copy

from matplotlib import pyplot as plt
import seaborn as sns

from tslib.tests import testdata
from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src.synthcontrol.multisyntheticControl import MultiRobustSyntheticControl


 # %%
# WALMART DATA FOR SEVERAL PRODUCT/STORES
df_walmart = pd.read_pickle("df_max_sales_2.pkl")
# %%

df_walmart["item_id"] = df_walmart["item_id"].astype(str)
df_walmart["store_id"] = df_walmart["store_id"].astype(str)

df_monthly = (
    df_walmart.groupby(["item_id", "store_id", pd.Grouper(key="date", freq="M")])["sales_units"]
    .sum()
    .reset_index()
)

df_monthly.groupby("item_id")["store_id"].nunique()
# %%
# Plot particular product per store
def plot_product_by_store(df, item_id) :
    df_sku = df[df["item_id"]==item_id]
    ax = sns.lineplot(x="date", y="sales_units", hue="store_id", data=df_sku)
    ax.set_title(f"item_id:{item_id}")


plot_product_by_store(df_monthly, "FOODS_3_080")  

foods_items = ['FOODS_3_001', 'FOODS_3_002', 'FOODS_3_004', 'FOODS_3_005',
       'FOODS_3_007', 'FOODS_3_008', 'FOODS_3_010', 'FOODS_3_011',
       'FOODS_3_012', 'FOODS_3_013', 'FOODS_3_014', 'FOODS_3_015',
       'FOODS_3_017', 'FOODS_3_019', 'FOODS_3_020', 'FOODS_3_021']   


for item in foods_items:
    plt.figure()
    plot_product_by_store(df_monthly, item)
# %%
# Example:
# target: product-store
# donors: other stores same product
# metrics: other products  
target_product = "FOODS_3_090" + "CA_1"
donors = "store_id"
metric = "item_id"
# %%
df_monthly["key"] = df_monthly["item_id"].astype(str) + "_" + df_monthly["store_id"].astype(str)
df_sales = df_monthly.pivot_table(values="sales_units", index="date", columns="key")

# %%
df_train = df_sales[df_sales.index <= "2015-08-01"]
df_test = df_sales[df_sales.index > "2015-08-01"]

# %%
# Split treat/donor units
treated_unit = "FOODS_3_090_CA_1" # Pick some serie (item+store)
# Donors are same item different store
all_series_label_m1 = [col for col in df_sales if col.startswith('FOODS_3_090')] 
# Remove treated unit from donors list
donor_units_m1 = set(list(all_series_label_m1)) - set([treated_unit])

# Metrics are other products
metric_2 = "FOODS_3_586"

all_series_label_m2 = [col for col in df_sales if col.startswith("FOODS_3_586")]

matrix_1_train = df_train[all_series_label_m1]
matrix_2_train = df_train[all_series_label_m2]

matrix_1_test = df_test[all_series_label_m1]
matrix_2_test = df_test[all_series_label_m2]

N = 9 # N: number of other donors
keySeriesLabel = '0'
otherSeriesLabels = []
for ind in range(1, N+1):
    otherSeriesLabels.append(str(ind))

AllSeriesLabels = []
for ind in range(0, N+1):
    AllSeriesLabels.append(str(ind))


matrix_1_train.columns = AllSeriesLabels
matrix_2_train.columns = AllSeriesLabels
matrix_1_test.columns = AllSeriesLabels
matrix_2_test.columns = AllSeriesLabels

# %%
# mRSC
singvals = 3
p = 1
weights = np.array([1, 0])
N = 9


mrsc_model = MultiRobustSyntheticControl(nbrMetrics=2,
    weightsArray=weights,
    seriesToPredictKey=keySeriesLabel,
    kSingularValuesToKeep=singvals,
    M=len(df_train),
    probObservation=p,
    modelType="svd",
    svdMethod="numpy",
    otherSeriesKeysArray=otherSeriesLabels,
)

# Fit model
mrsc_model.fit([matrix_1_train, matrix_2_train])

# %%
# Save denoised data
df_denoised = mrsc_model.model.denoisedDF()
df_denoised_m1 = df_denoised.iloc[0:19, : ]

# Predictions
predictions = mrsc_model.predict([matrix_1_test, matrix_2_test])
# mean_predictions = (predictions[0] + predictions[1])/2
# %%
# Plot predictions M1
plt.plot(
    df_sales.index, df_sales[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales.index,
    np.append(df_denoised_m1[keySeriesLabel], predictions[0], axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")
plt.show()
# %%
# Plot predictions M2
plt.plot(
    df_sales.index, df_sales[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales.index,
    np.append(df_denoised_m1[keySeriesLabel], predictions[1], axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")
plt.show()

# %%
# compute RMSE

mrscRMSE1 = np.sqrt(np.mean((predictions[0]- df_test[treated_unit])**2))
mrscRMSE2 = np.sqrt(np.mean((predictions[1] - df_test[treated_unit])**2))
# %%
# SVD check
df_train_total = pd.concat([(matrix_1_train) - np.mean(matrix_1_train), (matrix_1_train) - np.mean(matrix_2_train)], axis=1)
(_, s1, _) = np.linalg.svd((matrix_1_train) - np.mean(matrix_1_train))
(_, s2, _) = np.linalg.svd((matrix_2_train) - np.mean(matrix_2_train))
(_, stot, _) = np.linalg.svd(df_train_total)

spectrum1 = np.cumsum(np.power(s1, 2))/np.sum(np.power(s1, 2))
spectrum2 = np.cumsum(np.power(s2, 2))/np.sum(np.power(s2, 2))
spectrum_tot = np.cumsum(np.power(stot, 2))/np.sum(np.power(stot, 2))
plt.plot(spectrum1, label="metric1")
plt.plot(spectrum2, label="metric2")
plt.plot(spectrum_tot, label="combined")
plt.grid()
plt.legend()
plt.title("Cumulative energy")


plt.figure()
plt.plot(np.power(s1, 2),  label="metric1")
plt.plot(np.power(s2, 2),  label="metric2")
plt.plot(np.power(stot, 2), label="combined")
plt.grid()
plt.xlabel("Ordered Singular Values") 
plt.ylabel("Energy")
plt.title("Singular Value Spectrum")
plt.legend()

# %%
