# %%
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl
from tslib.src.synthcontrol.multisyntheticControl import MultiRobustSyntheticControl


# # %%
# # Uncomment to generate weekly data.
# df_walmart_total = pd.read_pickle("df_full.pkl")

# # Convert categorical columns to str to avoid grouping from recreating empty categories
# df_weekly = (
#     df_walmart_total.astype({"item_id": str, "store_id": str})
#     .groupby(["store_id", "dept_id", pd.Grouper(key="date", freq="W")])
#     .agg({"sales_units": "sum"})
#     .reset_index()
# )

# df_weekly.to_pickle("df_weekly.pkl")

# %%
# Load weekly data
df_weekly = pd.read_pickle("df_weekly.pkl")
# %%
# We represent different metrics as different categories

# Food metrics
df_weekly_foods1 = df_weekly[df_weekly["dept_id"] == "FOODS_1"]
df_weekly_foods2 = df_weekly[df_weekly["dept_id"] == "FOODS_2"]
df_weekly_foods3 = df_weekly[df_weekly["dept_id"] == "FOODS_3"]

# Hobbies metrics
df_weekly_hobbies1 = df_weekly[df_weekly["dept_id"] == "HOBBIES_1"]
df_weekly_hobbies2 = df_weekly[df_weekly["dept_id"] == "HOBBIES_2"]

# Household metrics
df_weekly_household1 = df_weekly[df_weekly["dept_id"] == "HOUSEHOLD_1"]
df_weekly_household2 = df_weekly[df_weekly["dept_id"] == "HOUSEHOLD_2"]

# %%
# Let's start with foods
df_sales_foods1 = df_weekly_foods1.pivot_table(
    values="sales_units", index="date", columns="store_id"
)
df_sales_foods2 = df_weekly_foods2.pivot_table(
    values="sales_units", index="date", columns="store_id"
)
df_sales_foods3 = df_weekly_foods3.pivot_table(
    values="sales_units", index="date", columns="store_id"
)

# %%

train_end = "2015-04-24"
# metric 1
df_train_foods1 = df_sales_foods1[df_sales_foods1.index <= train_end]
df_test_foods1 = df_sales_foods1[df_sales_foods1.index > train_end]

# metric 2
df_train_foods2 = df_sales_foods2[df_sales_foods2.index <= train_end]
df_test_foods2 = df_sales_foods2[df_sales_foods2.index > train_end]

# metric 1
df_train_foods3 = df_sales_foods3[df_sales_foods3.index <= train_end]
df_test_foods3 = df_sales_foods3[df_sales_foods3.index > train_end]


# %%
# Rank comparision

df_train_total = pd.concat(
    [
        (df_train_foods1 - df_train_foods1.mean()) / df_train_foods1.std(),
        (df_train_foods2 - df_train_foods2.mean()) / df_train_foods2.std(),
        (df_train_foods3 - df_train_foods3.mean()) / df_train_foods3.std(),
    ],
    axis=1,
)

# SV decomposition with centered data
(_, s1, _) = np.linalg.svd((df_train_foods1 - df_train_foods1.mean()) / df_train_foods1.std())
(_, s2, _) = np.linalg.svd((df_train_foods2 - df_train_foods2.mean()) / df_train_foods1.std())
(_, s3, _) = np.linalg.svd((df_train_foods3 - df_train_foods3.mean()) / df_train_foods1.std())
(_, stot, _) = np.linalg.svd(df_train_total)

spectrum1 = np.cumsum(s1 ** 2) / np.sum(s1 ** 2)
spectrum2 = np.cumsum(s2 ** 2) / np.sum(s2 ** 2)
spectrum3 = np.cumsum(s3 ** 2) / np.sum(s3 ** 2)
spectrum_tot = np.cumsum(stot ** 2) / np.sum(stot ** 2)

plt.figure()
plt.plot(spectrum1, label="metric 1")
plt.plot(spectrum2, label="metric 2")
plt.plot(spectrum3, label="metric 3")
plt.plot(spectrum_tot, label="combined")
plt.title("Cumulative energy")
plt.xlabel("Ordered Singular Values")
plt.legend()
plt.grid()

plt.figure()
plt.plot(s1 ** 2, label="metric 1")
plt.plot(s2 ** 2, label="metric 2")
plt.plot(s3 ** 2, label="metric 3")
plt.plot(stot ** 2, label="combined")
plt.title("Singular Value Spectrum")
plt.xlabel("Ordered Singular Values")
plt.ylabel("Energy")
plt.legend()
plt.grid()

# %%
# It seems that metric 3 does not sum for the analysis. We'll skip it when using mRSC

# %%
# Split treat/donor units
treated_unit = "CA_1"

# donor_units must be list and not numpy array
donor_units = list(df_train_foods1.columns[df_train_foods1.columns != "CA_1"])

# Hyperparams
singvals = 4
p = 1.0

# %%
# Vanilla RSC

# RSC for metric 1
rscmodel1 = RobustSyntheticControl(
    treated_unit,
    singvals,
    p=1.0,
    svdMethod="numpy",
    otherSeriesKeysArray=donor_units,
)

# fit the model
rscmodel1.fit(df_train_foods1)

df_denoised_rsc1 = rscmodel1.model.denoisedDF()
predictionsRSC1 = rscmodel1.predict(df_test_foods1)

rscRMSE1 = np.sqrt(np.mean((predictionsRSC1 - df_test_foods1[treated_unit]) ** 2))
print("RMSE of RSC on metric 1:", rscRMSE1)

# %%
plt.plot(
    df_sales_foods1.index, df_sales_foods1[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales_foods1.index,
    np.append(
        df_denoised_rsc1[treated_unit].iloc[0 : len(df_train_foods1)], predictionsRSC1, axis=0
    ),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test_foods2.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")

# %%

# RSC for metric 2
rscmodel2 = RobustSyntheticControl(
    treated_unit,
    singvals,
    p=1.0,
    svdMethod="numpy",
    otherSeriesKeysArray=donor_units,
)

# fit the model
rscmodel2.fit(df_train_foods2)

df_denoised_rsc2 = rscmodel2.model.denoisedDF()
predictionsRSC2 = rscmodel2.predict(df_test_foods2)

rscRMSE2 = np.sqrt(np.mean((predictionsRSC2 - df_test_foods2[treated_unit]) ** 2))
print("RMSE of RSC for metric 2:", rscRMSE2)

# %%
plt.plot(
    df_sales_foods2.index, df_sales_foods2[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales_foods1.index,
    np.append(
        df_denoised_rsc2[treated_unit].iloc[0 : len(df_train_foods2)], predictionsRSC2, axis=0
    ),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test_foods2.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")

# %%
# RSC for metric 3
rscmodel3 = RobustSyntheticControl(
    treated_unit,
    singvals,
    p=1.0,
    svdMethod="numpy",
    otherSeriesKeysArray=donor_units,
)

# fit the model
rscmodel3.fit(df_train_foods3)

df_denoised_rsc3 = rscmodel3.model.denoisedDF()
predictionsRSC3 = rscmodel3.predict(df_test_foods3)

rscRMSE3 = np.sqrt(np.mean((predictionsRSC3 - df_test_foods3[treated_unit]) ** 2))
print("RMSE of RSC for metric 3:", rscRMSE3)


# %%
plt.plot(
    df_sales_foods3.index, df_sales_foods3[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales_foods1.index,
    np.append(
        df_denoised_rsc3[treated_unit].iloc[0 : len(df_train_foods3)], predictionsRSC3, axis=0
    ),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test_foods3.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")

# %%
# mRSC
nbrMetrics = 2
weightsArray = [1.0, 1.0]
singvals = 4

# Model
# TODO: Check for M parameter that is commented in multisyntheticControl.py
mrsc_model = MultiRobustSyntheticControl(
    nbrMetrics,
    weightsArray,
    treated_unit,  # seriesToPredictKey,
    singvals,  # kSingularValuesToKeep
    len(df_train_foods1),  # M
    p=1.0,
    modelType="svd",
    svdMethod="numpy",
    otherSeriesKeysArray=donor_units,
)

# Fit model
mrsc_model.fit([df_train_foods1, df_train_foods2])

# Save denoised data
df_denoised = mrsc_model.model.denoisedDF()

# Predictions
combinedPredictionsArray = mrsc_model.predict(
    [df_test_foods1[donor_units], df_test_foods2[donor_units]]
)

# split the predictions for the metrics
predictionsmRSC_1 = combinedPredictionsArray[0]
predictionsmRSC_2 = combinedPredictionsArray[1]
# predictionsmRSC_3 = combinedPredictionsArray[2]

# compute RMSE
mrscRMSE1 = np.sqrt(np.mean((predictionsmRSC_1 - df_test_foods1[treated_unit]) ** 2))
mrscRMSE2 = np.sqrt(np.mean((predictionsmRSC_2 - df_test_foods2[treated_unit]) ** 2))
# mrscRMSE3 = np.sqrt(np.mean((predictionsmRSC_3 - df_test_foods3[treated_unit]) ** 2))

print("\n\n *** mRSC rmse1:")
print(mrscRMSE1)

print("\n\n *** mRSC rmse2:")
print(mrscRMSE2)

# print("\n\n *** mRSC rmse3:")
# print(mrscRMSE3)

# %%
plt.plot(
    df_sales_foods1.index, df_sales_foods1[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales_foods1.index,
    np.append(df_denoised[treated_unit].iloc[0 : len(df_train_foods1)], predictionsmRSC_1, axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test_foods1.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")

# %%
plt.plot(
    df_sales_foods2.index, df_sales_foods2[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales_foods2.index,
    np.append(df_denoised[treated_unit].iloc[len(df_train_foods2) :], predictionsmRSC_2, axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=df_test_foods2.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")


# %%
mask_weights = np.abs(mrsc_model.model.weights) >= 0.2
greater_donors = np.array(donor_units)[mask_weights]
greater_donors

# %%
plt.plot(df_train_foods1.index, df_train_foods1[treated_unit], "r:", label="treated unit")
plt.plot(df_train_foods1.index, df_train_foods1[greater_donors], label="greater donors")
plt.legend()

# %%
