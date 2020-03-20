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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from tslib.tests import testdata
from tslib.src import tsUtils
from tslib.src.synthcontrol.syntheticControl import RobustSyntheticControl


test_dir = os.path.dirname(testdata.__file__)

prop99Filename = test_dir + "/prop99.csv"

basqueFilename = test_dir + "/basque.csv"

# %%
# BASQUE COUNTRY STUDY
df = pd.read_csv(basqueFilename)
pivot = df.pivot_table(values="gdpcap", index="regionname", columns="year")
pivot = pivot.drop("Spain (Espana)")
dfBasque = pd.DataFrame(pivot.to_records())

allColumns = dfBasque.columns.values

states = list(np.unique(dfBasque["regionname"]))
years = np.delete(allColumns, [0])

basqueKey = "Basque Country (Pais Vasco)"
states.remove(basqueKey)
otherStates = states

yearStart = 1955
yearTrainEnd = 1971
yearTestEnd = 1998

singvals = 1
p = 1.0

trainingYears = []
for i in range(yearStart, yearTrainEnd, 1):
    trainingYears.append(str(i))

testYears = []
for i in range(yearTrainEnd, yearTestEnd, 1):
    testYears.append(str(i))

trainDataMasterDict = {}
trainDataDict = {}
testDataDict = {}
for key in otherStates:
    series = dfBasque[dfBasque["regionname"] == key]

    trainDataMasterDict.update({key: series[trainingYears].values[0]})

    # randomly hide training data
    (trainData, pObservation) = tsUtils.randomlyHideValues(
        copy.deepcopy(trainDataMasterDict[key]), p
    )
    trainDataDict.update({key: trainData})
    testDataDict.update({key: series[testYears].values[0]})

series = dfBasque[dfBasque["regionname"] == basqueKey]
trainDataMasterDict.update({basqueKey: series[trainingYears].values[0]})
trainDataDict.update({basqueKey: series[trainingYears].values[0]})
testDataDict.update({basqueKey: series[testYears].values[0]})

trainMasterDF = pd.DataFrame(data=trainDataMasterDict)
trainDF = pd.DataFrame(data=trainDataDict)
testDF = pd.DataFrame(data=testDataDict)

# model
rscModel = RobustSyntheticControl(
    basqueKey,
    singvals,
    len(trainDF),
    probObservation=1.0,
    modelType="svd",
    svdMethod="numpy",
    otherSeriesKeysArray=otherStates,
)

# fit the model
rscModel.fit(trainDF)

# save the denoised training data
denoisedDF = rscModel.model.denoisedDF()

# predict - all at once
predictions = rscModel.predict(testDF)

# plot
yearsToPlot = range(yearStart, yearTestEnd, 1)
interventionYear = yearTrainEnd - 1
plt.plot(
    yearsToPlot,
    np.append(trainMasterDF[basqueKey], testDF[basqueKey], axis=0),
    color="red",
    label="observations",
)
plt.plot(
    yearsToPlot,
    np.append(denoisedDF[basqueKey], predictions, axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=interventionYear, linewidth=1, color="black", label="Intervention")
plt.ylim((0, 12))
legend = plt.legend(loc="lower right", shadow=True)
plt.title("Abadie et al. Basque Country Case Study - $p = %.2f$" % p)
plt.show()


# %%
# CALIFORNIA PROP 99 STUDY
df = pd.read_csv(prop99Filename)
df = df[df["SubMeasureDesc"] == "Cigarette Consumption (Pack Sales Per Capita)"]
pivot = df.pivot_table(values="Data_Value", index="LocationDesc", columns=["Year"])
dfProp99 = pd.DataFrame(pivot.to_records())

allColumns = dfProp99.columns.values

states = list(np.unique(dfProp99["LocationDesc"]))
years = np.delete(allColumns, [0])

caStateKey = "California"
states.remove(caStateKey)
otherStates = states

yearStart = 1970
yearTrainEnd = 1989
yearTestEnd = 2015

singvals = 3
p = 1.0

trainingYears = []
for i in range(yearStart, yearTrainEnd, 1):
    trainingYears.append(str(i))

testYears = []
for i in range(yearTrainEnd, yearTestEnd, 1):
    testYears.append(str(i))

trainDataMasterDict = {}
trainDataDict = {}
testDataDict = {}
for key in otherStates:
    series = dfProp99[dfProp99["LocationDesc"] == key]

    trainDataMasterDict.update({key: series[trainingYears].values[0]})

    # randomly hide training data
    (trainData, pObservation) = tsUtils.randomlyHideValues(
        copy.deepcopy(trainDataMasterDict[key]), p
    )
    trainDataDict.update({key: trainData})
    testDataDict.update({key: series[testYears].values[0]})

series = dfProp99[dfProp99["LocationDesc"] == caStateKey]
trainDataMasterDict.update({caStateKey: series[trainingYears].values[0]})
trainDataDict.update({caStateKey: series[trainingYears].values[0]})
testDataDict.update({caStateKey: series[testYears].values[0]})

trainMasterDF = pd.DataFrame(data=trainDataMasterDict)
trainDF = pd.DataFrame(data=trainDataDict)
testDF = pd.DataFrame(data=testDataDict)

# model
rscModel = RobustSyntheticControl(
    caStateKey,
    singvals,
    len(trainDF),
    probObservation=1.0,
    modelType="svd",
    svdMethod="numpy",
    otherSeriesKeysArray=otherStates,
)

# fit the model
rscModel.fit(trainDF)

# save the denoised training data
denoisedDF = rscModel.model.denoisedDF()

# predict - all at once
predictions = rscModel.predict(testDF)

# plot
yearsToPlot = range(yearStart, yearTestEnd, 1)
interventionYear = yearTrainEnd - 1
plt.plot(
    yearsToPlot,
    np.append(trainMasterDF[caStateKey], testDF[caStateKey], axis=0),
    color="red",
    label="observations",
)
plt.plot(
    yearsToPlot,
    np.append(denoisedDF[caStateKey], predictions, axis=0),
    color="blue",
    label="predictions",
)
plt.axvline(x=interventionYear, linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title("Abadie et al. Prop 99 Case Study (CA) - $p = %.2f$" % p)
plt.show()


# %%
# WALMART DATA FOR SEVERAL PRODUCT/STORES
df_walmart = pd.read_pickle("df_max_sales.pkl")
# %%
# Convert categorical columns to str to avoid grouping from recreating empty categories
df_monthly = (
    df_walmart.astype({"item_id": str, "store_id": str})
    .groupby(["item_id", "store_id", pd.Grouper(key="date", freq="W")])["sales_units"]
    .sum()
    .reset_index()
)
df_monthly["key"] = df_monthly["item_id"].astype(str) + "_" + df_monthly["store_id"].astype(str)
df_monthly = df_monthly.drop(columns=["item_id", "store_id"])
# %%
df_sales = df_monthly.pivot_table(values="sales_units", index="date", columns="key")

# %%
ts_split = TimeSeriesSplit(n_splits=5)
splits = ts_split.split(df_sales)
# for train, test in splits:
#     print(f"Split Train: {len(train)} rows | Test: {len(test)} rows")
df_sales_idx = df_sales.reset_index().index
split_masks = [
    (df_sales_idx.isin(train_idxs), df_sales_idx.isin(test_idxs))
    for train_idxs, test_idxs in splits
]


# %%
# df_train = df_sales[df_sales.index <= "2015-08-01"]
# df_test = df_sales[df_sales.index > "2015-08-01"]

# %%
# Split treat/donor units
treated_unit = df_sales.columns[0]  # Pick some serie
donor_units = np.array(df_sales.columns[1:])  # Donor series
# donor_units = greater_donors

# Hyperparams
singvals = 10
p = 1

# Model

rsc_model = RobustSyntheticControl(
    seriesToPredictKey=treated_unit,
    kSingularValues=singvals,
    p=p,
    modelType="svd",
    svdMethod="numpy",
    otherSeriesKeysArray=donor_units,
)

clf = GridSearchCV(rsc_model, {"kSingularValues": [1, 3, 5, 10, 15]}, cv=split_masks)
clf.fit(df_sales)

# Select best model as rsc_model
rsc_model = clf.best_estimator_

# Use only the last cross-validation split to test
last_cv_split = split_masks[-1]
df_train = df_sales[last_cv_split[0]]
df_test = df_sales[last_cv_split[1]]

# Fit model
# rsc_model.fit(df_train)

# %%
# Save denoised data
# df_denoised = rsc_model.model.denoisedDF()

# %%
plt.plot(
    df_sales.index, df_sales[treated_unit], color="red", label="observations",
)
plt.plot(
    df_sales.index, rsc_model.predict(df_sales), color="blue", label="predictions",
)
plt.axvline(x=df_test.index[0], linewidth=1, color="black", label="Intervention")
legend = plt.legend(loc="lower left", shadow=True)
plt.title(f"{treated_unit} - p = {p:.2f}")
plt.show()


# %%
mask_weights = np.abs(rsc_model.model.weights) >= 0.2
greater_donors = donor_units[mask_weights]
greater_donors

# %%
plt.plot(df_train.index, df_train[treated_unit], "r:")
plt.plot(df_train.index, df_train[greater_donors])
plt.legend()

# %%
