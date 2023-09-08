"""
Title: data_reduction_comparison.py

Description:
    Compare the performance of the various dimensionality reduction techniques
    using cross-validation with random forests

Author: Andrew Zehr

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score
from utils.data_handling import read_JFJ_data, perform_pca
from utils.model_evaluation import visualize_feature_separation
from utils.constants import data_folder


def cross_validation_accuracy(data, dust_event_info, num_folds = 5):
    y = dust_event_info.sde_event[data.index]
    df = data.copy()
    df["sde"] = y
    
    kf = KFold(n_splits = num_folds, shuffle = False)
    results = []
    
    for trainIndex, testIndex in kf.split(df):
        train_X = df.iloc[trainIndex].drop("sde", axis = 1)
        test_X = df.iloc[testIndex].drop("sde", axis = 1)
        
        train_y = df.iloc[trainIndex].sde
        test_y = df.iloc[testIndex].sde
        
        rf = RandomForestClassifier(class_weight="balanced", n_estimators = 250, random_state=42)
        rf.fit(train_X, train_y)
        
        y_pred = rf.predict(test_X)
        
        results.append(balanced_accuracy_score(test_y, y_pred))
    
    cv_accuracy = np.mean(results)
    cv_std = np.std(results)
    
    return cv_accuracy, cv_std, results
    
    


dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")

dataFull = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED.csv")["2020-01-01 00:00:00":"2020-12-31 23:59:59"].dropna()
data40Bins = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_May2023.csv")["2020-01-01 00:00:00":"2020-12-31 23:59:59"].dropna()
data3Bins = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_reducedVersion_May2023.csv")["2020-01-01 00:00:00":"2020-12-31 23:59:59"].dropna()

accuracyFull, stdFull, resultsFull = cross_validation_accuracy(dataFull, dust_event_info, num_folds = 6)
accuracy40Bins, std40Bins, results40Bins = cross_validation_accuracy(data40Bins, dust_event_info, num_folds = 6)
accuracy3Bins, std3Bins, results3Bins = cross_validation_accuracy(data3Bins, dust_event_info, num_folds = 6)


data40Bins_pca = perform_pca(data40Bins, new_dim = 0.95)

visualize_feature_separation(data40Bins_pca, dust_event_info, "40bins_95variance_pca.pdf")

# Now do the same with Trex Feature selection
trex_full_data = read_JFJ_data(r"\processed\cleaned_May2023_dist_features.csv")["2020-01-01 00:00:00": "2020-12-31 23:59:59"].dropna()
trex_features = pd.read_csv(r"trex_selected_features.csv", index_col = 0)

# V_D40 is dropped due to concerns about instrument anomalies
trex10 = trex_full_data.loc[:, trex_features.loc[str(0.01)][trex_features.loc[str(0.01)] == 1].index].drop("V_D40", axis = 1)
trex20 = trex_full_data.loc[:, trex_features.loc[str(0.1)][trex_features.loc[str(0.1)] == 1].index].drop("V_D40", axis = 1)
trex25 = trex_full_data.loc[:, trex_features.loc[str(0.8)][trex_features.loc[str(0.8)] == 1].index].drop("V_D40", axis = 1)

trex2 = trex_full_data[["Vfrac_coarse", "AE_SSA"]]

accuracy10, std10, results10 = cross_validation_accuracy(trex10, dust_event_info, num_folds = 6)
accuracy20, std20, results20 = cross_validation_accuracy(trex20, dust_event_info, num_folds = 6)
accuracy25, std25, results25 = cross_validation_accuracy(trex25, dust_event_info, num_folds = 6)

accuracy2, std2, results2 = cross_validation_accuracy(trex2, dust_event_info, num_folds = 6)


# Plot the differences in bins
midDiameters_reduced = pd.read_csv(
    data_folder + r"\raw\Jungfraujoch\midpoint_diameters_size_distr_JFJ_2015_to_2021_May2023.csv", header = None).values.squeeze()

midDiameters_original= pd.read_csv(
    data_folder + r"\raw\Jungfraujoch\midpoint_diameters_size_distr_JFJ_2020.csv", header = None).values.squeeze()




nsd_cols_original = []
vsd_cols_original = []
for col in dataFull.columns:
    if "V_D" in col:
        vsd_cols_original.append(col)
    elif "D" in col:
        nsd_cols_original.append(col)


nsd_cols_reduced = []
vsd_cols_reduced = []
for col in data40Bins.columns:
    if "V_D" in col:
        vsd_cols_reduced.append(col)
    elif "D" in col:
        nsd_cols_reduced.append(col)


# Pick a random hour to visualize
time = 4000

dataFull_entry = dataFull.iloc[time]
data40Bins_entry = data40Bins.iloc[time]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

sns.lineplot(x = np.log10(midDiameters_original).squeeze(), y = dataFull_entry[nsd_cols_original].values, ax = ax1) 
sns.lineplot(x = np.log10(midDiameters_reduced).squeeze(), y = data40Bins_entry[nsd_cols_reduced].values, ax = ax2) 



fig.suptitle(f'Aerosol Number Distribution   ({dataFull_entry.name})', fontsize=16)
ax1.set_title('Original data')
ax2.set_title('40 bins')
ax1.set_ylabel("dN/dlogDp")
ax2.set_ylabel("dN/dlogDp")
fig.supxlabel("log-Diameter (log-micrometers)")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

sns.lineplot(x = np.log10(midDiameters_original).squeeze(), y = dataFull_entry[vsd_cols_original].values, ax = ax1) 
sns.lineplot(x = np.log10(midDiameters_reduced).squeeze(), y = data40Bins_entry[vsd_cols_reduced].values, ax = ax2) 

fig.suptitle(f'Aerosol Volume Distribution  ({dataFull_entry.name})', fontsize=16)
ax1.set_title('Original data')
ax2.set_title('40 bins')
fig.supxlabel("log-Diameter (log-micrometers)")



