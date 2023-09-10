"""
Title: data_preprocessing.py
    
Author: Andrew Zehr

Date Created: 02.03.23

Description:
    Preprocess data: cleaning, dimensionality reduction, seasonality and trend 
    removal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.model_evaluation import evaluate_model, visualize_feature_seperation

from utils.data_handling import (
    read_JFJ_data, data_cleaner, size_dist_handler, deseasonalize, drop_pca_outliers
    )

from utils.constants import (
    data_folder
    )

# Read in 2020 data
data2020 = read_JFJ_data(r"\raw\Jungfraujoch\SDE_2020_raw_data.csv")

# Use 2020 midpoints, but same as to 2015
midDiameters = pd.read_csv(data_folder + r"\raw\Jungfraujoch\midpoint_diameters_size_distr_JFJ_2020.csv", header = None).values

data2020 = data_cleaner(data2020, midDiameters)


# Import data going back to 2016 (currently without labels)
# This data is only complete (with size and volume distributions) back to November 23, 2016
rawData_2016_2020 = read_JFJ_data(r"\raw\Jungfraujoch\aerosol_data_JFJ_2015_to_2021.csv",
                                  date_range = ["2016-11-16 14:00:00", "2030-01-01 00:00:00"])



# Cleans and prepares data (see preprocess_helper_functions for details)
# Note: The cleaned data set still contains nan by design
rawdata_2016_2020_dist = size_dist_handler(rawData_2016_2020, midDiameters)

# Use a pretty aggresive filter on the number distribution threshold 
data_2016_2020 = data_cleaner(rawdata_2016_2020_dist, midDiameters, Ntot_thresh = 8000)


"""
Rework this

# p = 1953 is a sample during a dust event (picked arbitrarily)
p = 1953

# Plot sample distribution of size and volume distribution 
fig1 = plt.figure()
ax = fig1.add_subplot(3,1,1)
plt.plot(midDiameters[0], data2020[p-1:p][number_dist_cols].values[0])
plt.title("Particle Count Distribution")

ax = fig1.add_subplot(3,1,3) 
plt.plot(midDiameters[0], data2020[p-1:p][volume_dist_cols].values[0]) 
plt.title("Particle Volume Distribution")
plt.xlabel("Diameter (micrometers)")


# Plot sample distribution of size and volume distribution on a log scale
fig1 = plt.figure()
ax = fig1.add_subplot(3,1,1)
plt.plot(np.log10(midDiameters[0]), data2020[p-1:p][number_dist_cols].values[0])
plt.title("Particle Count Distribution")

ax = fig1.add_subplot(3,1,3) 
plt.plot(np.log10(midDiameters[0]), data2020[p-1:p][volume_dist_cols].values[0]) 
plt.title("Particle Volume Distribution")
plt.xlabel("log-Diameter (log-micrometers)")
"""




"""
Do PCA for entire data, since 2016
"""

# Adjust for seasonality (using only data before 2020 so there isn't any
# data leakage)

# Drop dates with weird data spikes until better solution found
spike_points = ["2018-03-14 07:00:00",
                "2018-06-30 17:00:00",
                "2018-08-28 02:00:00",
                "2018-08-28 03:00:00",
                "2019-01-09 13:00:00",
                "2020-03-11 09:00:00",
                "2020-12-03 14:00:00"]



scaler = StandardScaler()

X_all = data_2016_2020.dropna(axis=0)

X_all_scaled = scaler.fit_transform(X_all)

X_all_index = X_all.index

# Select 12 because it accounts for 95 percent of variance 
# (while only using less than 4% of original features)
pca = PCA(n_components = 12)
pca_fit = pca.fit(X_all_scaled)
X_all_pca = pca_fit.transform(X_all_scaled)

data_2016_2020_pca = pd.DataFrame(data = X_all_pca, index = X_all_index, columns = range(1,13))

data_2016_2020_pca["month"] = (data_2016_2020_pca.index).month

data_2016_2020_pca_log = np.log10((data_2016_2020_pca - data_2016_2020_pca.min() + 1))

data_2016_2020_pca.iloc[:, :4].plot(subplots=True)
data_2016_2020_pca.iloc[:, 4:8].plot(subplots=True)
data_2016_2020_pca.iloc[:, 8:-1].plot(subplots=True)




# Save this data (this has no seasonality adjustment)
# data_2016_2020_pca.to_csv(data_folder + r"\processed\data_2016_2020_pca.csv")


# This data will be used to calculate the monthly averages
std_devs = data_2016_2020_pca.std()


# Look at only data up until 2020 to calculate monthly averages
# (avoid data leakage from test set)
data_2016_2019_pca = data_2016_2020_pca[:"2019-12-31 23:00:00"]

monthAvg = pd.DataFrame(data_2016_2019_pca.groupby(by = "month").mean())


# Preprocess 2020 Data:
data2020_pca = data_2016_2020_pca[
    "2020-01-01 00:00:00":"2020-12-31 23:59:00"]



data2020_pca_desea = data2020_pca.copy()

# Subtract monthly average from all the columns (using data from before 2020 to avoid leakage)
for i in (range(1,13)):
    data2020_pca_desea.loc[:,i] = data2020_pca[i] - monthAvg.loc[
        data2020_pca_desea["month"], i].values
    

data2020_pca_desea_log = np.log10((data2020_pca_desea - data2020_pca_desea.min() + 1)).drop(["month"], axis = 1)

    
data2020_pca_desea["sde_event"] = data2020["sde_event"]
data2020_pca_desea["sde_event_nr"] = data2020["sde_event_nr"]
data2020_pca_desea["Confidence"] = data2020["Confidence"]



data2020_pca_desea.iloc[:,:4].plot(subplots = True)
data2020_pca_desea.iloc[:,4:8].plot(subplots = True)
data2020_pca_desea.iloc[:,8:12].plot(subplots = True)

# data2020_pca_desea.to_csv(data_folder + r"\processed\data2020_preprocessed.csv")




# Try Kernel PCA
kern_pca = KernelPCA(n_components=20, kernel = "rbf", eigen_solver = "randomized")

kern_pca_fit = kern_pca.fit(X_all_scaled)

data_kernPCA = kern_pca_fit.transform(X_all_scaled)

kernPCA_df = pd.DataFrame(data = data_kernPCA[:,:12], index = X_all_index, columns = range(1,13))


eigen_val = kern_pca_fit.eigenvalues_

# Save data
# kernPCA_df.to_csv(data_folder + r"\processed\data2020_kernelPCA.csv")


# Do seasonality estimation and removal with:
'''data_2016_2020'''


df = data_2016_2020.copy()
df = df.replace([np.inf, -np.inf], np.nan, inplace=False)
df = df.interpolate()

df_deseason = deseasonalize(df, method = "smooth", h = 500)


scaler = StandardScaler()
scaler_fit = scaler.fit(df_deseason)

scaled_deseasonal_data = scaler_fit.transform(df_deseason)

pca = PCA(n_components = 12)
pca_fit = pca.fit(scaled_deseasonal_data)
deseasonal_pca = pca_fit.transform(scaled_deseasonal_data)


df_deseasonal_pca = pd.DataFrame(data = deseasonal_pca, index = df_deseason.index, columns = range(1,13))
df_deseasonal_pca_drop = drop_pca_outliers(df_deseasonal_pca)


# df_deseasonal_pca_drop.to_csv(data_folder + r"\processed\deseason_pca.csv")


from scipy.linalg import fractional_matrix_power

# Look at the cleaned data with 40 bins
data = read_JFJ_data(
    r"\processed\cleaned_May2023_dist_features.csv")
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

midDiameters = pd.read_csv(
    data_folder + r"\raw\Jungfraujoch\midpoint_diameters_size_distr_JFJ_2015_to_2021_May2023.csv"
    ).values





# Whiten the distribution variables (y = sigma^(-1/2) * x)
nsd_cols = [c for c in data.columns if c.startswith("D")]
vsd_cols = [c for c in data.columns if c.startswith("V_D")]

other_cols = [c for c in data.columns if c not in nsd_cols and c not in vsd_cols]

other_data = data.loc[:, other_cols]

scaler = StandardScaler()
other_data_scaled = pd.DataFrame(scaler.fit_transform(other_data), 
                                 columns = other_data.columns,
                                 index = other_data.index)

# Center the data
num_dist_data = data.loc[:, nsd_cols]
num_dist_means = np.mean(num_dist_data)
num_dist_data_center = num_dist_data - num_dist_means


vol_dist_data = data.loc[:, vsd_cols]
vol_dist_means = np.mean(vol_dist_data)
vol_dist_data_center = vol_dist_data - vol_dist_means


n = data.shape[0]

# Calculate var/covar matrices
nsd_cov = (1 / (n - 1)) * np.matmul(num_dist_data_center.values.T,
                                    num_dist_data_center.values)

vsd_cov = (1 / (n - 1)) * np.matmul(vol_dist_data_center.values.T, 
                                    vol_dist_data_center.values)



nsd_data_whitened = pd.DataFrame(np.matmul(num_dist_data_center.values,
                              fractional_matrix_power(nsd_cov, -0.5)),
                                 columns = num_dist_data.columns,
                                 index = num_dist_data.index)



vsd_data_whitened = pd.DataFrame(np.matmul(vol_dist_data_center.values,
                              fractional_matrix_power(vsd_cov, -0.5)),
                                 columns = vol_dist_data.columns,
                                 index = vol_dist_data.index)

data_dist_whitened = pd.concat(
    [other_data_scaled, nsd_data_whitened, vsd_data_whitened], axis = 1
    )


pca = PCA(n_components = 40)
pca_fit = pca.fit(data_dist_whitened)

plt.plot(np.cumsum(pca_fit.explained_variance_ratio_))

data40_bins_whitened_pca = pd.DataFrame(pca_fit.transform(data_dist_whitened),
                                        index = data_dist_whitened.index)

# data40_bins_whitened_pca.to_csv(data_folder + r"\processed\cleaned_data_may2023_pca.csv")

dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
labels2020_all = dust_event_info.sde_event






split_date = "2020-06-30 23:59:59"

lda_data2020 = data_dist_whitened["2020-01-01 00:00:00":]

# Try transforming the data to normal to fit the LDA
quant_transform = QuantileTransformer(output_distribution='normal')
data_transformed = pd.DataFrame(quant_transform.fit_transform(lda_data2020), 
                                index = lda_data2020.index,
                                columns = lda_data2020.columns)


"""Change this to change if you look at data with quantile transform or not"""
data_lda = lda_data2020.copy()

"""Start of Block"""
lda_train = data_lda[:split_date]
lda_test = data_lda[split_date:]

y_train_lda = labels2020_all[lda_train.index]
y_test_lda = labels2020_all[lda_test.index]

y_all = labels2020_all[data_lda.index]



# Try doing LDA to see if it can seperate the feature distributions for dust events and non-dust events
lda = LinearDiscriminantAnalysis()
lda.fit(lda_train, y_train_lda)

predict_lda = lda.predict(lda_test)
predict_probs = lda.predict_proba(lda_test)

pred = [0 if i > 0.5 else 1 for i in predict_probs[:,0]]

evaluate_model(dust_event_info, pd.Series(pred, index = lda_test.index))

predict_lda_train = lda.predict(lda_train)
evaluate_model(dust_event_info, pd.Series(predict_lda_train, index = lda_train.index))



lda_transform_data = pd.DataFrame(lda.transform(data_lda), index = data_lda.index)


visualize_feature_seperation(lda_transform_data, dust_event_info, "test_lda.pdf")
"""End of block"""



# Look at the coefficients for LDA and see which ones have large impact
coef_table = pd.DataFrame({"coef": lda.coef_.reshape(-1), "coef_mag": abs(lda.coef_.reshape(-1))}, 
                          index = data_lda.columns)
# coef_table.to_csv("lda_coefs.csv")

lda_coef = pd.read_csv("lda_coefs.csv")




# Prepare the full cleaned_May2023 dataset
may2023_data = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_May2023.csv")

may2023_data_integrated_values = read_JFJ_data(
    r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_reducedVersion_May2023.csv").iloc[:, -6:]

midDiameters = pd.read_csv(
    data_folder + r"\raw\Jungfraujoch\midpoint_diameters_size_distr_JFJ_2015_to_2021_May2023.csv"
    )

may2023_data = size_dist_handler(may2023_data, midDiameters.values.reshape(-1))

may2023_data_joined = may2023_data.join(may2023_data_integrated_values)
may2023_data_joined["neg_AE_SSA"] = (may2023_data_joined.AE_SSA < 0).astype(int)

# may2023_data_joined.to_csv("cleaned_May2023_dist_features.csv")


clipped_impute = clip_dataframe(impute_data, limit = 4)

# clipped_impute.to_csv("imputed_df_clipped_4std.csv")
