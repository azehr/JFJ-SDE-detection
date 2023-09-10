"""
Investigate missing values

Causes of missingness:
    1: Missing measurements
       a. CPC-3772 - N_N11   [1]
       b. Nephelometer - scattering/backscattering   [2]
       c. Aethalometer - absorption   [3]
       d. SMPS - (d < 500 nm)    [4]
       e. FIDAS - (d > 500 nm)   [5]
       f. Derived values [6]
    
    2: Dropped outliers (whole row set to zero in cleaned data file, only 73 instanced of this or ~)
       a. Integrated volume or size distribution (Ntot > 20000 Vtot > 1000)
       b. Extreme SSA < -2 or >3
       c. absorption coefficients < -1 Mm-1
       
       All outlier reasons correspond to [7] in cause status
"""

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer



import pandas as pd
import numpy as np
from utils.data_handling import read_JFJ_data, size_dist_handler
from statsmodels.stats.proportion import proportions_ztest
import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def assign_missing_cause(row, reduced = True):
    cause = []
    
    if (np.isnan(row).sum() == 0):
        cause.append(0) # If there is no missing data
    
    if np.isnan(row["N_N11"]):
        cause.append(1) # CPC measurements missing
        
    if (np.isnan(row[["BsB_S12", "BsG_S12", "BsR_S12", "BbsB_S12",
                      "BbsG_S12", "BbsR_S12"]]).sum() > 0):
        cause.append(2) # Nephelometer TSI
        
    if (np.isnan(row[["BaCorr2_A13", "BaCorr3_A13", "BaCorr4_A13",
                     "BaCorr5_A13", "BaCorr6_A13", "BaCorr7_A13", "BaCorr1_A13"]]).sum() > 0):
        cause.append(3) # Aetholometer AE33
    
    
    if reduced:
        # Take sample of the SMPS columns
        if (np.isnan(row[["D1", "V_D1"]]).sum() > 0):
            cause.append(4) # SPMS
            
        # Take sample of the FIDAS columns
        if (np.isnan(row[["D38", "V_D38"]]).sum() > 0):
            cause.append(5) # FIDAS
    if reduced == False:
        # Take sample of the SMPS columns
        if (np.isnan(row[["D0_016849", "V_D0_016849"]]).sum() > 0):
            cause.append(4) # SPMS
            
        # Take sample of the FIDAS columns
        if (np.isnan(row[["D19_822390", "V_D19_822390"]]).sum() > 0):
            cause.append(5) # FIDAS
        
    if (cause == []):
        cause = [6] # Derived
    
    return cause


rawData = pd.read_csv(r"data\raw\Jungfraujoch\aerosol_data_JFJ_2015_to_2021.csv",
                      parse_dates=["DateTimeUTC"], index_col=0)["2017-01-01 00:00:00":]

models = pd.read_csv("model_predictions.csv", parse_dates=["DateTimeUTC"], index_col=0)
dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")

no_outlier_data = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_May2023.csv")["2017-01-01 00:00:00":]



completeDates = no_outlier_data[no_outlier_data.index.isin(models.index)]
missingDates = no_outlier_data[~no_outlier_data.index.isin(models.index)]


print(f"{100 * completeDates.sde_event.mean():.2f}% of non-missing hours are a dust storm")
print(f"{100 * missingDates.sde_event.mean():.2f}% of missing hours are a dust storm")

proportions_ztest([int(sum(missingDates.sde_event)), int(sum(completeDates.sde_event))], [len(missingDates), len(completeDates)], alternative="larger")

missing_feature = missingDates.isna().sum()


# There are 64 extra events dropped by Rob for having too high of a threshold


no_outlier_data["missing_status"] = no_outlier_data.apply(lambda row: assign_missing_cause(row), axis = 1).astype(str)
rawData["missing_status"] = rawData.apply(lambda row: assign_missing_cause(row, reduced=False), axis = 1).astype(str)



outlier_index = rawData.dropna().drop(no_outlier_data.dropna().index).index

rawData.loc[outlier_index, "missing_status"] = "[7]"
no_outlier_data.loc[outlier_index, "missing_status"] = "[7]"

to_drop = ["[1, 2, 3, 4, 5]", "[7]", "[1, 2, 4, 5]", "[2, 3, 4, 5]", "[3, 4, 5]", "[2, 4, 5]"] 


df_plot = no_outlier_data[["missing_status", "sde_event"]]

"""
{"[0]":"Not Missing", 
                                               "[1, 2, 3, 4]": "CPC, TSI, AE33, SPMS",  
                                               "[6]": 'Derived',
                                               "[3]": 'AE33',                      
                                               "[1, 2, 3]": "CPC, TSI, AE33",
                                               "[2]": "TSI",
                                               "[4]": "SPMS",
                                               "[1]": "CPC",
                                               "[7]": "Outlier",
                                               "[2, 3]": "Other",
                                               "[1, 2, 4]": "Other",
                                               "[1, 2]": "Other"}
"""


cmap = colors.LinearSegmentedColormap.from_list('Custom', 
                                                ["#f0eeeb", 
                                                 "#eb092b",
                                                 "#000000",
                                                 "blue",
                                                 "brown",
                                                 "green",
                                                 "yellow",
                                                 "purple",
                                                 "orange",
                                                 "#9bddb1",
                                                 "#d48c84"], 11)


dates = [s[:10] for s in df_plot.index.strftime("%m.%d")]
date_locs = list(range(len(dates)))

fig = plt.figure(figsize=(10,5), dpi = 100)
ax = sns.heatmap(np.transpose(df_plot), cmap=cmap)
colorbar = ax.collections[0].colorbar

colorbar.set_ticks([0.45, 1.35, 2.25, 3.15, 4.05, 4.95, 5.85, 6.75, 7.65, 8.55, 9.45])

colorbar.set_ticklabels(["Not Missing",
                         "SDE",
                         "CPC, TSI, AE33, SPMS",
                         'Derived',
                         'AE33',
                         "CPC, TSI, AE33",
                         "TSI",
                         "SPMS",
                         "CPC",
                         "Outlier",
                         "Other",
                         ])

plt.xticks(date_locs[::672], dates[::672])
plt.xticks(rotation=70)
plt.xlabel("")
plt.tight_layout()







train = no_outlier_data[:"2020-07-01 00:00:00"].dropna()
test = no_outlier_data["2020-07-01 00:00:00":].dropna()

X_train = train.drop("N_N11", axis = 1)
X_test = test.drop("N_N11",  axis = 1)

y_train = train["N_N11"]
y_test = test["N_N11"]


ridge = RidgeCV()
ridge = ridge.fit(X_train, y_train)


ridge_pred = ridge.predict(X_test)
print(f"Mean Absolute Error for ridge regression is {mean_absolute_error(y_test, ridge_pred)}")


lasso = LassoCV(max_iter=10000)
lasso = lasso.fit(X_train, y_train)


lasso_pred = lasso.predict(X_test)
print(f"Mean Absolute Error for lasso regression is {mean_absolute_error(y_test, lasso_pred)}")


rf = RandomForestRegressor(n_estimators = 500, max_depth=10)
rf = rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print(f"Mean Absolute Error for Random Forest Regression is {mean_absolute_error(y_test, rf_pred)}")



gb = GradientBoostingRegressor(n_estimators=100)
gb = gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)
print(f"Mean Absolute Error for Gradient Boosting Regression is {mean_absolute_error(y_test, gb_pred)}")



# Try using iterative imputation on the dataset, however don't do this on areas that are missing multiple instruments measurements
to_drop = ["[1, 2, 3, 4, 5]", "[7]", "[1, 2, 4, 5]", "[2, 3, 4, 5]", "[3, 4, 5]", "[2, 4, 5]"] 

df_to_impute = no_outlier_data[~no_outlier_data.missing_status.isin(to_drop)]
df_to_impute.replace({np.inf: np.nan, -np.inf: np.nan}, inplace=True)

imputer = IterativeImputer(max_iter=15)
imputed = imputer.fit_transform(df_to_impute.drop("missing_status", axis = 1))

df_imputed = pd.DataFrame(imputed, columns = df_to_impute.drop("missing_status", axis = 1).columns, index = df_to_impute.index)
df_imputed["missing_status"] = df_to_impute.missing_status
df_imputed["missing_status"] = [0 if x =="[0]" else 1 for x in df_imputed.missing_status]
df_imputed = df_imputed[df_imputed.N_N11 > 0]




diams = pd.read_csv(r"data/raw/Jungfraujoch/midpoint_diameters_size_distr_JFJ_2015_to_2021_May2023.csv", header=None)

df_imputed_size_features = size_dist_handler(df_imputed, diams[:-1])
# df_imputed_size_features.to_csv("iterative_impute_df.csv")



# Check appropriateness of ridge regression
# Take a few sample features (N_N11, BsB_S12, BaCorr2_A13, D1, V28, AAE)  (one from each machine)
# and see how good the other machines features can impute it

data = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_May2023.csv")["2017-01-01 00:00:00":].dropna()



data_train = data[:"2020-01-01 00:00:00"]
data_test = data["2020-01-01 00:00:00":]

sc = StandardScaler()
sc.fit(data_train)

train = pd.DataFrame(sc.transform(data_train), columns = data_train.columns, index = data_train.index)
test = pd.DataFrame(sc.transform(data_test), columns = data_test.columns, index = data_test.index)

features_to_impute = ["N_N11", "BsB_S12", "BaCorr2_A13", "D1", "V_D28", "AAE"]

cpc_features = ["N_N11"]
aethalometer_features = ["BaCorr2_A13", "BaCorr3_A13", "BaCorr4_A13", "BaCorr5_A13",
"BaCorr6_A13", "BaCorr7_A13", "BaCorr1_A13"]
nephelometer_features = ["BsB_S12", "BsG_S12", "BsR_S12", "BbsB_S12", "BbsG_S12", "BbsR_S12"]
SPMS_features = no_outlier_data.iloc[:, np.r_[22:42, 62:82]].columns 
FIDAS_features = no_outlier_data.iloc[:, np.r_[42:62, 82:102]].columns
derived_features = no_outlier_data.iloc[:, 14:22].columns

all_feature_lists = [cpc_features, aethalometer_features, nephelometer_features, SPMS_features, FIDAS_features]



from matplotlib.backends.backend_pdf import PdfPages
with PdfPages("rf_imputation_actual_fitted.pdf") as pdf_pages:
    for i, feature in enumerate(features_to_impute):
            
        feature_lists_to_use = [feature_list for feature_list in all_feature_lists if feature not in feature_list]
        features_to_use = [feature for feature_list in feature_lists_to_use for feature in feature_list]
            
        y_train = train[feature]
        X_train = train.loc[:, features_to_use]
            
        y_test = test[feature]
        X_test = test.loc[:, features_to_use]
            
        #ridge = RidgeCV(alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3])
        #ridge_fit = ridge.fit(X_train, y_train)
        
        rf = RandomForestRegressor(n_estimators=200, max_depth = 3)
        rf_train = rf.fit(X_train, y_train)
        
        train_pred = rf_train.predict(X_train)
        test_pred = rf_train.predict(X_test)

        #train_pred = ridge_fit.predict(X_train)
        #test_pred = ridge_fit.predict(X_test)
            
        print(f"Results for: {feature}")
        print(f"Training MSE is {mean_squared_error(y_train, train_pred):.2f}. R2 score is {r2_score(y_train, train_pred):.2f}")
        print(f"Testing MSE is {mean_squared_error(y_test, test_pred):.2f}. R2 score is {r2_score(y_test, test_pred):.2f}")
            
        residuals = y_test - test_pred
            
        figu = plt.figure(i)
        ax = sns.scatterplot(x = y_test, y = test_pred)
        ax.set_title(f"Actual vs Fitted {feature} (R2 = {r2_score(y_test, test_pred):.1f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        #ax.axhline(0, linestyle='--', color = "black")
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle = "--", color = "black")
        pdf_pages.savefig(figu)


    
  
    