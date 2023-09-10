"""
Test the T-Rex data sets to see performance on CV-Logistic Regression
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from utils.model_evaluation import evaluate_model, visualize_predictions
from utils.data_handling import read_JFJ_data

trex_features = pd.read_csv(r"trex_selected_features.csv", index_col = 0)

data = read_JFJ_data(r"\processed\cleaned_May2023_dist_features.csv")

data2020 = data.dropna()["2020-01-01 00:00:00":]
dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")

lda_coef = pd.read_csv("lda_coefs.csv", index_col=0)
top11_features = lda_coef.sort_values("coef_mag", ascending =False).iloc[:11, :]
top_lda_feature_names = top11_features.index


l1_coef = pd.read_csv("logistic_regression_L1_coef.csv", index_col=0)
top11_features_log = l1_coef.sort_values("coef_magnitude", ascending =False).iloc[:11, :]
top_log_feature_names = top11_features_log.index


def calculate_trex_accuracy(trex_features, dust_event_info, df, fdr = "All"):
   
    split_date = "2020-06-30 23:59:59"
    
    
    row = trex_features.loc[str(fdr)]
    
    
    
    X = df.loc[:, row[row == 1].index]
    
    # X = df.loc[:, top_log_feature_names]
    
    print(f"Number of features: {X.shape[1]}")
    
    y = dust_event_info.sde_event
    
    X_train = X[:split_date]
    X_test = X[split_date:]
    
    y_train = y[X_train.index][:split_date]
    y_test = y[X_test.index][split_date:]
    
    clasif = RandomForestClassifier(n_estimators = 2000,
                                    max_depth = 5, class_weight = "balanced")
    clasif.fit(X_train, y_train)
    
    pred = clasif.predict(X_test)

    
    evaluate_model(dust_event_info, pd.Series(pred, index = X_test.index))
    # visualize_predictions(y_test, pred)



# Get the baseline performance (with all features)
calculate_trex_accuracy(trex_features, dust_event_info, data2020, fdr = "All")


# Now look at the Trex selection for an FDR threshold of 0.05
calculate_trex_accuracy(trex_features, dust_event_info, data2020, fdr = "0.05")

# Now look at the Trex selection for an FDR threshold of 0.8
calculate_trex_accuracy(trex_features, dust_event_info, data2020, fdr = "0.8")


# Now look at the Trex selection for an FDR threshold of 0.01 (only 11 features)
calculate_trex_accuracy(trex_features, dust_event_info, data2020, fdr = "0.01")



# Create a dataframe with just the data from the lowest bounds
row = trex_features.loc["0.8"]


X = data.loc[:, row[row == 1].index]

X = X.replace([-np.inf, np.inf], np.nan)
X = X.dropna()

X = X.drop(["V_D40"], axis = 1)

qt = QuantileTransformer(output_distribution="normal")
X_trans = pd.DataFrame(qt.fit_transform(X), index = X.index, columns = X.columns)

    
# X.to_csv("fdr0.01_features.csv")

