"""
Title: baselines.py

Description:
    Implementation of the baseline models:
        - SSA Exponent
        - Kernel SVC
        - Random Forest

Author: Andrew Zehr

"""

# Import helper packages and functions
import pandas as pd
import numpy as np
from utils.data_handling import read_JFJ_data
from utils.model_evaluation import evaluate_model, plot_predictions
from sklearn.model_selection import KFold
from sklearn.metrics import jaccard_score

# Import baseline predictors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Define SSA threshold estimator
def SSA_treshold_estimator(df: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing hourly JFJ data with "AE_SSA" column

    Returns
    -------
    pred_naive_SSA: pd.DataFrame
          Dataframe of predictions using SSA exponent threshold method  
          
    """
    
    SSA_values = df.AE_SSA
    
    pred_SSA = np.zeros_like(SSA_values)
    
    streak = 0
    current_streak_indices = []
    
    for i, sign in enumerate(SSA_values):
        if (sign < 0):
            streak += 1
            current_streak_indices.append(i)
        else:
            if (streak >= 6):
                pred_SSA[current_streak_indices] = 1
            streak = 0
            current_streak_indices = []

    return(pd.Series(pred_SSA.astype(int), index = df.index, name = "Neg. SSA"))


# Import the data and labels
data2017 = read_JFJ_data(r"\final\cleaned_impute.csv", 
                         date_range = ["2017-01-01 00:00:00","2017-12-31 23:59:59"])
data2020 = read_JFJ_data(r"\final\cleaned_impute.csv", 
                         date_range = ["2020-01-01 00:00:00","2020-12-31 23:59:59"])

# Labels
dust_events_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
dust_events_2020 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")


# Create folds for baseline model cross-validation
kf = KFold(n_splits=5, shuffle=False)


y_2020 = dust_events_2020.loc[data2020.index].sde_event
predictions = pd.Series(dtype=int, name = "RF")


"""Random Forest"""
# Perform cross-validation for random forest
scores = []
rf_best_predictions_2020 = None
best_score = 0
for depth in [1,2,3, 5, 7, None]:
    predictions = pd.Series(dtype=int)
    for i, fold in enumerate(kf.split(data2020)):
        np.random.seed(1)
        
        
        X_train = data2020.iloc[fold[0]]
        X_test = data2020.iloc[fold[1]]

        y_train = y_2020.loc[X_train.index]
        y_test = y_2020.loc[X_test.index]
        
        rf = RandomForestClassifier(class_weight="balanced", 
                                    n_estimators = 250, max_depth = depth)
        rf.fit(X_train, y_train)
        
        pred = rf.predict(X_test)
        
        predictions = pd.concat(
            [predictions, pd.Series(pred, index = X_test.index)], axis = 0)
    
    score = jaccard_score(y_2020, predictions)
    scores.append(score)
    
    if score > best_score:
        best_score = score
        rf_best_predictions_2020 = predictions.copy()
    
evaluate_model(rf_best_predictions_2020, dust_events_2020)
plot_predictions(pd.Series(rf_best_predictions_2020, name = "RF"), dust_events_2020)

# Now fit final RF model and predict on 2017
rf_final = RandomForestClassifier(class_weight="balanced", n_estimators = 500, max_depth = 7)
       
rf_final.fit(data2020, y_2020)

rf_preds_2017 = pd.Series(rf_final.predict(data2017), 
                          index = data2017.index, name="RF")

# Evaluate performance for 2017
evaluate_model(rf_preds_2017, dust_events_2017)
plot_predictions(rf_preds_2017, dust_events_2017)



""" SVC """
# Cross-validation for SVC
scores = []
svc_best_predictions_2020 = None
best_score = 0
for c in [1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2, 1e3, 1e4]:
    predictions = pd.Series(dtype=int, name = "SVC")
    for i, fold in enumerate(kf.split(data2020)):
        np.random.seed(1)
        
        X_train = data2020.iloc[fold[0]]
        X_test = data2020.iloc[fold[1]]

        y_train = y_2020.loc[X_train.index]
        y_test = y_2020.loc[X_test.index]
        
        svc = SVC(class_weight="balanced", C=c)
        
        svc.fit(X_train, y_train)
        
        pred = svc.predict(X_test)
        
        predictions = pd.concat(
            [predictions, pd.Series(pred, index = X_test.index)], axis = 0)
    
    score = jaccard_score(y_2020, predictions)
    scores.append(score)
    
    if score > best_score:
        best_score = score
        svc_best_predictions_2020 = predictions.copy()
    
svc_best_predictions_2020.name = "Logistic Regression"
evaluate_model(svc_best_predictions_2020, dust_events_2020)
plot_predictions(svc_best_predictions_2020, dust_events_2020)

# Now fit final RF model and predict on 2017
svc_final = SVC(class_weight = "balanced", C=1e2)
       
svc_final.fit(data2020, y_2020)

svc_preds_2017 = pd.Series(svc_final.predict(data2017), 
                          index = data2017.index, name="SVC")

# Evaluate performance on 2017 data
evaluate_model(svc_preds_2017, dust_events_2017)
plot_predictions(svc_preds_2017, dust_events_2017)



"""SSA Exponent"""
ssa_data_all = read_JFJ_data(r"\processed\Data_cleaned_Rob\aerosol_data_JFJ_2015_to_2021_CLEANED_reducedVersion_May2023.csv")
SSA_all_preds = SSA_treshold_estimator(ssa_data_all)

SSA_preds_2020 = SSA_all_preds["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
SSA_preds_2017 = SSA_all_preds["2017-01-01 00:00:00":"2017-12-31 23:59:59"]

evaluate_model(SSA_preds_2020, dust_events_2020)
plot_predictions(SSA_preds_2020, dust_events_2020, show_missing = False)

evaluate_model(SSA_preds_2017, dust_events_2017)
plot_predictions(SSA_preds_2017, dust_events_2017, show_missing = False)


# Save the model predictions
model_predictions_2020 = pd.DataFrame({"RF": rf_best_predictions_2020, "RBF SVC": svc_best_predictions_2020}, index = data2020.index)
model_predictions_2017 = pd.DataFrame({"RF": rf_preds_2017, "RBF SVC": svc_preds_2017}, index = data2017.index)

# Write the model predictions (this has already been done and is thus commented-out)
# model_predictions_2020.to_csv("predictions_2020.csv")
# model_predictions_2017.to_csv("predictions_2017.csv")
