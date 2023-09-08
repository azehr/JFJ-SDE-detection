"""
Title: simple_hmm.py

Author: Andrew Zehr

Description: 
    Fit a simple HMM (only 2-states) on the data. The model can be fit supervised
    or unsupervised. 
    
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.vhmm import VariationalGaussianHMM


from utils.data_handling import read_JFJ_data
from utils.model_evaluation import ( 
    evaluate_model, 
    smooth_predictions, 
    plot_predictions
    )
from utils.calibration import plot_calibration_curve
from utils.hmm_storage import load_hmm_model, save_hmm_model


def fit_hmm(X_train: pd.DataFrame, dust_event_info: pd.DataFrame, algorithm = "viterbi", outlier_threshold = 3):
    """Fits a supervised 2-State HMM on the training data and set labels"""
    
    # Add check to see if the training data is in the right 
    if ((min(X_train.index) < min(dust_event_info.index)) | (max(X_train.index) > max(dust_event_info.index))):
        print("Error: The range of training data provided falls outside range of labeled data")
        return
    

    dust_train = dust_event_info.loc[X_train.index]
    
    # Estimate transition matrix using the training data
    total_hours = len(dust_train.sde_event) 
    total_SDE = len(dust_train.sde_event_nr.unique()) - 1
    total_SDE_hours = sum(dust_train.sde_event)
    total_non_SDE_hours = total_hours - total_SDE_hours
    
    prob_0_to_1 = total_SDE / total_non_SDE_hours
    prob_1_to_0 = total_SDE / total_SDE_hours
    
    
    # Transition Matrix and Initial Distribution
    transMat = np.array([[1-prob_0_to_1, prob_0_to_1],[prob_1_to_0,1-prob_1_to_0]])
    initialProb = np.array([total_non_SDE_hours / total_hours, total_SDE_hours / total_hours]) # Fixed Initial Prob
    
    
    # Now remove obvious outliers: (any value larger than 3 standard deviations away from 0)
    # This helps the stability of the std_deviation estimate
    std_devs = X_train.std()
    
    # Define a function to check if a row should be dropped
    def should_drop(row):
        for col in X_train.columns:
            if abs(row[col]) > (std_devs[col] * outlier_threshold): 
                return True
            return False

    # Use apply() and a lambda function to identify rows to drop
    rows_to_drop = X_train.apply(lambda row: should_drop(row), axis=1)

    X_train_cleaned = X_train[~rows_to_drop]

    Y_train = dust_train.sde_event[X_train_cleaned.index]

    
    # Now seperate the training data into instances with dust-storms and those without
    merged_df = pd.merge(X_train_cleaned, Y_train, how = "left", 
                     left_index=True, right_index=True)


    # Boolean index to select only rows where bool_col is True in B
    bool_index = merged_df['sde_event'] == 1

    # Select only rows where bool_col is True in B and keep only columns from A
    X_train_SDE = merged_df[bool_index].drop(columns="sde_event")

    X_train_no_SDE = merged_df[~bool_index].drop(columns="sde_event")


    SDE_means = X_train_SDE.mean()
    no_SDE_means = X_train_no_SDE.mean()

    SDE_cov = np.cov(X_train_SDE, rowvar = False)
    no_SDE_cov = np.cov(X_train_no_SDE, rowvar = False)
    
    
    hmm = GaussianHMM(n_components=2, covariance_type = "full", algorithm = algorithm)
    hmm.transmat_ = transMat
    hmm.startprob_ = initialProb
    hmm.means_ = np.array([no_SDE_means, SDE_means])
    hmm.covars_ = np.array([no_SDE_cov, SDE_cov])
    
    return hmm




# Import Data
imputed_data = read_JFJ_data(r"\final\cleaned_impute.csv")
pca_data = read_JFJ_data(r"\final\data_pca.csv")
trex_data = read_JFJ_data(r"\final\data_10trex_features.csv")
ae_data = read_JFJ_data(r"\final\tcn_ae_data.csv")

dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
Y = dust_event_info.sde_event

# Import model prediction dataframes to save results (this is only necessary if you
# want to collect all results in one place and create joint graphs)
model_results_2020 = pd.read_csv("predictions_2020.csv", index_col=0)
model_results_2017 = pd.read_csv("predictions_2017.csv", index_col=0)


# Select which data to actually use - trex_data.copy(), pca_data.copy(), ae_data.copy(), etc...
data = trex_data.copy()

data2020 = data["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
data2017 = data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]


"""
Here I am "optimizing" for the seed. While this seems like cheating, the 
Baum-Welch algorithm can get stuck in local minimums and multiple 
initializations help it reach a global optimum. The selection of which seed is 
the best is done without reference to the labels. So this is
equivalent to selecting the best out of multiple initializations.
"""

best = 0
scores = []
for j in range(10):
    hmm = GaussianHMM(n_components = 2, covariance_type = "full", random_state= j, n_iter = 100)
    hmm_fit = hmm.fit(data)
    score = hmm_fit.score(data)
    scores.append(score)
        
best_seed = np.argmax(scores)



# Now train with all data
hmm_all = GaussianHMM(n_components=2, covariance_type = "full", random_state = best_seed, verbose = True, n_iter = 100, algorithm="viterbi")
hmm_all.fit(data) # Fully converged

probs = pd.Series(hmm_all.predict_proba(data2020)[:,0], index = data2020.index, name = "probs")

plot_calibration_curve(Y[probs.index], probs)

# Save the model if needed
# save_hmm_model(hmm_all, "best_2state_ae_hmm.pkl")

""" Can load the model here, without having to fit it first"""
# hmm_all = load_hmm_model("best_2state_pca_hmm.pkl")


hmm_all_pred = pd.Series(hmm_all.predict(data2020), index = data2020.index, name = "HMM")

if (hmm_all_pred.mean() > 0.4):
    hmm_all_pred = (hmm_all_pred * -1) + 1


# Do the smoothing to see if it helps
hmm_all_pred_smooth = smooth_predictions(hmm_all_pred).apply(lambda x: 1 if x >0.5 else 0)


evaluate_model(hmm_all_pred, dust_event_info)
plot_predictions(hmm_all_pred, dust_event_info, show_missing = True, figsize = (10,5))

## Save Results
model_results_2020["2s_HMM_pca"] = hmm_all_pred



# See if it works on 2017 data
pred_2017 = pd.Series(hmm_all.predict(data2017), index = data2017.index, name = "Simple HMM")

if (pred_2017.mean() > 0.4):
    pred_2017 = (pred_2017 * -1) + 1
pred_2017_smooth = smooth_predictions(pred_2017).apply(lambda x: 1 if x >0.5 else 0)


evaluate_model(pred_2017, dust_event_info_2017)
plot_predictions(pred_2017, dust_event_info_2017, show_missing = True, figsize = (10,5), start = "2017-01-01 00:00:00")


model_results_2017["2s_HMM_pca"] = pred_2017




""" Try a variational HMM with all variables"""
full_data_2020 = imputed_data["2020-01-01 00:00:00":]

vi = VariationalGaussianHMM(n_components=2, n_iter=100, algorithm = "viterbi", verbose = True)
vi.fit(imputed_data)
vi_pred = pd.Series(vi.predict(full_data_2020), index = full_data_2020.index, name = "Variational HMM")
vi_pred_sm = smooth_predictions(vi_pred).apply(lambda x: 1 if x >0.5 else 0)


evaluate_model(vi_pred, dust_event_info)
plot_predictions(vi_pred_sm, dust_event_info, figsize = (10,6))



""" Fit a supervised 2-State HMM on the 2020 labels and test on 2017 data """
supervisedHMM = fit_hmm(data2020, dust_event_info, outlier_threshold = 4)

superivsedHMM_preds_2020 = pd.Series(supervisedHMM.predict(data2020), 
                                     index = data2020.index, 
                                     name = "Supervised HMM")

superivsedHMM_preds_2017 = pd.Series(supervisedHMM.predict(data2017), 
                                     index = data2017.index, 
                                     name = "Supervised HMM")

# Evaluate on 2020
evaluate_model(superivsedHMM_preds_2020, dust_event_info)
plot_predictions(superivsedHMM_preds_2020, dust_event_info, show_missing = True, figsize = (10,5))

# Evaluate on 2017
evaluate_model(superivsedHMM_preds_2017, dust_event_info_2017)
plot_predictions(superivsedHMM_preds_2017, dust_event_info_2017, show_missing = True, figsize = (10,5), start = "2017-01-01 00:00:00")

#model_results_2017["Supervised HMM"] = superivsedHMM_preds_2017
#model_results_2020["Supervised HMM"] = superivsedHMM_preds_2020


