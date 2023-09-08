"""
Title: multi_state_hmm.py

Author: Andrew Zehr

Description: 
    Fit HMM with various number of states and see if these states correspond to
    dust events. Model selection using BIC
"""

import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn.vhmm import VariationalGaussianHMM
from scipy.stats import norm


import matplotlib.pyplot as plt
import seaborn as sns

from utils.data_handling import read_JFJ_data
from utils.model_evaluation import (
    evaluate_model, plot_predictions, smooth_predictions, summarize_states, plot_hmm_states
    )
from sklearn.metrics import confusion_matrix, auc
from utils.hmm_storage import load_hmm_model, save_hmm_model


def find_best_hmm(fit_data, min_states = 2, max_states = 10, fit_iter = 100, num_inits = 5, verbose = False, cov_type = "full"):
    """
    Finds the HMM with the best likelihood in a certain range of a number of 
    hidden states. For each model size, it initializes the model a certain number
    of times and selects the best model from these initialization

    Parameters
    ----------
    fit_data : pd.DataFrame
        Dataframe to fit the model on.
    min_states : int, optional
        Minumum number of states to fit. The default is 2.
    max_states : int, optional
        Maximum number of states to fit. The default is 10.
    fit_iter : int, optional
        Number of iterations allowed for each model fit. The default is 100.
    num_inits : int, optional
        Number of times to fit a model with a given size. The default is 5.
    verbose : bool, optional
        Whether to print model training progress. The default is False.
    cov_type : string, optional
        Type of covariance matrices to fit in model. The default is "full".

    Returns
    -------
    dict
        Dictionary containing best scores for each number of states, the best model,
        the best initialization seed for this model, and the number of states
        for the best model
    """
    
    best_score = 1e10
    best_scores_component = []
    best_comp = 0
    best_model = None
    best_seed = 0
    for n in range(min_states, max_states + 1):
        best_comp_score = 1e10
        for i in range(num_inits):
            i = i
            mod = GaussianHMM(n,
                              n_iter=fit_iter,
                              covariance_type=cov_type,
                              random_state=i,
                              verbose=verbose)
            
            mod.fit(fit_data)
            crit = mod.bic(fit_data)
            
            print(f"Training HMM({n}) BIC={crit} "
                  f"Iterations={len(mod.monitor_.history)} ")
            
            if crit < best_comp_score:
                best_comp_score = crit
            
            if crit < best_score:
                best_score = crit
                best_model = mod
                best_seed = i
                best_comp = n
                
            
        
        best_scores_component.append(best_comp_score)
        
    return {"best_scores": best_scores_component, "best_model": best_model, "best_seed": best_seed, "states": best_comp}




def roc_and_best_threshold(state_summary: pd.DataFrame, pred: pd.Series, dust_event_info: pd.DataFrame):
    '''
    Plots an ROC for multi-state models. It also calculates the area under the curve
    and returns the point of the ROC curve closest to the ideal point (top left corner).
    
    Note: the "optimal" point returned by this function is optimal in a very 
    specific sense of the word and will not necessarily match with the desired
    best model in the problem setting. 

    Parameters
    ----------
    state_summary : pd.DataFrame
        Dataframe summarizing states, this dataframe is the output of 
        the "summarize_states" function above.
    pred : pd.Series
        The state predictions of the multi-state model.
    dust_event_info : pd.DataFrame
        Data on the occurences of SDEs

    Returns
    -------
    float
        Returns the ideal Bayes Factor threshold .

    '''
    xAxis = [0] # True Positive Rate
    yAxis = [0] # False Positive Rate
    Y = dust_event_info.loc[pred.index].sde_event

    for bf in state_summary.bayes_factor:
        
        event_state_idx = state_summary.index[state_summary.bayes_factor >= bf].values
        binary_pred = np.isin(pred, event_state_idx).astype(int)
        conf_matrix = confusion_matrix(Y, binary_pred)
        
        
        FP = conf_matrix[0,1]  
        FN = conf_matrix[1,0]
        TP = conf_matrix[1,1]
        TN = conf_matrix[0,0]
        
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)

        
        xAxis.append(FPR)
        yAxis.append(TPR)


    sns.set_style("whitegrid")
    f, ax = plt.subplots()
    sns.lineplot(x = xAxis, y = yAxis)
    ax.plot([0, 1], [0, 1], color ="gray", linestyle = "dashed")
    plt.title(f"ROC: (AUC = {auc(xAxis, yAxis):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    
    
    state_summary["fpr"] = xAxis[1:]
    state_summary["tpr"] = yAxis[1:]
    
    state_summary["dist_to_optimal"] = np.sqrt((1 - state_summary["tpr"])**2 + (state_summary["fpr"])**2)
    
    return state_summary.loc[state_summary["dist_to_optimal"].idxmin(axis = 0)].bayes_factor
    
    


# Import Data
imputed_data = read_JFJ_data(r"\final\cleaned_impute.csv")
pca_data = read_JFJ_data(r"\final\data_pca.csv")
trex_data = read_JFJ_data(r"\final\data_10trex_features.csv")
ae_data = read_JFJ_data(r"\final\tcn_ae_data.csv")


dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
Y = dust_event_info.sde_event

# Import model prediction dataframes to save results
model_results_2020 = pd.read_csv("predictions_2020.csv", index_col=0)
model_results_2017 = pd.read_csv("predictions_2017.csv", index_col=0)


# Select which data to actually use - data_to_use.copy()
data = trex_data.copy()

data2020 = data["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
data2017 = data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]


# Using 8-states, find best initialization from 10
best = 0
scores = []
for j in range(10):
    hmm = GaussianHMM(n_components = 8, covariance_type = "full", random_state= j, n_iter = 20)
    hmm_fit = hmm.fit(data)
    score = hmm_fit.score(data)
    scores.append(score)
        
best_seed = np.argmax(scores)

hmm = GaussianHMM(n_components=10, covariance_type = "full", n_iter=200, verbose = True, random_state=best_seed)
hmm.fit(data)

pred = pd.Series(hmm.predict(data2020), index = data2020.index)

state_summary8, bin_preds8, event_states8 = summarize_states(pred, dust_event_info, bf_threshold = 1.16)
bin_preds8 = pred.isin([3,4,6]).astype(int)

print(f"Best Bayes Factor threshold is: {roc_and_best_threshold(state_summary8, pred, dust_event_info)}")

plot_hmm_states(data2020, hmm, pred, event_states8)


sm_pred8 = pd.Series(smooth_predictions(bin_preds8).apply(lambda x: 1 if x > 0.5 else 0).values, 
                    index = data2020.index)

evaluate_model(bin_preds8, dust_event_info)
plot_predictions(pd.Series(bin_preds8, index = data2020.index, name = "8-State HMM"), dust_event_info)

model_results_2020["8s_HMM_trex"] = bin_preds8


# Now, look at results for 2017
pred_2017 = pd.Series(hmm.predict(data2017), index = data2017.index)
bin_pred_2017 = pred_2017.isin([3,4]).astype(int)

evaluate_model(bin_pred_2017, dust_event_info_2017)
plot_predictions(pd.DataFrame(bin_pred_2017, index = data2017.index), 
                 dust_event_info_2017, start = "2017-01-01 00:00:00", dpi = 120)

model_results_2017["8s_HMM_trex"] = bin_pred_2017


# Plot the 8-state means and standard deviations
means_df = pd.DataFrame(hmm.means_, columns = data.columns, index = list(range(8)))
variance_df = pd.DataFrame(hmm.covars_.diagonal(axis1=1, axis2=2), columns = data.columns, index = list(range(8))).drop(4, axis = 0)

# Plot the normal distributions for each feature
num_states, num_features = means_df.shape

# Create a color map for each state
colors = plt.cm.tab20(np.linspace(0, 1, num_states))

# Iterate through features and plot normal distributions
for i in range(num_features):
    feature_name = means_df.columns[i]
    plt.figure(figsize=(8, 5))
    plt.title(f"Normal Distribution for {feature_name}")
    
    for j, state in enumerate(means_df.index):
        mean = means_df.iloc[j, i]
        variance = variance_df.iloc[j, i]
        std_dev = np.sqrt(variance)
        x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        density = norm.pdf(x, mean, std_dev)
        
        plt.plot(x, density, label=f"State {state}", color=colors[j])
    
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()



# Import Data
impute_data = read_JFJ_data(r"\processed\iterative_impute_df.csv")
# impute_data = read_JFJ_data(r"\processed\imputed_df_clipped_4std.csv")

trex25_features = read_JFJ_data(r"\processed\fdr0.8_features.csv").drop("V_D40", axis =1).columns
trex10_features = read_JFJ_data(r"\processed\fdr0.01_features.csv").drop("V_D40", axis =1).columns

data_25trex = impute_data.loc[:, trex25_features]
data_10trex = impute_data.loc[:, trex10_features]


dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
Y = dust_event_info.sde_event


"""Can select from either of these two data sets when training """
X_25 = data_25trex.copy()
X_25_train = X_25[:"2019-12-31 23:59:00"]
X_25_test = X_25["2020-01-01 00:00:00":]

X_10 = data_10trex.copy()
X_10_train = X_10[:"2019-12-31 23:59:00"]
X_10_test = X_10["2020-01-01 00:00:00":]

X_10_2017 = X_10["2017-01-01 00:00:00":"2017-12-31 23:59:59"]
X_25_2017 = X_25["2017-01-01 00:00:00":"2017-12-31 23:59:59"]


# Do teh BIC selection for the number of states
results = find_best_hmm(X_10_train, min_states=2, max_states=35, fit_iter = 20, verbose=True, num_inits=1)

sns.set_style("whitegrid")
sns.lineplot(x = list(range(2,31)), y = results["best_scores"])
plt.title("HMM BIC")
plt.xlabel("Number of States")
plt.ylabel("BIC")
plt.xticks(range(2, 31, 3))




