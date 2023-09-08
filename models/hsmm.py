"""
Title: hsmm.py

Description:
    Implementation of the Hidden Semi Markov Model using categorical sojurn
    distribution. Uses the edhsmm package (https://github.com/poypoyan/edhsmm).
    
    This implementation allows self-transitions to the same state after the 
    sojurn waiting time.

Author: Andrew Zehr

"""


import pandas as pd
import matplotlib.pyplot as plt

from edhsmm.hsmm_base import GaussianHSMM
from utils.data_handling import read_JFJ_data
from utils.model_evaluation import evaluate_model, smooth_predictions, summarize_states, plot_predictions
from utils.hmm_storage import load_hmm_model, save_hmm_model



# Import Data
impute_data = read_JFJ_data(r"\processed\imputed_df_clipped_4std.csv")

# V_D40 is dropped (everywhere) because it contains anomalous data
trex10_features = read_JFJ_data(r"\processed\fdr0.01_features.csv").drop("V_D40", axis = 1).columns
data_10trex = impute_data.loc[:, trex10_features]

# Load ground truth labels
dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")

Y = dust_event_info.sde_event

X_10 = data_10trex.copy()



""" Partition the data """
X_10_train = X_10[:"2019-12-31 23:59:00"]
X_10_test = X_10["2020-01-01 00:00:00":]
X_10_2017 = X_10["2017-01-01 00:00:00":"2017-12-31 23:59:59"]


# Training with the D3 feature causes numerical errors and doesn't fit the model
testset = X_10_train.drop("D3", axis = 1).to_numpy() 

# Define model
R = GaussianHSMM(n_states = 8, n_durations = 100, n_iter = 100)


# EM algorithm
R.fit(testset)
model_8_state = load_hmm_model("hsmm_8state_100duration_10trex_noD3.pkl")
model_2_state = load_hmm_model("hsmm_2state_100duration_10trex_noD3.pkl")
# save_hmm_model(R, "hsmm_2state_100duration_10trex_noD3.pkl")

# Use either a saved model or the model fit locally
model = model_2_state


# Make predictions and interpret states
preds = pd.Series(model.predict(X_10_test.drop("D3", axis = 1))[0], index = X_10_test.index, name = "ED_HSMM")
hsmm_state_summary, hsmm_pred_bin, hsmm_event_states = summarize_states(preds, dust_event_info, bf_threshold = 1)
hsmm_pred_bin = preds.isin([5,6,4]).astype(int)
preds_smooth = smooth_predictions(hsmm_pred_bin).apply(lambda x: 1 if x >0.5 else 0)


# Evaluate for 2020
evaluate_model(hsmm_pred_bin, dust_event_info)
plot_predictions(pd.Series(hsmm_pred_bin, name = "HSMM"), dust_event_info)


# Make predictions for 2017 and evalluate performance
preds_2017 = pd.Series(model.predict(X_10_2017.drop("D3", axis = 1))[0], index = X_10_2017.index, name = "ED_HSMM")
preds_bin_2017 = preds_2017.isin([5,6,4]).astype(int)
preds_smooth_2017 = smooth_predictions(preds_bin_2017).apply(lambda x: 1 if x >0.5 else 0)



evaluate_model(preds_2017, dust_event_info_2017)
plot_predictions(pd.Series(preds_2017, name = "HSMM"),
                 dust_event_info_2017, start = "2017-01-01 00:00:00",
                 title = "HSMM 2017")




#save_hmm_model(R, "hsmm_8state_100duration_10trex_noD3.pkl")


# Plot the fit sojurn duration distributions here
plt.plot(model.dur[0,:], label = "State 0")
plt.plot(model.dur[1,:], label = "State 1")
#plt.plot(model.dur[2,:], label = "State 2")
#plt.plot(model.dur[3,:], label = "State 3")
plt.xlabel("Duration")
plt.ylabel("Probability")
plt.title("DUration Distribution for HSMM (self-transitions allowed)")
plt.legend()