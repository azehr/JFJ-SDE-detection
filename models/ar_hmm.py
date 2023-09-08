"""
Title: ar_hmm.py

Description:
    Trains and evaluates AR-HMM models. Performs BIC selection for both the number of states 
    and the order of the autoregressive component. The base AR_HMM code (hmm_history)
    was provided by Dr. Christian Donner of the Swiss Data Science Center
    (https://datascience.ch/team_member/christian-donner/)

Author: Andrew Zehr

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.hmm_history import GaussianARHMM

from utils.data_handling import read_JFJ_data
from utils.model_evaluation import (
    evaluate_model, 
    smooth_predictions,
    summarize_states,
    plot_predictions,
    )
from utils.hmm_storage import load_hmm_model, save_hmm_model
import seaborn as sns

def fit_hmm(X, K, H, seed=1, convergence = 1e-4, min_iter=10):
    """ Fit the specified AR-HMM"""
    hmm = GaussianARHMM(X.to_numpy(), K=K, H=H, noise=1e-8, seed=seed)
    hmm.run_em(conv=convergence, min_iter=10)
    return hmm


def hmm_predict(hmm, X, H = 1):
    """ Compute Marginal Probabilities for Latent space """
   
    X_t = torch.tensor(X.to_numpy())
    
    
    if H > 1:
       X_t = torch.cat((X_t[:1].repeat(H-1, 1), X_t), dim=0)

    
    logPx = hmm.get_data_log_likelihoods(X_t)
    fwd_msg, log_pXt = hmm.forward_pass(logPx)
    bwd_msg = hmm.backward_pass(logPx)
    qZ, qZZ = hmm.compute_marginals(fwd_msg, bwd_msg, logPx)
    result = pd.DataFrame(qZ.numpy(), columns = range(0, qZ.shape[1]), index = X.index)


    result["map"] = result.apply(lambda row: row.idxmax(), axis=1)
    result["viterbi"] = viterbi_algorithm(fwd_msg, bwd_msg)
    


    if qZ.shape[1] == 2:
        if (result["map"].mean() >= 0.5):
            result["map"].replace({0:1, 1:0}, inplace = True)
            result.rename(columns = {0:1, 1:0}, inplace = True)
    
        
        if (result["viterbi"].mean() >= 0.5):
            result["viterbi"].replace({0:1, 1:0}, inplace = True)
    
    return result


def viterbi_algorithm(forward_messages, backward_messages):
    """ Implements the Viterbi algorithm """
    T, K = forward_messages.shape
    
    forward_messages = forward_messages.numpy()
    backward_messages = backward_messages.numpy()

    # Step 1: Compute the joint probabilities
    joint_probabilities = forward_messages * backward_messages

    # Step 2: Find the most likely state at each time step
    most_likely_states = np.argmax(joint_probabilities, axis=1)

    # Step 3: Backtrack to find the overall most likely sequence
    most_likely_sequence = [most_likely_states[-1]]
    for t in range(T - 2, -1, -1):
        most_likely_states = np.argmax(forward_messages[t] * backward_messages[t + 1] * joint_probabilities[t], axis=0)
        most_likely_sequence.insert(0, most_likely_states)

    return most_likely_sequence


# Load data
imputed_data = read_JFJ_data(r"\final\cleaned_impute.csv")
pca_data = read_JFJ_data(r"\final\data_pca.csv")
trex_data = read_JFJ_data(r"\final\data_10trex_features.csv")
ae_data = read_JFJ_data(r"\final\tcn_ae_data.csv")

# Load ground truth labels
dust_event_info = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")
Y = dust_event_info.sde_event

"""
# Visualize autocorrelation within largest dust event ["2020-03-25 21:00:00":"2020-03-31 18:00:00"]

from statsmodels.graphics.tsaplots import plot_acf
acf_data = imputed_data["2020-03-25 21:00:00":"2020-03-31 18:00:00"]

acf_data_N_N11 = acf_data["N_N11"]
fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(acf_data_N_N11, ax = ax, title = "")
plt.grid()
plt.title("ACF of N_N11 during SDE")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.setp(ax.spines.values(), linewidth=0.3)

"""

# Import model prediction dataframes to save results
model_results_2020 = pd.read_csv("predictions_2020.csv", index_col=0)
model_results_2017 = pd.read_csv("predictions_2017.csv", index_col=0)


# Select which data to actually use
data_trex2020 = trex_data["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
data_trex2017 = trex_data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]

data_pca2020 = pca_data["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
data_pca2017 = pca_data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]

data_ae2020 = ae_data["2020-01-01 00:00:00":"2020-12-31 23:59:59"]
data_ae2017 = ae_data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]


""" Select number of hidden states and autoregressive order """
# Implement the BIC selection for states
max_components = 20
H_bic = 1
bic = []
components = []

for k in range(41,42):
    arhmm_bic = fit_hmm(trex_data, K=k, H=H_bic, convergence=2e-4, seed = 2023)
    bic_mod = arhmm_bic.compute_BIC(torch.tensor(trex_data.to_numpy())).item()
    bic.append(bic_mod)
    components.append(k)
    print(f"Done fitting model with {k} components!")



sns.set_theme()
sns.lineplot(x = components, y = bic)
plt.title("BIC by number of states AR-HMM (L = 1)")
plt.xlabel("Number of States K")
plt.ylabel("BIC")


# Implement the BIC selection for lag length
max_lag = 10
K_bic = 8
bic_lag = []
lag = []

for h in range(6,max_lag + 1):
    arhmm_bic = fit_hmm(trex_data, K=K_bic, H=h, convergence=1e-3, seed = 2023)
    bic_mod = arhmm_bic.compute_BIC(torch.tensor(trex_data.to_numpy())).item()
    bic_lag.append(bic_mod)
    lag.append(h)
    print(f"Done fitting model with AR({h}) observation model!")



sns.set_style("whitegrid")
sns.lineplot(x = lag, y = bic_lag)
plt.title("BIC by order of AR(p) process in AR-HMM")
plt.xlabel("p: Order of AR(p)")
plt.ylabel("BIC")





""" 1.  2-State, AR(1) HMM, 10 trex features"""
K1 = 2
H1 = 1

arhmm_1 = GaussianARHMM(trex_data.to_numpy(), K=K1, H=H1, noise=1e-8, seed=2023)
arhmm_1 = fit_hmm(trex_data, K=K1, H=H1, convergence=1e-4)
arhmm_1_preds = hmm_predict(arhmm_1, data_trex2020)
arhmm_1_preds_smooth = smooth_predictions(arhmm_1_preds.viterbi).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_1_preds_smooth, dust_event_info)
plot_predictions(arhmm_1_preds_smooth, dust_event_info)

# save_hmm_model(arhmm_1, "ar1_hmm_10trex.pkl")

saved_hmm = load_hmm_model("ar1_hmm_10trex.pkl")
saved_preds = hmm_predict(saved_hmm, data_trex2020)
saved_preds_smooth = smooth_predictions(saved_preds.viterbi).apply(lambda x: 1 if x > 0.5 else 0)
evaluate_model(saved_preds.viterbi, dust_event_info)
plot_predictions(saved_preds.viterbi, dust_event_info, figsize = (8,6))

model_results_2020["2s_arhmm_trex"] = saved_preds.viterbi

# Do it for 2017 too
arhmm_2s_2017 = hmm_predict(saved_hmm, data_trex2017).viterbi
evaluate_model(arhmm_2s_2017, dust_event_info_2017)
plot_predictions(arhmm_2s_2017, dust_event_info_2017, 
                 start = "2017-01-01 00:00:00")
model_results_2017["2s_arhmm_trex"] = arhmm_2s_2017



""" 2.  8-State, AR(1) HMM, 10 trex features"""
K2 = 8
H2 = 1

arhmm_2 = GaussianARHMM(trex_data.to_numpy(), K=K2, H=H2, noise=1e-8, seed=2023)
arhmm_2 = fit_hmm(trex_data, K=K2, H=H2, convergence=1e-4)

arhmm_2 = load_hmm_model("8state_ar1_hmm_10trex.pkl")

arhmm_2_preds = hmm_predict(arhmm_2, data_trex2020)


arhmm_2_state_summary, arhmm_2_pred_bin, arhmm_2_event_states = summarize_states(arhmm_2_preds.viterbi, dust_event_info, bf_threshold = 1)
arhmm_2_pred_bin.name = "8-State ARHMM"
arhmm_2_preds_smooth = smooth_predictions(arhmm_2_pred_bin).apply(lambda x: 1 if x > 0.5 else 0)
arhmm_2_pred_state = arhmm_2_preds.viterbi.isin([3,2,7]).astype(int)

model_results_2020["8s_arhmm_trex"] = arhmm_2_pred_bin


evaluate_model(arhmm_2_pred_bin, dust_event_info)
plot_predictions(arhmm_2_pred_bin, dust_event_info)

# save_hmm_model(arhmm_2, "8state_ar1_hmm_10trex.pkl")


# Try it with 2017 labels now 
X_10_2017 = trex_data["2017-01-01 00:00:00":"2017-12-31 23:59:59"]

arhmm_2_preds_2017 = hmm_predict(arhmm_2, X_10_2017)
arhmm_2_state_summary_2017, arhmm_2_pred_bin_2017, arhmm_2_event_states_2017 = summarize_states(arhmm_2_preds_2017.viterbi, dust_event_info_2017, bf_threshold = 1)
arhmm_2_pred_bin_2017 = arhmm_2_preds_2017.viterbi.isin([3,2,7]).astype(int)
arhmm_2_pred_bin_2017.name = "8-State ARHMM"

arhmm_2_preds_smooth_2017 = smooth_predictions(arhmm_2_pred_bin_2017).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_2_pred_bin_2017, dust_event_info_2017)
plot_predictions(arhmm_2_pred_bin_2017, dust_event_info_2017)

model_results_2017["8s_arhmm_trex"] = arhmm_2_pred_bin_2017




""" 3.  2-State, AR(1) HMM, PCA """
K3 = 2
H3 = 1

arhmm_3 = fit_hmm(pca_data, K=K3, H=H3, convergence=1e-4)
arhmm_3_preds = hmm_predict(arhmm_3, data_pca2020)
arhmm_3_preds_smooth = smooth_predictions(arhmm_3_preds.viterbi).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_3_preds.viterbi, dust_event_info)
plot_predictions(arhmm_3_preds.viterbi, dust_event_info)

save_hmm_model(arhmm_3, "2s_arhmm_pca")
model_results_2020["2s_arhmm_pca"] = arhmm_3_preds.viterbi

# Now for 2017:
arhmm_3_preds_2017 = hmm_predict(arhmm_3, data_pca2017)
evaluate_model(arhmm_3_preds_2017.viterbi, dust_event_info_2017)
plot_predictions(arhmm_3_preds_2017.viterbi, dust_event_info_2017)
model_results_2017["2s_arhmm_pca"] = arhmm_3_preds_2017.viterbi


""" 4.  8-State, AR(1) HMM, PCA"""
K4 = 8
H4 = 1

arhmm_4 = fit_hmm(pca_data, K=K4, H=H4, convergence=1e-4)
arhmm_4_preds = hmm_predict(arhmm_4, data_pca2020)

arhmm_4_state_summary, arhmm_4_pred_bin, arhmm_4_event_states = summarize_states(arhmm_4_preds.viterbi, dust_event_info, bf_threshold = 1)
arhmm_4_pred_bin.name = "8-State ARHMM"

arhmm_4_preds_smooth = smooth_predictions(arhmm_4_pred_bin).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_4_pred_bin, dust_event_info)
plot_predictions(arhmm_4_pred_bin, dust_event_info)

model_results_2020["8s_arhmm_pca"] = arhmm_4_pred_bin

# Now for 2017
arhmm_4_preds_2017 = hmm_predict(arhmm_4, data_pca2017)
arhmm_4_2017_bin = arhmm_4_preds_2017.viterbi.isin(arhmm_4_event_states).astype(int)


evaluate_model(arhmm_4_2017_bin, dust_event_info_2017)
plot_predictions(arhmm_4_2017_bin, dust_event_info_2017)
model_results_2017["8s_arhmm_pca"] = arhmm_4_2017_bin


# save_hmm_model(arhmm_4, "8state_ar1_hmm_pca.pkl")



# # Plot the 8-state means and standard deviations
# means_df = pd.read_excel("AR_HMM_8_state.xlsx", sheet_name="means").iloc[:, 1:]
# variance_df = pd.read_excel("AR_HMM_8_state.xlsx", sheet_name="variances").iloc[:, 1:]

# means_df = means_df.drop(5, axis= 0)
# varaince_df = variance_df.drop(5, axis= 0)

# # Plot the normal distributions for each feature
# num_states, num_features = means_df.shape

# # Create a color map for each state
# colors = plt.cm.tab20(np.linspace(0, 1, num_states))

# # Iterate through features and plot normal distributions
# for i in range(num_features):
#     feature_name = means_df.columns[i]
#     plt.figure(figsize=(8, 5))
#     plt.title(f"Normal Distribution for {feature_name}")
    
#     for j, state in enumerate(means_df.index):
#         mean = means_df.iloc[j, i]
#         variance = variance_df.iloc[j, i]
#         std_dev = np.sqrt(variance)
#         x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
#         density = norm.pdf(x, mean, std_dev)
        
#         plt.plot(x, density, label=f"State {state}", color=colors[j])
    
#     plt.xlabel("Value")
#     plt.ylabel("Density")
#     plt.legend()
#     plt.grid(True)
#     plt.show()




""" 5.  8-State, AR(1) HMM, AE"""
K5 = 8
H5 = 1

arhmm_5 = fit_hmm(ae_data, K=K5, H=H5, convergence=1e-4)
arhmm_5_preds = hmm_predict(arhmm_5, data_ae2020)

arhmm_5_state_summary, arhmm_5_pred_bin, arhmm_5_event_states = summarize_states(arhmm_5_preds.viterbi, dust_event_info, bf_threshold = 1)
arhmm_5_pred_bin.name = "8-State ARHMM"

arhmm_5_preds_smooth = smooth_predictions(arhmm_5_pred_bin).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_5_pred_bin, dust_event_info)
plot_predictions(arhmm_5_pred_bin, dust_event_info)

model_results_2020["8s_arhmm_ae"] = arhmm_5_pred_bin

# Now for 2017
arhmm_5_preds_2017 = hmm_predict(arhmm_5, data_ae2017)
arhmm_5_2017_bin = arhmm_5_preds_2017.viterbi.isin(arhmm_5_event_states).astype(int)


evaluate_model(arhmm_5_2017_bin, dust_event_info_2017)
plot_predictions(arhmm_5_2017_bin, dust_event_info_2017,
                 start = "2017-01-01 00:00:00")
model_results_2017["8s_arhmm_ae"] = arhmm_5_2017_bin


""" 6.  2-State, AR(1) HMM, AE"""
K6 = 2
H6 = 1

arhmm_6 = fit_hmm(ae_data, K=K6, H=H6, convergence=1e-4)
arhmm_6_preds = hmm_predict(arhmm_6, data_ae2020)
arhmm_6_pred_bin = arhmm_6_preds.viterbi

arhmm_6_preds_smooth = smooth_predictions(arhmm_6_pred_bin).apply(lambda x: 1 if x > 0.5 else 0)


evaluate_model(arhmm_6_pred_bin, dust_event_info)
plot_predictions(arhmm_6_pred_bin, dust_event_info)

model_results_2020["2s_arhmm_ae"] = arhmm_6_pred_bin

# Now for 2017
arhmm_6_preds_2017 = hmm_predict(arhmm_6, data_ae2017)
arhmm_6_pred_bin_2017 = arhmm_6_preds_2017.viterbi

evaluate_model(arhmm_6_pred_bin_2017, dust_event_info_2017)
plot_predictions(arhmm_6_pred_bin_2017, dust_event_info_2017,
                 start = "2017-01-01 00:00:00")
model_results_2017["8s_arhmm_ae"] = arhmm_6_pred_bin_2017


