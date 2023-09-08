"""

Title: results_summary.py

Description:
    Plot all model results and collect performance metrics. Plots are for use
    in the final thesis.
    
Author: Andrew Zehr

Data Updated: 07.09.2023

"""

import pandas as pd
from utils.data_handling import read_JFJ_data
from utils.model_evaluation import (
    smooth_predictions, 
    plot_predictions, 
    evaluate_model,
    )


# Load the model predictions
predictions_2020 = pd.read_csv("predictions_2020.csv", index_col=0, parse_dates = ["DateTimeUTC"])
predictions_2017 = pd.read_csv("predictions_2017.csv", index_col=0, parse_dates = ["DateTimeUTC"])

# Create naive ensemble models
#predictions_2020['ensemble'] = round(predictions_2020.sum(axis=1) / predictions_2020.shape[1])
#predictions_2017['ensemble'] = round(predictions_2017.sum(axis=1) / predictions_2017.shape[1])

# Load the ground truth labels
dust_event_info_2020 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info.csv")
dust_event_info_2017 = read_JFJ_data(r"\raw\Jungfraujoch\dust_event_info_2017.csv")



# Create new dataframes of smoothed dataframes
predictions_smoothed_2020 = predictions_2020.apply(
    smooth_predictions, axis = 0).applymap(lambda x: round(x))

predictions_smoothed_2017 = predictions_2017.apply(
    smooth_predictions, axis = 0).applymap(lambda x: round(x))


# Evaluate the simple ensemble models
evaluate_model(predictions_2020.ensemble, dust_event_info_2020)
evaluate_model(predictions_2017.ensemble, dust_event_info_2017)



""" Plot Predictions """
baselines = ["Neg. SSA", "RF", "RBF SVC"]

# Plot the baselines
plot_predictions(predictions_2020.loc[:,baselines], dust_event_info_2020,
                 title = "Baseline Models: 2020", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False)

plot_predictions(predictions_2017.loc[:,baselines], dust_event_info_2017,
                 title = "Baseline Models: 2017", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False)


# Join the labels with the model predictions
predictions_labels_2020 = pd.DataFrame(dust_event_info_2020.sde_event).join(predictions_smoothed_2020, how="left").rename(columns={"sde_event": "True"})
predictions_labels_2017 = pd.DataFrame(dust_event_info_2017.sde_event).join(predictions_smoothed_2017, how="left").rename(columns={"sde_event": "True"})



# Plot 2020 predictions, simple HMM
simpleHMM_2020 = predictions_labels_2020.loc[:, ["2s_HMM_trex", "True", "8s_HMM_trex"]]
simpleHMM_2020.columns = ["2-State", "True", "8-State"]
plot_predictions(simpleHMM_2020, dust_event_info_2020,
                 title = "Simple HMM Models: 2020", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)

# Plot 2017 predictions, simple HMM
simpleHMM_2017 = predictions_labels_2017.loc[:, ["2s_HMM_trex", "True", "8s_HMM_trex"]]
simpleHMM_2017.columns = ["2-State", "True", "8-State"]
plot_predictions(simpleHMM_2017, dust_event_info_2017,
                 title = "Simple HMM Models: 2017", date_rotation = 70,
                 dark_theme = False, show_missing=True, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)



# Plot 2020 predictions, deep models
deep_2020 = predictions_labels_2020.loc[:, ["TCN", "True", "U-NET"]]
plot_predictions(deep_2020, dust_event_info_2020,
                 title = "Deep Models: 2020", date_rotation = 70, show_missing=True, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)


# Plot 2017 predictions, deep models
deep_2017 = predictions_labels_2017.loc[:, ["TCN", "True", "U-NET"]]
plot_predictions(deep_2017, dust_event_info_2017,
                 title = "Deep Models: 2017", date_rotation = 70, show_missing=True, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)



# Plot 2020 predictions, AR-HMM
ar_HMM_2020 = predictions_labels_2020.loc[:, ["2s_arhmm_trex", "True", "8s_arhmm_trex"]]
ar_HMM_2020.columns = ["2-State", "True", "8-State"]
plot_predictions(ar_HMM_2020, dust_event_info_2020,
                 title = "AR-HMM Models: 2020", date_rotation = 70,
                 show_missing=False, font = 15, figsize=(8,5), 
                 include_legend = False, custom_df = True)

# Plot 2017 predictions, AR-HMM
ar_HMM_2017 = predictions_labels_2017.loc[:, ["2s_arhmm_trex", "True", "8s_arhmm_trex"]]
ar_HMM_2017.columns = ["2-State", "True", "8-State"]
plot_predictions(ar_HMM_2017, dust_event_info_2017,
                 title = "AR-HMM Models: 2017", date_rotation = 70,
                 dark_theme = False, show_missing=True, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)


# Plot 2020 predictions, HSMM
hsmm_2020 = predictions_labels_2020.loc[:, ["2s_HSMM", "True", "8s_HSMM"]]
hsmm_2020.columns = ["2-State", "True", "8-State"]
plot_predictions(hsmm_2020, dust_event_info_2020,
                 title = "HSMM Models: 2020", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)

# Plot 2017 predictions, AR-HMM
hsmm_2017 = predictions_labels_2017.loc[:, ["2s_HSMM", "True", "8s_HSMM"]]
hsmm_2017.columns = ["2-State", "True", "8-State"]
plot_predictions(hsmm_2017, dust_event_info_2017,
                 title = "HSMM Models: 2017", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False, custom_df = True)


# Plot 2020 predictions, supervised HMM
plot_predictions(predictions_2020.loc[:,"Supervised HMM"], dust_event_info_2020,
                 title = "Supervised HMM Model: 2020", date_rotation = 70,
                 dark_theme = False, show_missing=False, font = 15, 
                 figsize=(8,5), include_legend = False)

# Plot 2017 predictions, supervised HMM
plot_predictions(predictions_2017.loc[:,"Supervised HMM"], dust_event_info_2017,
                 title = "Supervised HMM Model: 2017", date_rotation = 70,
                 dark_theme = False, show_missing=True, font = 15, 
                 figsize=(8,5), include_legend = False)


# ['RF', 'RBF SVC', 'U-NET', 'TCN', '2s_HMM_trex', '2s_HMM_pca',
#      '2s_HMM_ae', '8s_HMM_pca', '8s_HMM_trex', '8s_HMM_ae', '2s_arhmm_trex',
#       '2s_arhmm_pca', '8s_arhmm_trex', '8s_HSMM', '2s_HSMM', 'ensemble',
#       'Neg. SSA', '2s_arhmm_ae', 'Supervised HMM'],


# Calculate the performance for all models (change model or year as needed)
evaluate_model(predictions_2020["RF"], dust_event_info_2020)

# Calculate the smoothed performance for all models (change model or year as needed)
evaluate_model(predictions_smoothed_2020["RF"], dust_event_info_2020)


