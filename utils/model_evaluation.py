"""
Title: model_evaluation.py

Description:
    Contains helper functions for evaluating model performance, goodness of fit,
    as well as post-processing of the predictions.
    
    - Caluclate and display performance metrics
    - Plot model predictions compared to ground truth
    - Goodness of fit measures for the HMMs
    - Smooth predictions
    - Investigate and interpret hidden states 

Author: Andrew Zehr

"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, jaccard_score, balanced_accuracy_score, 
    precision_score, recall_score)

import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns


def evaluate_model(y_pred, dust_event_info: pd.DataFrame, return_values = False, 
                   print_eval = True, coverage_threshold = 0.5):
    '''
    Prints performance report for the model. If needed, it can also return the 
    individual metrics

    Parameters
    ----------
    dust_event_info : DataFrame
        Data frame included in "data\raw\Jungfraujoch\dust_event_info.csv"
        or "data\raw\Jungfraujoch\dust_event_info_2017.csv"
        
    y_pred : pd.Series
        Model predictions with datetime index.
        
    return_values: bool
        Whether to return the individual metrics in a dictionary. 
        Default is False.
        
    print_eval: bool
        Whether to print the output string or not. Default is True.
    
    coverage_threshold: float
        The percentage of a given dust event that needs to be correctly identified
        for it to be counted as detected. Including a value of zero means that any
        dust event which has one hour predicted within it will be counted as a
        hit

    Returns
    -------
    By default, just prints the model performance output. If desired, can return
    dictionary of metrics.
    
    '''
    
    
    y_true = dust_event_info["sde_event"] 
    y_true = y_true[y_pred.index]
    
    
    comparison = pd.DataFrame({
        "event": y_true, 
        "pred": y_pred, 
        "event_num": (dust_event_info["sde_event_nr"])[y_pred.index],
        "conf": (dust_event_info["Confidence"])[y_pred.index]})
   
    grouped = comparison.groupby("event_num")
    
    coverage = []
    confidence = []
    
    total_events = len(np.unique(comparison.event_num)) - 1
    
    for event_num, df in grouped:
        if (event_num != 0):
            coverage.append(np.mean(df.pred))
            confidence.append(df.conf[0])
            
    # Calculates the performance metrics  
    performance = pd.DataFrame({"coverage" : coverage, "confidence": confidence})
    
    num_missed = performance[performance.coverage == 0].shape[0]
    
    num_covered = performance[performance.coverage > coverage_threshold].shape[0]
    
    high_confidence = performance[performance.confidence == "high"]
    low_confidence = performance[performance.confidence == "low"]
    
    total_high = high_confidence.shape[0]
    total_low = low_confidence.shape[0]
    
    num_missed_high = high_confidence[high_confidence.coverage == 0].shape[0]
    num_covered_high = high_confidence[high_confidence.coverage > coverage_threshold].shape[0]
    
    
    num_missed_low = low_confidence[low_confidence.coverage == 0].shape[0]
    num_covered_low = low_confidence[low_confidence.coverage > coverage_threshold].shape[0]
    
    avg_coverage = np.mean(performance.coverage)
    avg_coverage_high = np.mean(high_confidence.coverage)
    avg_coverage_low = np.mean(low_confidence.coverage)
    
    
    balanced_score = balanced_accuracy_score(comparison.event, comparison.pred)
    jacc_score = jaccard_score(comparison.event, comparison.pred)

    confusion_mat = confusion_matrix(y_true, y_pred) / np.sum(
        confusion_matrix(y_true, y_pred), axis = 1).reshape(-1,1)
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    # Formats the output string to print
    output_string = f"""
         Model results:
         -------------------------------------------------------------
        
         Balanced Accuracy: {(100 * balanced_score):.3}%
         Positive (Event) Accuracy: {(100 * confusion_mat[1,1]):.3}%
         Negative (No Event) Accuracy: {(100 * confusion_mat[0,0]):.3}%
         
         Jaccard Index: {jacc_score:.3}
         
         Precision: {(100 * precision):.3}%
         Recall: {(100 * recall):.3}%
         
         
         In this data there are a total of {total_events} dust events
         
         Number of completely missed events: {num_missed} ({(100 * num_missed / total_events):.3}%)
         Number of events with over {(coverage_threshold * 100)}% coverage: {num_covered} ({(100 * num_covered / total_events):.3}%)
         
         The average coverage of the dust events is {(avg_coverage * 100):.3}%
             
         Of all events, {total_high} are classified as "high" confidence events
         and {total_low} are classified as "low" confidence events
         
         Average coverage of High-confidence events: {(avg_coverage_high * 100):.3}%
         Average coverage of Low-confidence events: {(avg_coverage_low * 100):.3}%

         Number of missed High-confidence events: {num_missed_high} ({(100 * num_missed_high / total_high):.3}%)
         Number of missed Low-confidence events: {num_missed_low} ({(100 * num_missed_low / total_low):.3}%)
         
         Number of High-confidence events with over {(coverage_threshold * 100)}% coverage: {num_covered_high} ({(100 * num_covered_high / total_high):.3}%)
         Number of Low-confidence events with over {(coverage_threshold * 100)}% coverage: {num_covered_low} ({(100 * num_covered_low / total_low):.3}%)
         --------------------------------------------------------------
    """
    if print_eval:
        print(output_string)

    
    if return_values:
        return {"neg_acc": confusion_mat[0,0],"pos_acc": confusion_mat[1,1],
                "bal_acc": balanced_score, "jacc_score": jacc_score, 
                "num_missed": num_missed, "precision": precision,
                "recall": recall}


def plot_predictions(
        df: pd.DataFrame,
        dust_event_info: pd.DataFrame,
        date_freq = 336,
        date_rotation = 90,
        figsize = (8,4),
        start = None,
        end = None,
        show_missing = True,
        title = "",
        dark_theme = False,
        returnFig = False,
        font = 10,
        include_legend = True,
        custom_df = False
        ):
    '''
    Plots the model predictions against the ground truth labels. The plot is
    highly customizable and can include more than one model's predictions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame (or Series) of model predictions. Each column represents a different
        model and the index is the data time. If 'custom_df = True', this can include
        the ground truth.
    dust_event_info : pd.DataFrame
        DataFrame of SDE occurences.
    date_freq : int, optional
        The gap between date ticks on x-axis. The default is 336.
    date_rotation : int, optional
        Angle of date tick labels, (90 = vertical). The default is 90.
    figsize : list, optional
        THe dimensions of the plot -> (width, height). The default is (8,4).
    start : string, optional
        DateTime for when the plot should start 
        (format is "yyyy-mm-dd hh:mm:ss"). The default is None.
    end : string, optional
        DateTime for when plot should end. See above for format.
        The default is None.
    show_missing : bool, optional
        Whether to plot missing data or not. The default is True.
    title : string, optional
        Title for plot. The default is "".
    dark_theme : bool, optional
        Whether to plot in color (currently blue), or black and white.
        The default is False.
    returnFig : bool, optional
        Whether to return the figure or just plot it. If you want access to the
        axes, you must return it. The default is False.
    font : int, optional
        Font size, must be adjusted with larger figure sizes. The default is 10.
    include_legend : bool, optional
        Whether to include color bar. The default is True.
    custom_df : bool, optional
        Indicates if 'df' should be ploted as is or have the ground truth labels
        added to it. The default is False.

    Returns
    -------
    fig : figure
        Returns plot figure. Otherwise, just displays the figure in the console.

    '''
    
    if start == None:
        start = min(df.index)
    if end == None:
        end = max(df.index)
    
    # Joins ground truth labels to model predictions
    y = pd.DataFrame(dust_event_info.sde_event[start:end])
    df_to_plot = y.join(df, how="left").rename(columns={"sde_event": "True"})
    
    if custom_df:
        df_to_plot = df.copy()
    
    
    if show_missing:
        df_to_plot.fillna(2, inplace=True)
        if dark_theme:
            # Edit the following colors if desired: [non-SDE, SDE, missing data]
            cmap = colors.LinearSegmentedColormap.from_list('Custom', ["#f5f5f5", "#000000", "#e1e1e1"], 3)
        else:
            cmap = colors.LinearSegmentedColormap.from_list('Custom', ["#f5f5f5", "#0c96eb", "#e1e1e1"], 3)
        ticks = [0.3, 1, 1.7]
        labels = ['No-SDE', 'SDE', 'Missing']
    else:
        df_to_plot.dropna(inplace=True)
        if dark_theme:
            # Edit the following colors if desired: [non-SDE, SDE]
            cmap = colors.LinearSegmentedColormap.from_list('Custom', ["#f5f5f5", "#000000"], 2)
        else:
            cmap = colors.LinearSegmentedColormap.from_list('Custom', ["#f5f5f5", "#0c96eb"], 2)

        ticks = [0.25, 0.75]
        labels = ['No-SDE', 'SDE']


    dates = [s[:10] for s in df_to_plot.index.strftime("%m.%d")]
    date_locs = list(range(len(dates)))
    
    plt.rcParams.update({'font.size': font})
    fig = plt.figure(figsize = figsize)     
    sns.set_style("ticks")
    ax = sns.heatmap(np.transpose(df_to_plot), cmap = cmap, cbar=include_legend)
    ax.invert_yaxis()
    
    if include_legend:
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels(labels)
        
    plt.xticks(date_locs[::date_freq], dates[::date_freq])
    plt.xticks(rotation=date_rotation)
    plt.xlabel("")
    plt.tight_layout()
    plt.title(title)
    
    if returnFig:
        return fig
    
    
def visualize_feature_separation(df: pd.DataFrame, dust_event_info: pd.DataFrame,
                                 file_name: str):
    '''
    Outputs a pdf with graphs showing the distribution of features for both 
    SDEs and non-SDEs. This is useful for checking how much separation there is
    and evaluating the effectiveness of dimensionality reduction efforts.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with data and datetime index.
    dust_event_info : pd.DataFrame
        Dataframe containing occurences of SDEs.
    file_name : string
        File name for output PDF.

    Returns
    -------
    None.
    
    '''
    y = dust_event_info.sde_event
    
    
    merged_df = pd.merge(df, y, how = "left", 
                         left_index=True, right_index=True)


    # Boolean index to select only rows where bool_col is True in B
    bool_index = merged_df['sde_event'] == 1

    # Select only rows where bool_col is True in B and keep only columns from A
    X_train_SDE = merged_df[bool_index].drop(columns="sde_event")

    X_train_no_SDE = merged_df[~bool_index].drop(columns="sde_event")
    
    sns.set_style("whitegrid")



    # See if there is seperation between the feature distributions for hours with 
    # dust events compared to hours without dust events

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(file_name) as pdf_pages:
        
        for i, col in enumerate(X_train_no_SDE):
            figu = plt.figure(i)
            sns.histplot(data = np.array(X_train_no_SDE["2020-01-01 00:00:00":].iloc[:,i]), label = "No SDE")
            sns.histplot(data = np.array(X_train_SDE["2020-01-01 00:00:00":].iloc[:,i]), color = "orange", label = "SDE")
            plt.title("Feature " + str(X_train_SDE.columns[i]))
            plt.legend()
            plt.xlabel("Value")
            pdf_pages.savefig(figu)
            
                       
def smooth_predictions(pred: pd.Series, weights = [1/9, 2/9, 1/3, 2/9, 1/9]) -> pd.Series:
    """
    This function smooths the predicted time series prediction, helping to eliminate
    singleton predictions. The length of the smoothing window and the weights can 
    be edited, but the default is a triangular kernel of length 5

    Parameters
    ----------
    pred : pd.Series
        Series of binary predictions.
    weight_values : TYPE, optional
        Weights of the smoothing kernel. The length of the kernel must be odd.
        The default is a triangular kernel: weights = [1/9, 2/9, 3/9, 2/9, 1/9]
        
        [1/3, 1/3, 1/3] is another good option (rectangular kernel)

    Returns
    -------
    pred_smooth : pd.Series
        Smoothed binary predictions.

    """
    
    window_length = len(weights)
    
    if window_length % 2 == 0:
        print("Window Length must be odd")
        return None
    
    if (sum(weights) > 1.01) & (sum(weights) < 0.99):
        # Small window given to allow for numerical errors
        print("Warning: The weights do not sum to 1")

    
    pred = pd.Series(pred)
    
    def smoother(window):
        return np.dot(window, weights)
    
    pred_smooth = (pred.rolling(window_length, center=True)
                   .apply(smoother)
                   .fillna(0))
    
    return pred_smooth
    

def summarize_states(preds: pd.Series, dust_event_info: pd.DataFrame, bf_threshold = 2) -> pd.DataFrame:
    """
    Compares the predicted states of a multi-state model to the ground truth labels.
    It calculates the Bayes factor for each state which is how much higher the 
    probability is of a certain hour being a dust event given that it is in the
    given state. A value greater than 1 means that state is correlated with SDEs
    and a value below one indicates it is associated with non-SDEs. It then turns
    the predictions into a binary 0 or 1 prediction based on a Bayes Factor threshold.
    
    Returns a dataframe summarizing the states.

    Parameters
    ----------
    preds : pd.Series
        State predictions of a state-space model.
    dust_event_info : pd.DataFrame
        Dataframe with SDE occurences.
    bf_threshold : TYPE, optional
        The threshold of Bayes Factor. The default is 2.

    Returns
    -------
    states_summary : pd.DataFrame
        Dataframe summarizing the states and how often they occur and how realted
        they are to dust events.
    binary_preds : pd.Series
        Binary predictions based on Bayes factor.
    event_state_idx : list
        The states that are classified as SDEs.

    """
    
   
    Y = dust_event_info.loc[preds.index].sde_event
   
    state_dust_comparison = pd.DataFrame({"State": preds, "Dust Event": Y})
    # state_dust_comparison["month"] = state_dust_comparison.index.month

    
    dustEvent_counts = state_dust_comparison.groupby(["State"]).apply(lambda df: sum(df["Dust Event"]))
    no_dustEvent_counts = state_dust_comparison.groupby(["State"]).apply(lambda df: sum(-1 * df["Dust Event"] + 1))
    counts = dustEvent_counts + no_dustEvent_counts


    states_summary = pd.DataFrame({"event": dustEvent_counts, "no event": no_dustEvent_counts, "total": counts})
    states_summary["fraction_events"] = states_summary.event / states_summary.total
    states_summary = states_summary.sort_values(by = "fraction_events", ascending = False)

    # states_summary["prob_state_given_dust"] = ((states_summary.total / total_hours) * states_summary.fraction_events) / Y.mean()
    states_summary["bayes_factor"] = states_summary.fraction_events / Y.mean()
    
    event_state_idx = states_summary.index[states_summary.bayes_factor >= bf_threshold].values
    binary_preds = pd.Series(np.isin(preds, event_state_idx).astype(int), index = Y.index)
    
    return states_summary, binary_preds, event_state_idx
   
    
def plot_hmm_states(df: pd.DataFrame, model, state_preds: pd.Series, event_states, include_histograms = True, fill_kde = False):
    """
    Plots the fitted emission distributions for SDEs and non-SDEs, compared to 
    the empirical distribution. Also compares the fitted and empirical covariances.
    Finally, plots the transition matrix of the model. If it is a multi-state 
    model, it combines the states' emission distributions into a Gaussian mixture.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of data used to fit or test model.
    model : TYPE
        HMM model object.
    state_preds : pd.Series
        The predicted states with a datetime index. 
    event_states : list
        List of states corresponding to SDEs.
    include_histograms : bool, optional
        Whether to include histogram of empirical data with emission dist.
        The default is True.
    fill_kde : bool, optional
        Whether to shade in the emission density plots. The default is False.

    Returns
    -------
    None.

    """
    full_cov = model.covars_
    variances = np.diagonal(full_cov, axis1 = 1, axis2 = 2)
    covariance_matrix = np.mean(full_cov, axis = 0)
    
    
    
    means = model.means_
    transition = model.transmat_
    
    event_rows = df[state_preds.isin(event_states)]
    non_event_rows = df[~state_preds.isin(event_states)]
    
    # Plot univariate emission distributions for SDE, non-SDE
    event_means = means[event_states, :]
    non_event_means = np.delete(means, event_states, axis = 0)
    event_stds = np.sqrt(variances[event_states, :])
    non_event_stds = np.sqrt(np.delete(variances, event_states, axis = 0))
    
    state_counts =  state_preds.value_counts()
    event_state_counts = state_counts[event_states].sort_index().values
    non_event_state_counts = state_counts[~state_counts.index.isin(event_states)].sort_index().values
    
    event_state_weights = event_state_counts / event_state_counts.sum()
    non_event_state_weights = non_event_state_counts / non_event_state_counts.sum()
    
    sns.set_style("whitegrid")
    
    num_samples = 50000
    for j, feature in enumerate(df.columns):
        event_data = []
        non_event_data = []
        real_data_events = event_rows[feature]
        real_data_non_events = non_event_rows[feature]
        for i in range(len(event_state_weights)):
            event_data_points = np.random.normal(event_means[i,j], event_stds[i,j], int(num_samples * event_state_weights[i]))
            event_data = np.concatenate((event_data, event_data_points))
        for i in range(len(non_event_state_weights)):
            non_event_data_points = np.random.normal(non_event_means[i,j], non_event_stds[i,j], int(num_samples * non_event_state_weights[i]))
            non_event_data = np.concatenate((non_event_data, non_event_data_points))

        sns.kdeplot(event_data, label='SDE', color = "#dd8452", fill = fill_kde)
        sns.kdeplot(non_event_data, label='Non-SDE', color = "#4c72b0", fill = fill_kde)
        
        
        if include_histograms:
            sns.histplot(real_data_events, stat = "density", color = "#dd8452")
            sns.histplot(real_data_non_events, stat = "density", color = "#4c72b0")
            
        plt.title(f"HMM Emission Probabilities for {feature}")
        plt.legend()
        plt.show()


    # Heatmap of correlation matrix between features
    sqrt_diagonal = np.sqrt(np.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / np.outer(sqrt_diagonal, sqrt_diagonal)
    
    sns.heatmap(correlation_matrix, xticklabels=df.columns, yticklabels=df.columns)
    plt.title("Correlation between features HMM emissions")
    plt.show()
    
    sns.heatmap(df.corr(), xticklabels=df.columns, yticklabels=df.columns)
    plt.title("Empirical Correlation between features ")
    plt.show()
    
    # Heatmap of transition matrix
    sns.heatmap(transition, annot=True)
    plt.title("Transition Matrix")
    plt.show()
    