"""
Author: Andrew Zehr

Title: data_handling.py

Description:
    Includes commonly used functions for preprocessing and loading JFJ data 
    of set format.
"""

import pandas as pd
import numpy as np
from utils.constants import data_folder
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


def data_cleaner(df: pd.DataFrame, diams, Vtot_thresh = 1000, Ntot_thresh = 20000) -> pd.DataFrame:
    '''
    Return cleaned dataframe with no inf values or pnysically impossible values,
    and with proper formating of nan
    Follows format provided by Dr. Robin Modini, although is not a subsitute for
    his workflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be cleanes that matches the expected format, 
        refer to the documentation.
    diams : arraylike 
        array of size distribution midpoint diameters
    Vtot_thresh : int, optional
        Threshold for integrated volume size distribution, values over this
        threshold set to nan. The default is 1000.
    Ntot_thresh : int, optional
        Threshold for integrated number size distribution, values over this
        threshold set to nan. The default is 20000.

    Returns
    -------
    cleaned_df : TYPE
        cleaned dataframe of proper dimension.

    '''
    df_cleaned = df.copy()
    
    # Replace AAE values of 0 with NaN
    df_cleaned.loc[df_cleaned["AAE"] == 0, "AAE"] = np.nan 

    # Replace inf values with Nan
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.replace("nan", np.nan, inplace=True)
    
    
    sd_cols = [c for c in df_cleaned.columns if ('D' in c) and ('Time' not in c)]
    nsd_cols = [c for c in sd_cols if 'V' not in c]
    vsd_cols = [c for c in sd_cols if 'V' in c]
    
    
    tmp_list = []
    for _,row in df_cleaned[nsd_cols].iterrows():
        tmp_list.append(np.trapz(row, np.log10(diams)))
    df_cleaned['Ntot'] = tmp_list
    
    tmp_list = []
    for _,row in df_cleaned[vsd_cols].iterrows():
        tmp_list.append(np.trapz(row, np.log10(diams)))
    df_cleaned['Vtot'] = tmp_list
    
    data_cols2 = [c for c in df_cleaned.columns if 'Time' not in c]
    
    
    for col in ['BaCorr2_A13', 'BaCorr3_A13', 'BaCorr4_A13', 'BaCorr5_A13', 'BaCorr6_A13', 'BaCorr7_A13', 'BaCorr1_A13', 'babs_450', 'babs_550', 'babs_700']:
        df_cleaned.loc[(df_cleaned[col]<-1), data_cols2] = np.nan
        
    for col in ['SSA_450', 'SSA_550', 'SSA_700']:
        df_cleaned.loc[(df_cleaned[col]<-2), data_cols2] = np.nan
        df_cleaned.loc[(df_cleaned[col]>3), data_cols2] = np.nan 
    
    
    df_cleaned.loc[(df_cleaned['Ntot']>Ntot_thresh), data_cols2] = np.nan
    df_cleaned.loc[(df_cleaned['Vtot']>Vtot_thresh), data_cols2] = np.nan
    
    df_cleaned.drop(columns=['Ntot', 'Vtot'], inplace = True)
    
    return df_cleaned


def size_dist_handler(df: pd.DataFrame, diams, coarse_thresh = 0.5) -> pd.DataFrame:
    '''
    Adds variables corresponding to the total-integrated volume and number size
    distributions. It also adds variables corresponding to fraction of coarse particles
    in the respective distributions.
    
    In the future, other features based on the volume and number distributions
    may need to be added (mode, median, variance, KI-divergence from normal, etc...)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with volume and size distribution features. Refer to the documentation
        if making a new file format 
        
    diams : np.array or Series
        Diameter midpoint values
        
    coarse_thresh : float
        threshold above which particles are considered coarse (in micrometers)
        Defaults to 0.5 (> 500nm) which corresponds to data recorded by FIDAS OPS

    Returns
    -------
    dist_df : pd.DataFrame
        Original DataFrame with additional ddistribution features added

    '''
    
    dist_df = df.copy()
    
    diams = np.array(diams).reshape(-1)


    # Add distribution summary variables
    column_names = dist_df.columns

    nsd_cols = []
    vsd_cols = []
    other_cols = []
    for col in column_names:
        if ("V_D" in col and col != "DateTimeUTC"):
            vsd_cols.append(col)
        elif ("D" in col and col != "DateTimeUTC"):
            nsd_cols.append(col)
        else:
            other_cols.append(col)

    coarse_index = np.argmax(diams > coarse_thresh)


    nsd_coarse = nsd_cols[coarse_index:]
    vsd_coarse = vsd_cols[coarse_index:]

    tmp_list = []
    for _,row in dist_df[nsd_cols].iterrows():
        tmp_list.append(np.trapz(row[:-1], np.log10(diams)))
    dist_df['Ntot'] = tmp_list

    tmp_list = []
    for _,row in dist_df[vsd_cols].iterrows():
        tmp_list.append(np.trapz(row[:-1], np.log10(diams)))
    dist_df['Vtot'] = tmp_list


    tmp_list = []
    for _,row in dist_df[nsd_coarse].iterrows():
        tmp_list.append(np.trapz(row[:-1], np.log10(diams[coarse_index:])))
    dist_df['Ntot_coarse'] = tmp_list

    tmp_list = []
    for _,row in dist_df[vsd_coarse].iterrows():
        tmp_list.append(np.trapz(row[:-1], np.log10(diams[coarse_index:])))
    dist_df['Vtot_coarse'] = tmp_list

    
    dist_df["Ntot"] = np.where(dist_df["Ntot"] < 0, 0, dist_df["Ntot"])
    dist_df["Vtot"] = np.where(dist_df["Vtot"] < 0, 0, dist_df["Vtot"])
    
    dist_df["Nfrac_coarse"] = dist_df['Ntot_coarse'] / dist_df['Ntot']
    dist_df["Vfrac_coarse"] = dist_df['Vtot_coarse'] / dist_df['Vtot']


    dist_df.loc[(dist_df['Vfrac_coarse']>1), "Vfrac_coarse"] = 1
    dist_df.loc[(dist_df['Vfrac_coarse']<0), "Vfrac_coarse"] = 0  
    
    return dist_df


def impute_df(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    '''
    Imputes data in the selected method
    
    Defaults to "linear" imputation. 
    
    ***Note: This function should be expanded to include other imputation types,
    or to allow different variables to be imputed in different ways
    
    wishlist: iterative imputation (equivalent to MICE in R)
    ***
    
    ***Note: This function is just in case a complete data set is needed for some
    application. In many cases, it is better not to impute or just discard incomplete 
    entries. Additionally, when imputing it could be that only some rows should be imputed,
    depending on the types and number of missing values. Consider the proper 
    approach before imputing. This function does not implement iterative 
    imputation, which is used later in the project.

    Parameters
    ----------
    df : pd.DataFrame
        Original data frame to be imputed.

    Returns
    -------
    df_imputed : TYPE
        Imputed data frame.

    '''
    
    if (method == "linear"):
        df_imputed = df.interpolate(axis=0)
    else:
        df_imputed = df.copy()
    
    return df_imputed 


def read_JFJ_data(filepath: str, date_range: list = [], date_format: str = "%d/%m/%Y %H:%M:%S") -> pd.DataFrame:
    '''
    Loads JFJ data from csv, ensuring data is properly handeled
    (date set to index, etc...)

    Parameters
    ----------
    filepath : string
        filepath of the csv to be opened, starting within "main\data" folder
    
    dateRange : empty list or list with 2 elements
        Datetime range of data to be saved to dataframe. If empty, the entire csv date range
        is used. If you want to use a one sided range, just ensure the other bound
        includes all the data int he range (can go beyond)
        
        example:
            ["2018-02-23 16:00:00", "2019-02-02 02:00:00"]
            
        or "one-sided":
            ["1900-01-01 00:00:00", ""2019-02-02 02:00:00""]
                

    Returns
    -------
    data : pd.DataFrame
        dataframe of desired data

    '''
    
    full_path = data_folder + filepath
    
    data = pd.read_csv(full_path, index_col = 0, parse_dates=True)
    
    # If dates aren't parsing correctly and you are receiving the "month and day switched" error
    # try:        data = pd.read_csv(full_path, index_col = 0)
    #             data.index = pd.to_datetime(data.index, format = "%d/%m/%Y %H:%M")
    #             data.to_csv(filename)


    # Then the csv should read in properly every time after that     

    
    if (len(date_range) == 2):
        data = data[date_range[0]:date_range[1]]
        
    if data.index.equals(data.sort_index().index):
        print("Dates Read correctly")
    else:
        data = pd.read_csv(full_path, index_col = 0)
        data.index = pd.to_datetime(data.index, format = "%d/%m/%Y %H:%M")

    return data


def scale_data(df, method: str = "standard", return_params: bool = False) -> pd.DataFrame:
    '''
    Scales the data in one of a number of methods :
     "centering", "standard", "zero-one"
     
     Can return the means and standard deviations 
     
    Future: Add option to scale different rows differently


    Parameters
    ----------
    df : pd.DataFrame
        dataframe to be scaled
    method : string, optional
        Method of scaling data.
        "centering" by subtracting mean.
        "standard" by subtracting mean and dividing by std.
        "zero-one" by setting min value to 0 and max to 1 
        
        The default is "standard".
    return_params : bool, optional
        Tells function whether to return the mean and standard deviation 
        (for standard), so they can be used to scale the test set using the 
        same parameters from the train set.

    Returns
    -------
    scaled_df : pd.DataFrame
        Dataframe with scaled data
    '''
    
    if (method == "centering"):
        means = df.mean()
        std_devs = 0
        scaled_df = df - means
        
        
    if (method == "standard"):
        means = df.mean()
        std_devs = df.std()
        scaled_df = (df - means) / std_devs
    
    if (method == "zero-one"):
        mins = df.min()
        maxs = df.max()
        scaled_df = (df - mins) / maxs
    
    if (return_params):
        return scaled_df, means, std_devs
    
    else:
        return scaled_df
    
    
def drop_pca_outliers(df: pd.DataFrame, std_threshold: float = 3) -> pd.DataFrame:
    std_devs = df.std()


    # Define a function to check if a row should be dropped
    def should_drop(row):
        for col in df.columns:
            if abs(row[col]) > (std_devs[col] * std_threshold):
                return True
        return False

    # Use apply() and a lambda function to identify rows to drop
    rows_to_drop = df.apply(lambda row: should_drop(row), axis=1)

    df_clean = df[~rows_to_drop]
    
    return df_clean


def deseasonalize(df: pd.DataFrame, method: str = "month", h: float = 500) -> pd.DataFrame:
    '''
    Performs a naive deseasonalization of the data using one of two methods:
        1. Takes the average feature value for a given month, i.e. January, 
           across all years and subtracts this value from all January observations
           
        2. Calculates a smoothed value using a Gaussian filter and subtract this
        value from all observations.
        
    In the future, more complicated methods such as SSA or other decompositions
    should be used.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with datetimeindex.
    method : string, optional
        Method used to deseasonalize the data.
        "month" takes the average for each column for each month over all years. 
        (i.e. there is the average value of column 1 for the month of December in each year)
        Then the corresponding average is subtracted from each entry
        
        "smooth" uses a kernel smoother with bandwidth, h, and subtracts this
        from the original data
        
        "decompose" uses
        
        The default is "month".
    h : int, optional
        Bandwidth used for the regression smoother. The default is 500.

    Returns
    -------
    df_deaseason : TYPE
        DESCRIPTION.

    '''
    
    
    if method == "month":
        df.loc[:,"month"] = df.index.month.values
        monthAvg = pd.DataFrame(df.groupby(by = "month").mean())
        
        # Make sure "month" is the last column in df
        column_to_move = df.pop("month")
        
        #insert column with insert(location, column_name, column_value)
        df.insert(len(df.columns), "month", column_to_move)
        
        df_deseason = df.copy()
        
        for i in range(df.shape[1]-1):
            df_deseason.iloc[:,i] = df.iloc[:,i] - monthAvg.iloc[
                df_deseason["month"].values - 1, i].values
            
    
    if method == "smooth":
        seasonal = ndimage.gaussian_filter1d(df, sigma = h, axis = 0)
        df_deseason = df - seasonal
        
        
    return df_deseason.drop("month", axis = 1)


def perform_pca(df: pd.DataFrame, new_dim: float = 0.95) -> pd.DataFrame:
    """
    Returns the original dataframe projected onto PCA components. Can specify either
    a number of final components or a percentage of variance to be explained.

    Parameters
    ----------
    df : pd.DataFrame
        data to be projected
    new_dim : float, optional
        If this number is between 0 and 1, it represents the fraction of variance
        to be kept in the transformed data
        
        If this number is an integer, then it specifies the number of componets to keep.
        If it is greater than the number of columns in the original dataframe, all components
        are kept.

    Returns
    -------
    transformed_data_frame : TYPE
        PCA projected dataframe

    """
    
    if new_dim <= 0:
        raise ValueError("The 'new_dim' argument must be greater than zero.")

    

    standardized_data = (df - df.mean()) / df.std()

    # Initialize PCA with all components
    pca = PCA()

    # Fit PCA on standardized data
    pca.fit(standardized_data)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    sns.set_style("whitegrid")
    sns.lineplot(x = range(1, df.shape[1] + 1), y = cumulative_variance_ratio)
    plt.title("Explained Variance Cumulative")
    plt.ylabel("Explained Variance Proportion")
    plt.xlabel("Number of Components")
    
    if ((new_dim < 1) & (new_dim > 0)):
        plt.hlines(new_dim, xmin = 0, xmax = df.shape[1], colors = "black", 
                   linestyles = "dotted")

    if (new_dim < 1):
        # Find the number of components that explain 95% of the variance
        n_components = np.argmax(cumulative_variance_ratio >= new_dim) + 1
    elif isinstance(new_dim, int):
        # Find the number of components that explain 95% of the variance
        n_components = new_dim
        
    else:
        raise ValueError("The number of components must either be a fraction less than 1 or an integer!.")
        
        
    
    # Perform PCA with the selected number of components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(standardized_data)


    # Create a new DataFrame with transformed data
    transformed_data_frame = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)], index = df.index)

    return transformed_data_frame


def clip_dataframe(df: pd.DataFrame, limit: float = 4) -> pd.DataFrame:
    """
    Clips the dataframe so all values fall within a certain number of standard
    deviations from the average feature value. This controls for extreme values 
    and in some cases helps numerical stability of HMM. Note: this does not set
    outliers to nan's but rather sets their values to the limit value. 
    This functionality has not been thouroughly tested.

    Parameters
    ----------
    df : pd.DataFrame
        Original data to be clipped.
    limit : TYPE, optional
        Number of standard deviations data can stray before 
        being clipped. The default is 4.

    Returns
    -------
    df_clipped : TYPE
        Data with clipped outliers.

    """
    
    df_clipped = df.copy()
    
    # Calculate the limit for flagging
    means = df_clipped.mean()
    std_devs = df_clipped.std()
    
    limit = 3 * std_devs
    
    # Identify values more than 3 standard deviations away from the mean
    flagged_indices = ((df_clipped - means).abs() > limit)
    
    # Replace flagged values with values 3 std deviations away from the mean
    for column in df_clipped.columns:
        values_to_replace = df_clipped.loc[flagged_indices[column], column]
        replacement_values = np.clip(values_to_replace, means[column] - limit[column], means[column] + limit[column])
        df_clipped.loc[flagged_indices[column], column] = replacement_values

    return df_clipped
