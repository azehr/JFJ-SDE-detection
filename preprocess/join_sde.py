"""
Title: join_sde.py

Author: Andrew Zer (code adapted from Romana Boiger)

Purpose: This code joins the 2020 Jungfraujoch data with the "ground truth" SDE 
         labels. The times where a dust event took place, as well as the 
         confidence level of the dust event are added to the dataframe.
"""


import pandas as pd
import numpy as np

# Import raw data from 2020
rawData = pd.read_csv("aerosol_data_JFJ_2020.csv")
rawData['DateTimeUTC'] = pd.to_datetime(rawData['DateTimeUTC'])


def SDE_ground_truth(df, SDE_start, SDE_end, SDE_conf):
    """
    input:
          df: a pandas dataframe with the Jungfraujoch timeseries data 
          
          SDE_start: list of the start datetime of SDE events in given 
                     time frame. These should be in "YYYY-MM-DD HH:MM:SS"
                     format with the time to the nearest whole hour.
                     (ex. "2020-02-08-14:00:00")
                            
          SDE_end: list of end date time for SDE events in given time frame.
                   Same format as SDE_events_start, lists should be of 
                   equal size
                          
          SDE_conf: list of confidence levels for givenSDE, "low" or "high"
                   follows the convention in (Brunner et al. 2021),
                   can be removed in future it data doesn't follow this form
                   same length as 'SDE_end' and 'sde_start'
    
    output:
        
    df_SDE: a new dataframe, that includes the original data along with 
            the time, the number and level of confidence of the dust events
    """
    
    classif = ['none'] + SDE_conf
    confLevel = pd.DataFrame(classif, columns = ["Confidence"])
    
    sde_start_ind = list() 
    for entry in SDE_start:
        sde_start = df.loc[df['DateTimeUTC'] == entry].index[0]
        sde_start_ind.append(sde_start)
        
    sde_end_ind = list()
    for entry in SDE_end:
        sde_end = df.loc[df['DateTimeUTC'] == entry].index[0]
        sde_end_ind.append(sde_end)
    
    
    ind_list = list()
    event_list = list()
    k = 1
    for i in range(len(sde_end_ind)):
        j = sde_start_ind[i]
        while j <= sde_end_ind[i]:
            ind_list.append(j)
            j= j+1
            event_list.append(k)
        k = k+1
    
    sde_event = np.zeros_like(df['AE_SSA'].values)
    sde_event[ind_list] = 1
    
    sde_event_nr = np.zeros_like(df['AE_SSA'].values)
    sde_event_nr[ind_list] = event_list
    
    
    df['sde_event'] = sde_event
    df['sde_event_nr'] = sde_event_nr
    df_SDE = df.join(confLevel, "sde_event_nr").set_index("DateTimeUTC")
    
    return df_SDE

# Define the SDE start and end times, along with the confidence levels (if applicable)
SDE_start = [
    '2020-02-08 14:00:00',
    '2020-02-28 17:00:00',
    '2020-03-09 14:00:00', 
    '2020-03-16 11:00:00',
    '2020-03-19 16:00:00',
    '2020-03-25 21:00:00',
    '2020-04-06 08:00:00',
    '2020-04-16 12:00:00',
    '2020-04-21 06:00:00',
    '2020-05-04 07:00:00',
    '2020-05-08 13:00:00',
    '2020-05-10 14:00:00',
    '2020-05-13 00:00:00',
    '2020-05-14 18:00:00',
    '2020-05-16 18:00:00',
    '2020-06-24 03:00:00',
    '2020-07-09 23:00:00',
    '2020-07-28 03:00:00',
    '2020-08-01 00:00:00',
    '2020-08-05 18:00:00',
    '2020-08-09 13:00:00',
    '2020-09-14 04:00:00',
    '2020-09-17 16:00:00',
    '2020-10-19 09:00:00',
    '2020-11-06 09:00:00',
    '2020-11-23 12:00:00',
    ]

SDE_end = [
    '2020-02-09 06:00:00',
    '2020-02-29 21:00:00',
    '2020-03-11 07:00:00', 
    '2020-03-18 15:00:00',
    '2020-03-22 22:00:00',
    '2020-03-31 18:00:00',
    '2020-04-07 12:00:00',
    '2020-04-17 23:00:00',
    '2020-04-21 16:00:00',
    '2020-05-05 19:00:00',
    '2020-05-10 10:00:00',
    '2020-05-11 06:00:00',
    '2020-05-13 15:00:00',
    '2020-05-16 05:00:00',
    '2020-05-17 02:00:00',
    '2020-06-29 00:00:00',
    '2020-07-10 23:00:00',
    '2020-07-29 03:00:00',
    '2020-08-02 01:00:00',
    '2020-08-08 12:00:00',
    '2020-08-13 21:00:00',
    '2020-09-15 20:00:00',
    '2020-09-20 23:00:00',
    '2020-10-23 09:00:00',
    '2020-11-09 03:00:00',
    '2020-11-25 03:00:00',
    ]

SDE_conf = [
    'high',
    'high',
    'low',
    'high',
    'high',
    'high',
    'low',
    'high',
    'low',
    'high',
    'low',
    'high',
    'high',
    'high',
    'high',
    'low',
    'high',
    'low',
    'low',
    'low',
    'low',
    'low',
    'low',
    'high',
    'high',
    'low',
    ]


data2020 = SDE_ground_truth(rawData, SDE_start, SDE_end, SDE_conf)


# Write the data to CSV

data2020.to_csv("SDE_2020_raw_data.csv")



sde_info = pd.DataFrame(
    {"start" : SDE_start, "end" : SDE_end, "confidence" : SDE_conf})

# sde_info.to_csv("SDE_start_end_times.csv")
