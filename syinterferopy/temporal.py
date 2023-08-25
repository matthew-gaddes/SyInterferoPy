#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 11:34:39 2023

@author: matthew
"""




#%%

def generate_random_tcs(n_tcs = 100, d_start = "20141231", d_stop = "20230801",
                        min_def_rate = 1., max_def_rate = 2. ):
    """ Generate n_tcs random time courses (ie temporal behavious of deformation.  )
    
    Inputs:
        n_tcs | int | Number of time series to generate.  
        d_start | string | start date, form YYYYMMDD
        d_end | string | start date, form YYYYMMDD
        min_def_rate | float | m/yr minimum deformation rate.  
        max_def_rate | float | m/yr maximum deformation rate.  
    Returns:
        tcs | rank 2 array | time courses as row vectors.  n_tcs rows.  
        def_dates | list of datetimes | all the dates that we have deformation for.  
    History:
        2023_08_24 | MEG | Written.  
    """
    import numpy as np
    from syinterferopy.temporal import tc_uniform_inflation
    
    for ts_n in range(n_tcs):
        def_rate = np.random.uniform(min_def_rate, max_def_rate)                                       # random deformation rate.  
        tc_def, def_dates = tc_uniform_inflation(def_rate, d_start, d_stop)         # generate time course
        if ts_n == 0:                                                               # if the first time.
            tcs = np.zeros((n_tcs, tc_def.shape[0]))                                # initialise array to store results
        tcs[ts_n, :] = tc_def                                                       # record one result as row vector to array.  
    return tcs, def_dates


#%%

def tc_uniform_inflation(def_rate = 0.07, d_start = "20141231", d_stop = "20230801"):
    """ Calculate the magnitdue of an inflating signal at each day.  Inflation is linear.  
    Inputs:
        def_rate | float | deformation rate, my/yr
        d_start | str | YYYYMMDD of when to start time series.  Inclusive.  
        d_stop | str | YYYYMMDD of when to stop time series.  Not inclusive.  
    Returns:
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    History:
        2023_08_17 | MEG | Written
    """
    from datetime import datetime, timedelta
    import numpy as np


    # make cumulative deformaitn for each day
    dstart = datetime.strptime(d_start, '%Y%m%d')                              # conver to dateteime
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # convert to datetime
    n_days = (dstop - dstart).days                                             # find number of days between dates
    max_def = def_rate * (n_days / 365)                                         # calculate the maximum deformation
    tc_def = np.linspace(0, max_def, n_days)                                    # divide it up to calucaulte it at each day
    
    # make datetime for each day.  
    def_dates = [dstart]
    dcurrent = def_dates[-1]

    while dcurrent < dstop:
        dnext = dcurrent + timedelta(days = 1)                                  # advance by one day
        def_dates.append(dnext)
        dcurrent = def_dates[-1]                                                # record the current date
    
    def_dates = def_dates[:-1]                                                  # drop last date
    return tc_def, def_dates

#%%


def generate_uniform_temporal_baselines(d_start = "20141231", d_stop = "20230801", tbaseline = 12):
    """ Given a date range, generate LiCSAR style short temporal baseline ifgs of the same size.  
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop | str | YYYYMMDD of when to stop time series
        tbaseline | int | temporal baseline, in days 
    Returns:
        acq_dates | list of datetimes | acq dates.  
        tbaselines | list of ints | timeporal baselines of short temporal baseline ifgs.  First one is 0.  
    History:
        2023_08_24 | MEG | Adapts from genereate_random_temporal_baselines.  
    """

    from datetime import datetime, timedelta
    import numpy as np

    dstart = datetime.strptime(d_start, '%Y%m%d')                              # 
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # 

    
    acq_dates = [dstart]
    tbaselines = [0]
    dcurrent = acq_dates[-1]

    while dcurrent < dstop:
        dnext = dcurrent + timedelta(days = tbaseline)                           # add the temp baseline to find the new date   
        if dnext < dstop:                                                        # check we haven't gone past the end date 
            acq_dates.append(dnext)                                              # if we haven't, record            
            tbaselines.append(tbaseline)
            dcurrent = acq_dates[-1]                                             # record the current date
        else:
            break                                                                # remember to exit the while if we have got to the last date

    return acq_dates, tbaselines



#%%

def generate_random_temporal_baselines(d_start = "20141231", d_stop = "20230801"):
    """ Given a date range, generate LiCSAR style short temporal baseline ifgs.  Takes into account that S1b was operational and 
    there were more 6 day ifg then.  
    Inputs:
        d_start | str | YYYYMMDD of when to start time series
        d_stop | str | YYYYMMDD of when to stop time series
    Returns:
        acq_dates | list of datetimes | acq dates.  
        tbaselines | list of ints | timeporal baselines of short temporal baseline ifgs.  First one is 0.  
    History:
        2023_08_17 | MEG | Written
    """
    usual_tbaselines = [6, 12, 12, 12, 12, 12, 12, 12, 12, 12,                  # temp baseline is chosen from this list at random
                         24, 24, 24, 24, 24, 24, 36, 48, 60, 72]                # If number features more, will be chosen more

    from datetime import datetime, timedelta
    import numpy as np

    # hard coded.  Not exact (i.e. some frames start and stop at different times.)
    s1b_dstart = datetime.strptime("20160901", '%Y%m%d')                       # launch 20160425, ramp up to fully operational over next few months.  
    s1b_dstop = datetime.strptime("20161223", '%Y%m%d')                        # power failure ended mission

    dstart = datetime.strptime(d_start, '%Y%m%d')                              # 
    dstop = datetime.strptime(d_stop, '%Y%m%d')                                # 

    
    acq_dates = [dstart]
    tbaselines = [0]
    dcurrent = acq_dates[-1]

    while dcurrent < dstop:
        if (s1b_dstart < dcurrent) and (dcurrent < s1b_dstop ):                  # check if the next date is during the S1b years.  
            tbaseline =  int(np.random.choice(usual_tbaselines[0:]))             # 6 day ifg possible
        else:
            tbaseline = int(np.random.choice(usual_tbaselines[1:]))       # 6 day ifg not possible 
    
        dnext = dcurrent + timedelta(days = tbaseline)                           # add the temp baseline to find the new date   
        if dnext < dstop:                                                        # check we haven't gone past the end date 
            acq_dates.append(dnext)                                              # if we haven't, record            
            tbaselines.append(tbaseline)
            dcurrent = acq_dates[-1]                                             # record the current date
        else:
            break                                                                # remember to exit the while if we have got to the last date

    return acq_dates, tbaselines


#%%



def sample_deformation_on_acq_dates(acq_dates, tc_def, def_dates):
    """ Given deformation on each day (tc_def and def_dates), find the deformation
    on the acquisition days (acq_dates).  
    
    Inputs:
        acq_dates | list of datetimes | acq dates.  
        tc_def | list of floats | cumulative deformation on each day.  Note that last day is not included.  
        def_dates | list of dates | dateteime for each day there is deformation for.  Note that last day is not includes
    Returns:
        tc_def_resampled | r2 numpy array | cumulative deformation on each acquisition day.  
    History:
        2023_08_23 | MEG | Written
    """
    
    import numpy as np
    
    n_acq = len(acq_dates)                                                  # 
    tc_def_resampled = np.zeros((n_acq, 1))                                 # initialise as empty (zeros)
    
    for acq_n, acq_date in enumerate(acq_dates):                        
        day_arg = def_dates.index(acq_date)                                 # find which day number the acquiisiont day is
        day_def = tc_def[day_arg]                                           # get the deformaiton for that day
        tc_def_resampled[acq_n, 0] = day_def                                # record
    
    return tc_def_resampled


#%%

        

def defo_to_ts(defo, tc):
    """ Multiply a rank 2 deformation (ie images) by a time course to make a 
    rank 3 time series.  Time course is cumulative and for each acquisition, so first one is 0.  
    Returns short temporal baseline ifgs, so first one is not zero.  
    
    Inputs:
        defo | r2 ma | deformatin patter, some pixels masked.  
        tc | r1 | cumulative time course.  
    Returns:
        defo_ts | r3 ma | cumulative time course.  
    History:
        2023_08_24 | MEG | Written.  
    """
    import numpy as np
    import numpy.ma as ma
    
    defo_ts = ma.zeros((tc.shape[0], defo.shape[0], defo.shape[1]))
    
    for acq_n, tc_value in enumerate(tc):
        defo_ts[acq_n, ] = defo * tc_value
        
    defo_ts_ifgs = ma.diff(defo_ts, axis = 0)                                   # convert from cumulative to daisy chain of ifgs (i.e. short temporal baseline)
    return defo_ts_ifgs
