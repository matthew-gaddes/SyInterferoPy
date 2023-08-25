#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:16:02 2020

@author: matthew
"""

import pdb

#%%


def create_random_ts_1_volc(outdir, dem_dict, n_pix = 224, d_start = "20141231", d_stop = "20230801",
                            n_def_location = 10, n_tcs = 10, n_atms = 10,
                            topo_delay_var = 0.00005, turb_aps_mean = 0.02):
    """Create random time series of Sentinel-1 data for a given volcano.  
    n_def_location random crops of the DEM with the deformation placed somewhere randomly are made (the majority of the deformation must be on land.  )
    n_tcs patterns for the temporal evolution of the deformation are made.  
    n_atms instances of atmospheric noise are made (turbulent and topographically correlated)
    Total number of time series = n_def_location * n_tcs * n_atms.  
    
    Inputs:
        dem_dict | dict | DEM and vairous parameters.  
        n_pix | int | size out interferograms outp.
        d_start | string | YYYYMMDD of time series start
        d_stop | string | YYYYMMDD of time series end
        
        n_def_locations | int | Number of random DEM crops with random deformation placement.  
        n_tcs | int | Number of random deformation time courses.  
        n_atms | int | Number of random time series atmospheres.  
        
        topo_delay_var | float | variance of the delays.  Higher value gives more variety in atmospheres, so bigger interfergoram atmospheres.  
        turb_aps_mean | float | mean strength of turbulent atmospheres, in metres.  Note the the atmosphere_turb funtion takes cmm, and the value provided in m is converted first
        
    Returns:
        Saves pickle file of the time series, the deformaiton time course, and hte DEM.  
        
    History:
        2023_08_24 | MEG | Written.  
    
    """
    import numpy.ma as ma
    import pickle
    
    from syinterferopy.random_generation import generate_dems_and_defos
    from syinterferopy.atmosphere import atmosphere_turb, atmosphere_topos
    from syinterferopy.temporal import defo_to_ts, generate_random_temporal_baselines, sample_deformation_on_acq_dates
    from syinterferopy.temporal import generate_uniform_temporal_baselines, generate_random_tcs
    from syinterferopy.aux import rescale_defo
   
    out_file_n = 0
    
    defos_m, dems = generate_dems_and_defos(dem_dict, n_pix, min_deformation = 0.05, max_deformation = 0.25, n_def_location = n_def_location)             # generate multiple crops of DEM with deformation places randomly on it.  
    defos = rescale_defo(defos_m, magnitude = 1)                                                                                                            # rescale deformation so maximum is awlays 1
    tcs, def_dates = generate_random_tcs(n_tcs, d_start, d_stop, min_def_rate = 1., max_def_rate = 2.)                                                                                        # generate random time courses for the deformation. 
    volc_outdir_name =  dem_dict['name'].replace(' ', '_')
    (outdir / volc_outdir_name).mkdir(parents = True, exist_ok = True)
    
    for defo_n, defo in enumerate(defos):
        for tc_n, tc in enumerate(tcs):
            #acq_dates, tbaselines = generate_random_temporal_baselines(d_start, d_stop)
            acq_dates, tbaselines = generate_uniform_temporal_baselines(d_start, d_stop)
            tc_resampled = sample_deformation_on_acq_dates(acq_dates, tc, def_dates)                                                # get the deformation on only the days when there's a satellite acquisitions.  Note that this is 0 on the first date.  
            defo_ts = defo_to_ts(defo, tc_resampled)                                                                                                        # time series of just the deformation.  ((n-acq) - 1 x ny x nx)
            
            for atm_n in range(n_atms):
                atm_turbs = atmosphere_turb(len(acq_dates)-1, dem_dict['lons_mg'][:n_pix,:n_pix], dem_dict['lats_mg'][:n_pix,:n_pix], mean_m = turb_aps_mean)           # generate some random atmospheres, turbulent.
                atm_topos = atmosphere_topos(len(acq_dates)-1, dems[defo_n,], delay_grad = -0.0003, delay_var = topo_delay_var)                                                              # generate some random atmospheres, topographically correlated
                ts = defo_ts + atm_turbs + atm_topos
                if atm_n == 0:
                    tss = ma.zeros((n_atms,  ts.shape[0], ts.shape[1], ts.shape[2]))                                                                # initiliase
                tss[atm_n,] = ts
                print(f"Deformation location: {defo_n} Time course: {tc_n} Atmosphere {atm_n}")
                
                # possible debug figure.  
                # n_cols = 10
                # f, axes = plt.subplots(4,n_cols)
                # all_signals = np.concatenate((defo_ts[np.newaxis], atm_turbs[np.newaxis], atm_topos[np.newaxis], ts[np.newaxis]), axis = 0)
                # signal_max = ma.max(all_signals)
                # signal_min = ma.min(all_signals)
               
                # for col_n in range(n_cols):
                #     axes[0, col_n].matshow(defo_ts[col_n,:,:], vmin = signal_min, vmax = signal_max)
                #     axes[1, col_n].matshow(atm_turbs[col_n,:,:], vmin = signal_min, vmax = signal_max)
                #     axes[2, col_n].matshow(atm_topos[col_n,:,:], vmin = signal_min, vmax = signal_max)
                #     axes[3, col_n].matshow(ts[col_n,:,:], vmin = signal_min, vmax = signal_max)
                # for ax in np.ravel(axes):
                #     ax.set_xticks([])
                #     ax.set_yticks([])
                
            with open(outdir / volc_outdir_name / f'defo_{defo_n:06d}_tc_{tc_n:06d}.pkl', 'wb') as f:
                pickle.dump(tss, f)
                pickle.dump(tc, f)
                pickle.dump(dems[defo_n], f)
            out_file_n += 1
                    
    


#%%



def generate_dems_and_defos(dem_dict, n_pix = 224, min_deformation = 0.05, max_deformation = 0.25, n_def_location = 10):
    """ Given a large dem, take a random crop of it of size n_pix and randomly place a deformaiotn patter that is mostly visilbe (i.e. not in water)
    repeate n_def_location times.  
    
    Inputs: 
        dem_dict | dict | dict_keys(['name', 'centre', 'side_length', 'dem', 'lons_mg', 'lats_mg'])
        n_pix | int | size length of square images output.  
        min_deformation | float | deformation must be at least this big.  
        max_deformation | float | deformation must be less than this.  
        
    Returns:
        defos | r3 ma | masked array of deformaitons.  
        dems | r3 ma | masked array of dems
        
    History:
        2023_08_24 | MEG | Written.  
    """
    import numpy as np
    import numpy.ma as ma
    
    from syinterferopy.syinterferopy import coherence_mask
    

    defo_source = 'mogi'
    generate_limit = 20                                                                     # exit if we get stuck in while unable to make deformaiton viable.  
    n_generate = 0                                                                            # count how many ifgs succesful made so we can stop when we get to n_ifgs
    attempt_generate = 0                                                                              # only succesfully generated ifgs are counted above, but also useful to count all
    defos = ma.zeros((n_def_location, n_pix, n_pix))                                                                                        # the data will be stored in a dictionary
    dems =  ma.zeros((n_def_location, n_pix, n_pix))                                                                                        # the data will be stored in a dictionary
    
    
    while (n_generate < n_def_location) and (n_generate < generate_limit):
               
        dem_large = dem_dict['dem']                                                                                             # open a dem
        dem_ll_extent = [(dem_dict['lons_mg'][0,0],   dem_dict['lats_mg'][0,0] ),                                   # get lon lat of lower left corner
                         (dem_dict['lons_mg'][-1,-1], dem_dict['lats_mg'][-1,-1])]                                  # and upper right corner

        defo_m, source_kwargs = create_random_defo_m(dem_large, dem_dict['lons_mg'], dem_dict['lats_mg'],
                                                     dem_dict['centre'], defo_source,                                                        #  make a deformation signal with a size within the bounds set by min and max.  
                                                     min_deformation, max_deformation)                                                      # Note that is is made at the size of the large DEM (dem_large)
        mask_coherence = coherence_mask(dem_dict['lons_mg'][:n_pix,:n_pix], dem_dict['lats_mg'][:n_pix,:n_pix])                              # generate coherence mask, but at the number of pixels required for the ouput, and not hte size of the large dem
        defo_m, dem, viable_location, loc_list, masks = def_and_dem_translate(dem_large, defo_m, mask_coherence, threshold = 0.3,           # do the random crop of the dem and the defo pattern, so reducing the size to that desired.  
                                                                              n_pixs=n_pix, defo_fraction = 0.8)                            # and check that the majority of the deformation pattern isn't in an incoheret area, or in water.          
        defo_m_ma = ma.array(defo_m, mask = masks['coh_water'])                                                                         # mask water and incoherence.  
        
        if viable_location:
            print(f"Succesfully randomly placed the deformation signal.  ")
            defos[n_generate,:,:] = defo_m_ma
            dems[n_generate,:,:] = dem
            n_generate += 1
            
    return defos, dems




#%%

def create_random_synthetic_ifgs(volcanoes, defo_sources, n_ifgs, n_pix = 224, outputs = ['uuu'], intermediate_figure = False, 
                                 coh_threshold = 0.7, noise_method = 'fft', cov_coh_scale = 5000,  
                                 min_deformation = 0.05, max_deformation = 0.25, snr_threshold = 2.0,
                                 turb_aps_mean = 0.02, turb_aps_length = 5000, turb_aps_interpolation_threshold = 5e3,
                                 topo_aps_mean = 56.0, topo_aps_var = 2.0, deflation = False):
    """
    A function to generate n random synthetic interferograms at subaerial volcanoes in the Smithsonian database at SRTM3 resolution (ie.e. ~90m).  Different deformation
    sources are supported (no deformatin, point (Mogi), sill or dyke), topographically correlated and turbulent atmopsheric phase screens (APS) are added,
    and areas of incoherence are synthesisd.  The outputs are as rank 4 arrays with channels last (ie n_ifgs x ny x nx x 3 ), and can be in a variety of
    styles (e.g. unwrapped across 3 channels, of unwrapped in channels 1 and 2 and the dem in 3).  The paper Gaddes et al. (in prep) details this 
    in more detail.  
    
    General structure:
            open_dem                            - these are required for making coastline and a topo correlated APS
            coherence_mask                      - synthesise areas of incoherence.  
                atmosphere_turb                 - generates the spatially correlated noise which is used to create areas of incoherence.  
            create_random_defo_m                - creates the random source_kwargs (e.g. depth/opening) and checks signals are of correct magnitude
                deformation_wrapper             - prepare grids in meters etc. and project 3D deformation to satellite LOS
                    deformation_Mogi            - if deformation is Mogi, take source_wargs and make 3d surfaced deformation
                    deformation_eq_dyke_sill    - if an Okada dislocation, take source_wargs and make 3d surfaced deformation
            def_and_dem_translate               - try random locations of the deformation signals and see if on land an in a coherent area.  
            atmosphere_turb                     - generate a turbulent APS
            atmosphere_topo                     - generate a topo correlated APS
            check_def_visible                   - check that signal to noise ratio (SNR) is acceptable and deformation pattern hasn't  dissapeared.  
            combine_signals                     - combine signals and return in different 3 channel formats (ie for use with AlexNet etc.  )
    
    Inputs:
        volcanoes | list of dicts | each volcano is a dictionary in the list, and contains various keys and values.  
                                    'dem': the dem 'lons_mg' : longitude of each pixel  (ie a meshgrid) 'lats_mg' : latitude of each pixel
        defo_sources | list | defo sources to be synthesised.  e.g. ['no_def', 'dyke', 'sill', 'mogi']
        n_ifgs | int | the number of interferogram to generate.  
        n_pix | int | Interferograms are square, with side length of this many pixels.  Note that we use SRTM3 pixels, so squares of ~90m side length.  
        intermediate_figure | boolean | If True, a figure showing the search for a viable deformatin location and SNR is shown.  
        coh_threshold | float | coherence is in range of 0-1, values above this are classed as incoherent
        noise_method | string | fft or cov.  fft is ~x100 faster, but you can't set teh length scale.  
        cov_coh_scale | float | sets spatial scale of incoherent areas.  Only required if 'noise_method' is cov
        min_deformation | float | Deformation must be above this size (in metres), even before checking the SNR agains the deformation and the atmosphere.  
        max_deformation | float | Deformation must be below this size (in metres), even before checking the SNR agains the deformation and the atmosphere.  
        snr_threshold | float | SNR of the deformation vs (topographically correlated APS + turbulent APS) must be above this for the signals to be considered as visible.  
        turb_aps_mean | float | mean strength of turbulent atmospheres, in metres.  Note the the atmosphere_turb funtion takes cmm, and the value provided in m is converted first
        turb_aps_length | float | Length scale of spatial correlatin, in metres. e.g. 5000m
        turb_aps_interpolation_threshold | int | If n_pix is larger than this, interpolation will be used to generate the extra resolution (as the spatially correlated noise function used here is very slow for large images).  Similar to the setting coh_interpolation_threshold
        topo_aps_mean | float | rad/km of delay for the topographically correlated APS
        topo_aps_var | float | rad/km.  Sets the strength difference between topographically correlated APSs
        deflation | boolean | if True, the sills and Mogi sources can be deflating (closing/ negative volume change.  )
    Returns:
        X_all | dict of masked arrays | keys are formats (e.g. uuu), then rank 4 masked array
        Y_class | rank 2 array | class labels, n x 1 (ie not one hot encoding)
        Y_loc | rank 2 array |  location of deformaiton, nx4 (xy location, xy width)
        Y_source_kwargs | list of dicts | stores the source_kwargs that were generated randomly to create each interferogram.  Also contains the source names (ie the same as Y_class, but as a string).  
    History:  
        2020/10/19 | MEG | Written from various scripts.  
        2020/10/26 | MEG | Add funtion to record the source_kwargs.  Intended for use if these are to be the label of interest (e.g. trianing a CNN to determine strike etc.  )
        2021_08_24 | MEG | Add option to set whether deflating sills and Mogi sources are allowed.  
    """
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    
    from syinterferopy.syinterferopy import coherence_mask, atmosphere_turb, atmosphere_topo
    from syinterferopy.aux import truncate_colormap                                                             # needed to plot the DEM with nice (terrain) colours
    
    # hard coded variables:
    count_max = 8                                                                                     # the number of times the function searches for acceptable deformation positions and SNR
        
    # begin to generate the data for this output file    
    succesful_generate = 0                                                                            # count how many ifgs succesful made so we can stop when we get to n_ifgs
    attempt_generate = 0                                                                              # only succesfully generated ifgs are counted above, but also useful to count all
    X_all = {}                                                                                        # the data will be stored in a dictionary, X_all
    for output in outputs:
        X_all[output]  = ma.zeros((n_ifgs, n_pix, n_pix, len(output)))                                # populate the dictionary with the required outputs (as keys)  and empty arrays (as values)
    Y_class = np.zeros((n_ifgs,1))                                                                    # initate for labels showing type of deformation
    Y_loc = np.zeros((n_ifgs,4))                                                                      # initate for labels showing location of deformation
    Y_source_kwargs = []                                                                              # initate an empty list for storing source kwargs (parameters like opening dip etc.) 
    
    while succesful_generate < n_ifgs:
        volcano_n = np.random.randint(0, len(volcanoes))                                                # choose a volcano at random
        defo_source = defo_sources[np.random.randint(0, len(defo_sources))]                             # random choice of which deformation source to use
        print(f"Volcano: {volcanoes[volcano_n]['name']} ", end = '')
               
        # 0: generate incoherence mask, choose dem choose if ascending or descending.  
        dem_large = volcanoes[volcano_n]['dem']                                                                                             # open a dem
        dem_ll_extent = [(volcanoes[volcano_n]['lons_mg'][0,0],   volcanoes[volcano_n]['lats_mg'][0,0] ),                                   # get lon lat of lower left corner
                         (volcanoes[volcano_n]['lons_mg'][-1,-1], volcanoes[volcano_n]['lats_mg'][-1,-1])]                                  # and upper right corner
        #import pdb; pdb.set_trace()
        mask_coherence = coherence_mask(volcanoes[volcano_n]['lons_mg'][:n_pix,:n_pix], volcanoes[volcano_n]['lats_mg'][:n_pix,:n_pix])     # generate coherence mask, but at the number of pixels required for the ouput, and not hte size of the large dem
                                                                                                                                            # if threshold is 0, all of the pixels are incoherent , and if 1, none are.  
        
        print(f"| Coherence mask generated ", end = '')
        if np.random.rand() < 0.5:
            asc_or_desc = 'asc'
            heading = 348
        else:
            asc_or_desc = 'desc'
            heading = 192
        print(f"| Deformation source: {defo_source} ", end = '')
        
        if intermediate_figure:
            f, axes = plt.subplots(3, count_max, figsize = (16, 8))                                         
            f.suptitle(f"{attempt_generate}: Volcano: {volcanoes[volcano_n]['name']} | Deformation label: {defo_source}")
            f.canvas.set_window_title(f"{attempt_generate}_Volcano:{volcanoes[volcano_n]['name']}")
            axes[0,0].set_ylabel('Location search \n (hatch = water or incoherent)')
            axes[1,0].set_ylabel('SNR search \n (defo + APS_turb + APT_topo')
            for axe_n, axe in enumerate(axes[0,:]):
                axe.set_title(f"Attempt # {axe_n}")
            cmap = plt.get_cmap('terrain')                                                          # makes sense for DEMs
            cmap = truncate_colormap(cmap, 0.2, 1)                                                  # but by deafult it starts at blue, so crop (truncate) that part off.  
            axes[2,0].imshow(dem_large, cmap = cmap)
            axes[2,0].set_xlabel('Full DEM')
            axes[2,1].imshow(mask_coherence)
            axes[2,1].set_xlabel('Coherence Mask')
            axes[2,1].yaxis.tick_right()
            for axe in np.concatenate((axes[0,1:], axes[1,:], axes[2,2:]), axis = 0):
                axe.set_axis_off()
            
        # 1: If no deformation, just generate topo. correlated and turbulent APS
        if defo_source == 'no_def':
            viable_location = viable_snr = True                                                                                                        # in the no deformation case, these are always True
            source_kwargs = {'source' : 'no_def'}
            defo_m = np.ones(dem_large.shape)                                                                                                          # in the no deformation case, make a deformaiton that is just zeros.  
            defo_m, dem, viable_location, loc_list, masks = def_and_dem_translate(dem_large, defo_m , mask_coherence, threshold = 0.3, 
                                                                                  n_pixs=n_pix, defo_fraction = 0.8)                                   # doesn't matter if this returns false.  Note that masks is a dictionary of deformation, coherence and water, and water
            dem = ma.array(dem, mask = masks['coh_water'])                                                                                             # mask the DEM for water and incoherence
            APS_turb_m = atmosphere_turb(1, volcanoes[volcano_n]['lons_mg'][:n_pix,:n_pix], volcanoes[volcano_n]['lats_mg'][:n_pix,:n_pix],            # generate a turbulent APS, but for speed not at the size of the original DEM, and instead at the correct n_pixs
                                         mean_m = turb_aps_mean)
            APS_turb_m = APS_turb_m[0,]                                                                                                                 # remove the 1st dimension      
            APS_topo_m = atmosphere_topo(dem, topo_aps_mean, topo_aps_var, difference=True)                                                             # generate a topographically correlated APS                                                                                                                                # helps to split up and clarify terminal output.  
            if intermediate_figure:
                axes[0,0].imshow(np.zeros((n_pix, n_pix)))
                axes[0,0].set_xlabel(f"No deformation")
                axes[0,0].set_axis_on()
                temp_combined = APS_topo_m + APS_turb_m
                axes[1,0].imshow(temp_combined)
                axes[1,0].set_xlabel(f"[{np.round(np.min(temp_combined), 2)}, {np.round(np.max(temp_combined),2 )}] m")
                axes[1,0].set_axis_on()
                plt.pause(4)
            
        # 2: Or, if we do have deformation, generate it and topo correlated and turbulent APS    
        else:                                                                                                                    
            viable_location = False; count = 0;                                                                     # prepare for while statement that will search for a viable deformation location.  
            # 2a: Try to make a deformation signal of the correct magnitude, and then place it on land.       
            while viable_location is False and count < count_max:                                                                                   # random translations of dem and deformation, but deformation must remain visible.  
                # print("QUICK FIX - DEFO ALWAYS SILL")    
                # defo_source = 'sill'
                defo_m, source_kwargs = create_random_defo_m(dem_large, volcanoes[volcano_n]['lons_mg'], volcanoes[volcano_n]['lats_mg'],
                                                             volcanoes[volcano_n]['centre'], defo_source,                                           #  make a deformation signal with a size within the bounds set by min and max.  
                                                             min_deformation, max_deformation, asc_or_desc,                                         # Note that is is made at the size of the large DEM (dem_large)
                                                             deflation = deflation)
                source_kwargs['source'] = defo_source                                                                                               # add name of source to dict of source_kwargs (e.g. depth/opening etc.  )
                defo_m, dem, viable_location, loc_list, masks = def_and_dem_translate(dem_large, defo_m, mask_coherence, threshold = 0.3,           # do the random crop of the dem and the defo pattern, so reducing the size to that desired.  
                                                                                      n_pixs=n_pix, defo_fraction = 0.8)                            # and check that the majority of the deformation pattern isn't in an incoheret area, or in water.  
                dem = ma.array(dem, mask = masks['coh_water'])                                                                                      # mask the DEM (for water and areas of incoherence)
                if intermediate_figure:
                    axes[0,count].imshow(ma.array(defo_m, mask = masks['coh_water']), vmin = np.min(defo_m), vmax = np.max(defo_m))
                    axes[0,count].imshow(ma.array(defo_m, mask = 1- masks['coh_water']), vmin = np.min(defo_m), vmax = np.max(defo_m))
                    axes[0,count].contourf(masks['coh_water'], 1, hatches=['', '//'], alpha=0)
                    axes[0,count].set_xlabel(f"Viable Location: {viable_location}")
                    axes[0,count].set_axis_on()
                    plt.pause(1)
                if viable_location == False:
                    count += 1
                
            if viable_location:                                                                                                          # 
                # 2b: If we have a viable size and location, try to make the atmospheric signals and check that deforamtion is still visible (ie acceptable signal to noise ratio)
                print(f"| Viable location ", end = '')
                viable_snr = False; count = 0                                                                       # make dem and ph_def and check that def is visible
                while viable_snr is False and count < count_max:
                    APS_turb_m = atmosphere_turb(1, volcanoes[volcano_n]['lons_mg'][:n_pix,:n_pix], volcanoes[volcano_n]['lats_mg'][:n_pix,:n_pix],            # generate a turbulent APS, but for speed not at the size of the original DEM, and instead at the correct n_pixs
                                                 mean_m = turb_aps_mean)
                    APS_turb_m = APS_turb_m[0,]                                                                                                       # remove the 1st dimension      
                    APS_topo_m = atmosphere_topo(dem, topo_aps_mean, topo_aps_var, difference=True)                                                   # generate a topographically correlated APS using the DEM
                    viable_snr, snr = check_def_visible(defo_m, masks['def'], APS_topo_m, APS_turb_m, snr_threshold)                                  # check that the deformation is visible over the ph_topo and ph_trub (SNR has to be above snr_threshold)
                    
                    if intermediate_figure:
                        temp_combined = defo_m + APS_topo_m + APS_turb_m
                        axes[1,count].imshow(temp_combined)
                        axes[1,count].set_xlabel(f"[{np.round(np.min(temp_combined), 2)}, {np.round(np.max(temp_combined),2 )}] m \n"
                                                 f"SNR: {np.round(snr, 2)}")
                        axes[1,count].set_axis_on()
                        plt.pause(1)
                    if viable_snr == False:
                        count +=1
                if viable_snr:
                    print(f"| Viable SNR ", end = '')
                else:
                    print('| SNR is too low. \n')
                plt.pause(2)
            else:
                print(f"| No viable location found. \n")
            
        # 3: If succesful, append to X (data) and Y (labels) arrays.                                                                                    # still in the main while loop, but out of the deformation / no deformation else statement.
        if (viable_location and viable_snr) or (defo_source == 'no_def'):
            X_all, Y_class, Y_loc, succesful = combine_signals(X_all, Y_class, Y_loc, defo_m, APS_topo_m, APS_turb_m,                                   # combine the various signals into different data forms (e.g. unwrapped or wrapped)
                                                               heading, dem, defo_source, defo_sources, loc_list, outputs, succesful_generate)         # (we have a succesful flag as sometimes this can fail due to Nans etc.  )
            #import pdb; pdb.set_trace()
            if succesful:
                Y_source_kwargs.append(source_kwargs)
                succesful_generate += 1                                                                                                                 # updat the countery of how many ifgs have been succesfuly made
                print(f"| Succesful write. \n")
            else:
                print(f"| Failed write.  \n")
        attempt_generate += 1                                                                                                                           # update the counter of how many have been made in total (successes and failures)
        plt.close()
            
    return X_all, Y_class, Y_loc, Y_source_kwargs


#%%

def combine_signals(X_all, Y_class, Y_loc, defo_m, APS_topo_m, APS_turb_m,
                    heading, dem, defo_source, defo_sources, loc_list, outputs,
                    succesful_generate, sar_speckle_strength = 0.05):
    """ Given the synthetic outputs and labels (X and Y) and the parts of the synthetic data, combine into different formats and write to dictionary (X_all)
    Inputs:
        X_all | dict of masked arrays | keys are formats (e.g. uuu), then rank 4 masked array
        Y_class | rank 2 array | class labels, n x 1 (ie not one hot encoding)
        Y_loc | rank 2 array |  location of deformaiton, nx4 (xy location, xy width)
        defo_m | rank 2 array | deformation in metres, not masked
        APS_topo_m | rank 2 array | topographically correlated APS, incoherence and water masked out.  
        APS_turb_m | rank 2 array | tubulent APS, not masked
        heading | float | in degrees.  e.g. 192 or 012
        dem | rank 2 masked array | the  DEM.  Needed to make radar amplitude.  
        defo_source_n | int | 0 = no def, 1 = dyke, 2 = sill, 3 = Mogi.  No 0's should be passed to this as it makes no deformatio nsignals.  
        loc_list | list of tuples | xy of centre of location box, and xy width.  e.g [(186, 162), (69, 75)]
        outputs | list of strings | e.g. ['uuu', 'uud']
        succesful_generate | int | which number within a file we've generated so far
        sar_speckle_strength | float | strength (variance) of gaussain speckled noise added to SAR real and imaginary
    Returns:
        X_all | as above, but updated
        Y_class  | as above, but updated
        Y_loc  | as above, but updated
        succesful | boolean | True if no nans are present.  False if nans are
    History:
        2020/08/20 | MEG | Written from exisiting scripts.  
        2020/10/19 | MEG | Fix bug that nans_present was being returned, instead of succesful (and they are generally the opposite of each other)
    """
    import numpy as np
    import numpy.ma as ma
    from matplotlib.colors import LightSource                                                                          # used to make a synthetic Radar return.  
    
    def normalise_m1_1(r2_array):
        """ Rescale a rank 2 array so that it lies within the range[-1, 1]
        """
        import numpy as np
        r2_array = r2_array - np.min(r2_array)
        r2_array = 2 * (r2_array/np.max(r2_array))
        r2_array -= 1
        return r2_array
    
    s1_wav = 0.056                                                                                                      # Sentinel-1 wavelength in m
    incidence = 30                                                                                                      # ditto - 30 roughly right for S1
    
    # 1 Create ph_all and classes and locations, depending on if we want deformation or not.  
    if defo_source == 'no_def':
        ph_all = ((4*np.pi)/s1_wav)*(APS_topo_m + APS_turb_m)                                                           # comnbine the signals in m, then convert to rads
        Y_loc[succesful_generate,:] = np.array([0,0,0,0])                                                               # 0 0 0 0 for no deformaiton
    else:
        ph_all = ((4*np.pi)/s1_wav) * (APS_topo_m + APS_turb_m + defo_m)                                                 # combine the signals in m, then convert to rads.  Here we include deformation.  
        Y_loc[succesful_generate,:] = np.array([loc_list[0][0],loc_list[0][1],loc_list[1][0],loc_list[1][1]])            # location of deformation 
    Y_class[succesful_generate,0] = defo_sources.index(defo_source)                                                      # write the label as a number
    ph_all_wrap = (ph_all + np.pi) % (2 * np.pi ) - np.pi                                                                # wrap

    #1 Genreate SAR amplitude 
    look_az = heading - 90     
    look_in = 90 - incidence                                                 # 
    ls = LightSource(azdeg=look_az, altdeg=look_in)
    sar_amplitude = normalise_m1_1(ls.hillshade(dem))                                                                   # in range [-1, 1]

    # make real and imaginary
    ifg_real = sar_amplitude * np.cos(ph_all_wrap)
    ifg_imaginary = sar_amplitude * np.sin(ph_all_wrap)
    ifg_real = ifg_real + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])                   # add noise to real
    ifg_imaginary = ifg_imaginary + sar_speckle_strength * np.random.randn(ifg_real.shape[0], ifg_real.shape[1])         # and imaginary

    # make things rank 4
    unwrapped = ma.expand_dims(ma.expand_dims(ph_all, axis = 0), axis = 3)
    dem = ma.expand_dims(ma.expand_dims(dem, axis = 0), axis = 3)
    ifg_real = ma.expand_dims(ma.expand_dims(ifg_real, axis = 0), axis = 3)
    ifg_imaginary = ma.expand_dims(ma.expand_dims(ifg_imaginary, axis = 0), axis = 3)
    wrapped = ma.expand_dims(ma.expand_dims(ph_all_wrap, axis = 0), axis = 3)
    
    # Check for Nans
    nans_present = False                                                                                                   #initiate
    for signal in [unwrapped, dem, ifg_real, ifg_imaginary, wrapped]:
        nans_present = np.logical_or(nans_present, np.max(np.isnan(signal).astype(int)).astype(bool))                     # check if nans in current signal, and update succesful.  
    if nans_present:
        succesful = False
        print(f"| Failed due to Nans ", end = '')
    else:
        succesful = True
        for output in outputs:
            if output == 'uuu':
                X_all[output][succesful_generate,] = ma.concatenate((unwrapped, unwrapped, unwrapped), axis = 3)                             # uuu
            elif output == 'uud':
                X_all[output][succesful_generate,] = ma.concatenate((unwrapped, unwrapped, dem), axis = 3)                                # uud
            elif output == 'rid':
                X_all[output][succesful_generate,] = ma.concatenate((ifg_real, ifg_imaginary, dem), axis = 3)                     # rid
            elif output == 'www':
                X_all[output][succesful_generate,] = ma.concatenate((wrapped, wrapped, wrapped), axis = 3)              # www
            elif output == 'wwd':
                X_all[output][succesful_generate,] = ma.concatenate((wrapped, wrapped, dem), axis = 3)                      #wwd
            elif output == 'ud':
                X_all[output][succesful_generate,] = ma.concatenate((unwrapped, dem), axis = 3)                                # uud
            else:
                raise Exception("Error in output format.  Should only be either 'uuu', 'uud', 'rid', 'www', or 'wwd'.  Exiting.  ")

    return X_all, Y_class, Y_loc, succesful

      
    


    


#%%

def check_def_visible(ph_def, mask_def, ph_topo, ph_turb, snr_threshold = 2.0, debugging_plot = False):
    """A function to check if a (synthetic) deformation pattern is still visible
    over synthetic topo correlated and turbulent atmospheres.  
    
    Inputs:
        ph_def | r2 array | deformation phase
        mask_def | rank 2 array of ints | maks showing where the deformation is - 1s where deforming
        ph_topo | r2 array | topo correlated APS
        ph_turb | r2 array | turbulent APS
        snr_threshold | float | sets the level at which deformation is not considered visible over topo and turb APS
                                bigger = more likely to accept, smaller = less (if 0, will never accept)
        debugging_plot | boolean | if True, a figure is produced to help interepret the correct SNR.  Could give problems with dependencies.  
    Returns:
        viable | boolean | True if SNR value is acceptable.  
        snr | float | SNR
    History:
        2019/MM/DD | MEG | Written as part of f01_real_image_dem_data_vXX.py
        2019/11/06 | MEG | Extracted from sctipt and placed in synth_ts.py
        2020/08/19 | MEG | WIP
    
    """
    import numpy as np
    import numpy.ma as ma  
       
    ph_def = ma.array(ph_def, mask = (1-mask_def))
    ph_atm =  ma.array((ph_turb + ph_topo), mask = (1-mask_def))
    snr = np.var(ma.compressed(ph_def)) / np.var(ma.compressed(ph_atm))
    
    if snr > snr_threshold:
        viable = True
    else:
        viable = False
        
    # import matplotlib.pyplot as plt
    # from small_plot_functions import create_universal_cmap
    # cmap, vmin, vmax = create_universal_cmap([ph_def, ph_atm])
    # f, axes = plt.subplots(1,2)
    # f.suptitle(f"SNR: {snr}")
    # axes[0].imshow(ph_def, vmin = vmin, vmax = vmax, cmap = cmap)
    # axes[1].imshow(ph_atm, vmin = vmin, vmax = vmax, cmap = cmap)
    return viable, snr


#%%

def random_crop_of_r2(r2_array, n_pixs=224, extend = False):
    """  Randomly select a subregion of a rank 2 array.  If the array is quite small, we can select areas
    of size n_pixs that go over the edge by padding the edge of the array first.  Works well with things like deformation 
    (which is centered in the r2_array so the near 0 edges can be interpolated), but poorly with things like a dem.  
    Inputs:
        r2 array | rank 2 array | e.g. a dem or deformation pattern of size 324x324
        n_pixs | int | side lenght of subregion.  E.g. 224 
        extend | boolean | if True, the padding described above is carried out.  
    Returns:
        r2_array_subregion ||
        pos_xy           
    History:
        2019/03/20 | MEG | Update to also output the xy coords of where the centre of the old scene now is
        2020/08/25 | MEG | Change so that masks are bundled together into a dictionary
        2020/09/25 | MEG | 
        2020/10/09 | MEG | Update docs
    """
    import numpy as np
    
    ny_r2, nx_r2 = r2_array.shape
    if extend:                                                                                  # sometimes we select to extend the image before cropping.  This only works if the edges of the signal are pretty much constant already.
        r2_array = np.pad(r2_array, int(((3*n_pixs)-ny_r2)/2), mode = 'edge')                   # pad teh edges, so our crop can go off them a bit (helps to get signals near the middle out to the edge of the cropped region)

    centre_x = int(r2_array.shape[1]/2)        
    centre_y = int(r2_array.shape[0]/2)        

    x_start = np.random.randint(centre_x - (0.9*n_pixs), centre_x - (0.1*n_pixs))           # x_start always includes hte centre, as it can only go 90% of the scene width below the centre, and at max 10% of the scene width below the cente
    y_start = np.random.randint(centre_y - (0.9*n_pixs), centre_y - (0.1*n_pixs))           # note that this could be done more efficiently, but it is done this way to ensure that the centre is always in the crop (as this is where a deformation signal will be in SyInteferoPy)

    r2_array_subregion = r2_array[y_start:(y_start+n_pixs), x_start:(x_start+n_pixs)]           # do the crop
    x_pos = np.ceil((r2_array.shape[1]/2) - x_start)                                                     # centre of def - offset
    y_pos = np.ceil((r2_array.shape[0]/2) - y_start)                                                     # 
    pos_xy = (int(x_pos), int(y_pos))                                                                       # ints as we're working with pixel numbers
    return r2_array_subregion, pos_xy



#%%

def def_and_dem_translate(dem_large, defo_m, mask_coh, threshold = 0.3, n_pixs=224, defo_fraction = 0.8):
    """  A function to take a dem, defo source and coherence mask, randomly traslate the defo and dem, 
    then put together and check if the deformation is still visible.  
    Inputs:
        dem_large | rank 2 masked array | height x width , the dem
        defo_m | rank 2 array | height x width, the deformation signals, projected into the satellite LOS (either ascending or descending)
        n_pixs | int | output size in pixels.  Will be square
        threshold | decimal | e.g. if 0.2, and max abs deforamtion is 10, anything above 2 will be considered deformation.  
        defo_fraction | decimal | if this fraction of deformation is not in masked area, defo, coh mask, and water mask are deemed compatible
        
    Returns:
        defo_m_crop
        dem
        viable
        loc_list
        masks | dict | contains water, coh_water and deformation masks
        
    History:
        2019/MM/DD | MEG | Written as part of f01_real_image_dem_data_vXX.py
        2019/11/06 | MEG | Extracted from sctipt and placed in synth_ts.py
        
                
        """
    
    import numpy as np
    import numpy.ma as ma
    from syinterferopy.aux import localise_data
    
    
    # start the function
    dem, _ = random_crop_of_r2(dem_large, n_pixs, extend = False)                          # random crop of the dem, note that extend is False as we can't extend a DEM (as we don't know what topography is)
    defo_m_crop, def_xy = random_crop_of_r2(defo_m, n_pixs, extend = True)                 # random crop of the deformation, note that extend is True as deformation signal is ~0 at edges, so no trouble to interpolate it
    mask_water = ma.getmask(dem).astype(int)                                               # convert the boolean mask to a more useful binary mask, 0 for visible, 1 for masked.  

    try:
        loc_list = localise_data(defo_m_crop, centre = def_xy)                                 # xy coords of deformation for localisation label (when training CNNs etc)
        viable = True
    except:
        loc_list = []                                                                       # return empty so that function can still return variable
        viable = False
            
    # determine if the deformation is visible
    mask_def = np.where(np.abs(defo_m_crop) > (threshold * np.max(np.abs(defo_m_crop))), np.ones((n_pixs, n_pixs)), np.zeros((n_pixs, n_pixs)))     # anything below the deformation threshold is masked.  
    mask_coh_water = np.maximum(mask_water, mask_coh)                                                                                       # combine incoherent and water areas to one mask   
    ratio_seen = 1 - ma.mean(ma.array(mask_coh_water, mask = 1-mask_def))                                                                   # create masked array of mask only visible where deforamtion is, then get mean of that area
    #defo_m_crop = ma.array(defo_m_crop, mask = mask_coh_water)                                                                                     # mask out where we won't have radar return
    if ratio_seen < defo_fraction:
        viable = False                                                      # update viable is now not viable (as the deformation is not visible)
    masks = {'coh_water' : mask_coh_water,
             'water'     : mask_water,
             'def'       : mask_def}
    return defo_m_crop, dem, viable, loc_list, masks


#%%

def create_random_defo_m(dem, lons_mg, lats_mg, deformation_ll, defo_source,
                         min_deformation_size = 0.05, max_deformation_size = 0.25, 
                         asc_or_desc = 'random', count_readjust_threshold = 20,
                         deflation = False):
    """ Given a dem, create a random deformation pattern of acceptable magnitude.  The DEM is only required for geocoding the deformation patterns.  
    Projected into either ascending or descending satellite LOS.  
    Note that if min and max are changed significantly bu the soure_kwargs not, the function will be not be able to make a deformation
    that satisfies them (e.g. if the min deformaiton is a meter but the dyke opening is small), and will get stuck in the while condition.  
    Inputs:
        dem | rank 2 masked array |  The DEM, as a masked array.  
        lons_mg | rank 2 array | longitudes of the bottom left corner of each pixel.  
        lats_mg | rank 2 array | latitudes of the bottom left corner of each pixel.  
        deformation_ll | tuple | lon lat of deformation source.  
        defo_source_n | string | either dyke, sill or Mogi source.  
        min_deformation_size | float | in metres, deformation must be at least this large.  
        max_deformation_size | float | in metres, deformation must be at least this large.  
        asc_or_dec | string | 'asc' or 'desc' or 'random'.  If set to 'random', 50% chance of each.  
        count_readjust_threshold | float | after this many attempts at making a singal, the kwargs are adjusted by the factor above
        deflation | boolean | if True, the sills and Mogi sources can be deflating (closing/ negative volume change.  )
        
    Returns:
        defo_m | rank 2 masked array | displacment in satellite LOS at each location, with the same mask as the dem.  
        source_kwargs | dict | dictionary of the source_kwargs that were generated randomly.  
    History:
        2020/08/12 | MEG | Written
        2020/08/25 | MEG | Update from identifying sources by number to identifying by name.  
        2020/09/24 | MEG | Comment and write docs.  
        2020/10/09 | MEG | Update bug (deformation_wrapper was returning masked arrays when it should have returned arrays)
        2020/10/26 | MEG | Return source_kwargs for potential use as labels when perofming (semi)supervised learning
        2021_05_06 | MEG | Add a catch incase defo_source doesn't match dyke/sill/mogi (including wrong case).  
        2021_06_16 | MEG | Change so that sills and mogi source can be deflation.  
        021_08_24 | MEG | Add argument that controls whether deflation of sills and Mogi sources is included.  
    """
    import numpy as np
    import numpy.ma as ma
    from syinterferopy.syinterferopy import deformation_wrapper
    import random
    
    deformation_magnitude_acceptable = False                                                                   # initiate
    source_kwargs_scaling = 1                                                                                  # when set to 1, the source kwargs aren't changed.  When >1, they are increased (increasing deformation signal), and the opposite for < 1
    count = 0                                                                                                  # to keep track of the number of attempts at making a signal of the correct magnitude.  
    defo_magnitudes = []                                                                                       # initiate a list that will record the signal strength of each attempt at generating a signal.  
    
    while (deformation_magnitude_acceptable is False):                                                          # keep trying to make deformation patterns of acceptable magnitude.  
        # 0: Set the parameters randomly for the forward model of the source
        if defo_source == 'dyke':                                                       
            source_kwargs = {'strike'    : np.random.randint(0, 359),                                           # in degrees
                             'top_depth' : 2000 * np.random.rand(),                                             # in metres
                             'length'    : 10000 * np.random.rand(),                                            # in metres
                             'dip'       : np.random.randint(75, 90),                                           # in degrees
                             'opening'   : 0.1 + 0.6 * np.random.rand()}                                        # in metres.  
            source_kwargs['bottom_depth'] = source_kwargs['top_depth'] + 6000 * np.random.rand()                # bottom depth is top depth plus a certain random amount
        elif defo_source == 'sill':
            source_kwargs = {'strike'   : np.random.randint(0, 359),                                            # in degrees
                             'depth'    : 1500 + 2000 * np.random.rand(),                                       # in metres
                             'width'    : 2000 + 4000 * np.random.rand(),                                       # in metres
                             'length'   : 2000 + 4000 * np.random.rand(),                                       # in meters
                             'dip'      : np.random.randint(0,5),                                               # in degrees
                             'opening'  : (0.2 + 0.8 * np.random.rand())}                                       # in metres
            if deflation:
                source_kwargs['opening'] *=  random.choice([-1, 1])                                             # if deflation is allowed, 50% of the time switch the sign of the volume change.  

        elif defo_source == 'mogi':
            source_kwargs = {'volume_change' : (int(2e6 + 1e6 * np.random.rand())),                             # in metres, always positive
                             'depth'         : 1000 + 3000 * np.random.rand()}                                  # in metres
            if deflation:
                source_kwargs['volume_change'] *=  random.choice([-1, 1])                                       # if deflation is allowed, 50% of the time switch the sign of the volume change.  
        else:
            raise Exception(f"defo_source should be either 'mogi', 'sill', or 'dyke', but is {defo_source}.  Exiting.")
                    
        # 1: Apply the scaling factor to the source_kwargs                                                      # (ie if we're never getting a signal of acceptable size, adjust some kwargs to increase/decrease the magnitude.  )
        adjustable_source_kwargs = ['opening', 'volume_change']
        for source_kwarg in source_kwargs:                                                                      # loop through all the settings (kwargs) for that source
            if source_kwarg in adjustable_source_kwargs:                                                        # if the one we're currently on is in the list of ones that can be adjusted
                source_kwargs[source_kwarg] *= source_kwargs_scaling                                            # adjust it
        
        # 2:  Generate the deformation
        defo_m, _, _, _ = deformation_wrapper(lons_mg, lats_mg, deformation_ll, defo_source,
                                              dem = None, asc_or_desc = asc_or_desc,  **source_kwargs)     # create the deformation pattern, using the settings we just generated randomly.  
           
        # 3: Check that it is of acceptable size (ie not a tiny signal, and not a massive signal).  
        defo_magnitudes.append(np.max(np.abs(ma.compressed(defo_m))))                                                       # get the maximum absolute signal (ie we don't care if its up or down).  
        
        if (min_deformation_size < defo_magnitudes[-1]) and (defo_magnitudes[-1] < max_deformation_size):                   # check if it's in the range of acceptable magnitudes.  
            deformation_magnitude_acceptable = True                                                                         # if  it is, update the boolean flag to leave the while statement.  
            # import matplotlib.pyplot as plt
            # f, ax = plt.subplots(1)
            # ax.imshow(defo_m)
        
        if count == count_readjust_threshold:                                                                                   # if we get to the count threshold, we look at adjusting the source_kwargs.  
            print(f"After {count} unsuccessful attempts at making a signal of the correct magnitude, the mean signal is "
                  f"{np.mean(defo_magnitudes)}, so either the 'opening' or 'volume_change' will be adjusted accordingly.  "
                  f"Note that you should probably change the 'min_deformation_size' or 'max_deformation_size', or change the "
                  f"source_kwargs create_random_defo to give a signal that fits within the min and max constraints.  "
                  f"However, for now we can conitnue and try to adjust the soure_kwargs to get a signal that fits within the min/max constraints.  ")
            if np.min(defo_magnitudes) > max_deformation_size:                                                               # if the minimum signal was still too big
                source_kwargs_scaling -= 0.2                                                                                 # all the kwargs will be reduced.
            else:
                source_kwargs_scaling += 0.2                                                                                # and in the opppostie case, increased.  
            defo_magnitudes = []                                                                                            # clear ready for the next attempt at making a signal of the correct size.  
            count = 0                                                                                                       # and reset the counter.  
        else:
            count += 1    

    return defo_m, source_kwargs
            

#%%




