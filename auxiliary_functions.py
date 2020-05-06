#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:14:05 2020

@author: matthew
"""



#%%

def matrix_show(matrix, title='', ax = None, fig = None, db = False):
    """Visualise a matrix 
    Inputs:
        matrix | r2 array or masked array
        title | string
        ax | matplotlib axes
        db | boolean | bug fix for Spyder debugging.  If True, will allow figure to show when 
                        debugging
    
    2017/10/18 | update so can be passed an axes and plotted in an existing figure
    2017/11/13 | fix bug in how colorbars are plotted.  
    2017/12/01 | fix bug if fig is not None
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots()
    matrix = np.atleast_2d(matrix)                   # make at least 2d so can plot column/row vectors
    
    if isinstance(matrix[0,0], np.bool_):           # boolean arrays will plot, but mess up the colourbar
        matrix = matrix.astype(int)                 # so convert
    
    matrixPlt = ax.imshow(matrix,interpolation='none', aspect='auto')
    fig.colorbar(matrixPlt,ax=ax)
    ax.set_title(title)
    if db:
        plt.pause(1)


#%%
        
def signal_atmosphere_topo(dem_m, strength_mean = 56.0, strength_var = 2.0, difference = False):
    """ Given a dem, return a topographically correlated APS, either for a single acquistion
    or for an interferometric pair.
    Inputs:
        dem_m | r4 ma | rank4 masked array, with water masked. Units = metres!
        strength_mean | float | rad/km of delay.  default is 56.0, taken from Fig5 Pinel 2011 (Statovolcanoes...)
        strength_var  | float | variance of rad/km delay.  Default is 2.0, which gives values similar to Fig 5 of above.
        difference | boolean | if False, returns for one acquisitin.  If true, returns for an interferometric pair (ie difference of two acquisitions)

    Outputs:
        ph_topo | r4 ma | topo correlated delay in m.  UNITS ARE M

    2019/09/11 | MEG | written.
    """
    import numpy as np
    import numpy.ma as ma

    envisat_lambda = 0.056                       #envisat/S1 wavelength in m
    dem = 0.001 * dem_m                        # convert from metres to km

    if difference is False:
        ph_topo = (strength_mean + strength_var * np.random.randn(1)) * dem
    elif difference is True:
        ph_topo_aq1 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # this is the delay for one acquisition
        ph_topo_aq2 = (strength_mean + strength_var * np.random.randn(1)) * dem                         # and for another
        ph_topo = ph_topo_aq1 - ph_topo_aq2                                                             # interferogram is the difference, still in rad



    else:
        print("'difference' must be either True or False.  Exiting...")
        import sys; sys.exit()

    # convert from rad to m
    ph_topo_m = (ph_topo / (4*np.pi)) * envisat_lambda                               # delay/elevation ratio is taken from a paper (pinel 2011) using Envisat data


    if np.max(ph_topo_m) < 0:                                                       # ensure that it always start from 0, either increasing or decreasing
        ph_topo_m -= np.max(ph_topo_m)
    else:
        ph_topo_m -= np.min(ph_topo_m)

    ph_topo_m = ma.array(ph_topo_m, mask = ma.getmask(dem_m))

    return ph_topo_m



#%%

def Mogi(m,xloc,nu,mu):
    """
    Computes displacements, strains and stresses from a point (Mogi) source. 
    Inputs m and xloc can be matrices; for multiple models, the deformation fields from each are summed.

    Inputs:
        m = 4xn volume source geometry (length; length; length; length^3)
            (x-coord, y-coord, depth(+), volume change)
     xloc = 3xs matrix of observation coordinates (length)
       nu = Poisson's ratio
       mu = shear modulus (if omitted, default value is unity)
    
    Outputs:
        U = 3xs matrix of displacements (length)
            (Ux,Uy,Uz)
        D = 9xn matrix of displacement derivatives
            (Dxx,Dxy,Dxz,Dyx,Dyy,Dyz,Dzx,Dzy,Dzz)
        S = 6xn matrix of stresses
            (Sxx,Sxy,Sxz,Syy,Syz,Szz)
    
    History:
        For information on the basis for this code see:
        Okada, Y. Internal deformation due to shear and tensile faults in a half-space, Bull. Seismol. Soc. Am., 82, 1018-1049, 1992.
        1998/06/17 | Peter Cervelli.
        2000/11/03 | Peter Cervelli, Revised 
        2001/08/21 | Kaj Johnson,  Fixed a bug ('*' multiplication should have been '.*'), , 2001. Kaj Johnson
        2018/03/31 | Matthw Gaddes, converted to Python3  #, but only for U (displacements)
    """
    #import scipy.io
    import numpy as np
    
    _, n_data = xloc.shape
    _, models = m.shape
    Lambda=2*mu*nu/(1-2*nu)
    U=np.zeros((3,n_data))                        # set up the array to store displacements
                  
    for i in range(models):                         # loop through each of the defo sources
        C=m[3]/(4*np.pi)
        x=xloc[0,:]-float(m[0,i])                          # difference in distance from centre of source (x)
        y=xloc[1,:]-float(m[1,i])                         # difference in distance from centre of source (y)
        z=xloc[2,:]
        d1=m[2,i]-z
        d2=m[2,i]+z
        R12=x**2+y**2+d1**2
        R22=x**2+y**2+d2**2
        R13=R12**1.5
        R23=R22**1.5
        R15=R12**2.5
        R25=R22**2.5
        R17=R12**3.5
        R27=R12**3.5
            
        #Calculate displacements
        U[0,:] = U[0,:] + C*( (3 - 4*nu)*x/R13 + x/R23 + 6*d1*x*z/R15 )
        U[1,:] = U[1,:] + C*( (3 - 4*nu)*y/R13 + y/R23 + 6*d1*y*z/R15 )
        U[2,:] = U[2,:] + C*( (3 - 4*nu)*d1/R13 + d2/R23 - 2*(3*d1**2 - R12)*z/R15)
    return U


def signal_deformation(dem, water_mask, mogi_cent, source_cent, ijk_m, ll_extent, pix_in_m=92.6, heading=192, incidence=32):
    """ A function to make deformation from a mogi source and crop to size
    """
    import numpy as np    
    import numpy.ma as ma
    
    mogi_loc_pix = ll2xy((ll_extent[2], ll_extent[0]), 1200, np.array([[mogi_cent[0][0], mogi_cent[0][1]]]))            #centre  
    mogi_loc_pix[0,1] = np.size(dem, axis=0) - mogi_loc_pix[0,1]                                                                # convert xy in matrix stlye  (ie from the top left, and not bottom left)
    mogi_loc_m = pix_in_m * mogi_loc_pix                                                                                            # convert to m
    mogi_cent.append(np.array([[mogi_loc_m[0,0], mogi_loc_m[0,1], mogi_cent[1], mogi_cent[2]]]).T)              # (xloc (m), yloc(m), depth (m), volume change (m^3)
    U_mogi = Mogi(mogi_cent[3],ijk_m,0.25,30e9)                                                                   # 3d displacement
    look = np.array([[np.sin(heading)*np.sin(incidence)],
                      [np.cos(heading)*np.sin(incidence)],
                      [np.cos(incidence)]])                                                                     # ground to satelite unit vector
    U_mogi_los = look.T @ U_mogi                                                                                # 3d displacement converted to LOS displacement
    
    x_pixs = (ll_extent[1] - ll_extent[0])*1200                     # coordinated of points in matrix form (ie 00 is top left)
    y_pixs = (ll_extent[3] - ll_extent[2])*1200
    
    defo_signal = np.reshape(U_mogi_los, (x_pixs, y_pixs))                                                        # column vector to rank 2 array


    
    defo_signal_crop, crop_ll_ur = crop_matrix_with_ll(defo_signal, (ll_extent[2], ll_extent[0]), 
                                                       1200, source_cent[0], source_cent[1])

    defo_signal_crop = ma.array(defo_signal_crop, mask = water_mask)
    
    return defo_signal, defo_signal_crop


#%%


def dem_make(west, east, south, north, path_tiles, download, void_fill = False):
    """ Make a dem for a region using SRTM3 tiles.  Option to void fill these, and can create a folder 
    to keep tiles in so that they don't need to be redownloaded.  '

    Input:
        west | -179 -> 180 | west of GMT is negative
        east | -179 -> 180 | west of GMT is negative
        south | -90 -> 90  | northern hemishpere is positive
        north | -90 -> 90  | northern hemishpere is positive
        path | string | location of previously downloaded SRTM1 tiles
        download | 1 or 0 | 1 is allowed to download tiles, 0 is not allowed to (checking for files that can't be downloaded is slow)
        void_fill | boolean |
    Output:
        dem | rank 2 array | the dem
        lons | rank 1 array | longitude of bottom left of each 1' x1' grid cell
        lats | rank 1 array | latitude of bottom left of each 1' x1' grid cell
    History:
        2017/01/?? | written?
        2017/02/27 | saved before void filling verion
    
    """
    import numpy as np
    import wget                                                                # for downloading tiles
    import zipfile                                                             # tiles are supplied zipped
    import os
    from scipy.interpolate import griddata                                     # for interpolating over any voids

    def read_hgt_file(file, samples):
        """
        function to open SRTM .hgt files
        taken from: https://librenepal.com/article/reading-srtm-data-with-python/
        """
        with open(file) as hgt_data:
            # Each data is 16bit signed integer(i2) - big endian(>)
            elevations = np.fromfile(hgt_data, np.dtype('>i2'), samples*samples).reshape((samples, samples))
        return elevations

    def tile_downloader(region, tile):
        """
        download SRTM 3 tiles, given the region it's in (e.g. Eurasia, and the lat lon that describes it). 
        Inputs:
            region
            tile
        returns:
            .hgt file
        """
        path_download = 'https://dds.cr.usgs.gov/srtm/version2_1/SRTM3/'            # other locations, but easiest to http from
        filename = wget.download(path_download + region + '/' + tile  + '.hgt.zip')


    samples = 1200                                                          # pixels in 1' of latitude/longitude. 1200 for SRTM3
    null = -32768                                                           # from SRTM documentation
    regions = ['Africa', 'Australia', 'Eurasia', 'Islands', 'North_America', 'South_America']

    lats = np.arange(south, north, 1)
    lons = np.arange(west, east, 1)
    num_x_pixs = lons.size * samples
    num_y_pixs = lats.size * samples
    dem = null * np.ones((num_y_pixs, num_x_pixs))                             # make the blank array of null values

    for j in lons:                                                                                  # one column first, make the name for the tile to try and download
        for k in lats:                                                                              # and then rows for that column
            void_fill_skip = False                                                                  # reset for each tile
            download_success = False                                                                # reset to haven't been able to download        
            # 0: get the name of the tile
            if k >= 0 and j >= 0:                       # north east quadrant
                tile = 'N' + str(k).zfill(2) + 'E' + str(j).zfill(3)                                    # zfill pads to the left with zeros so always 2 or 3 digits long.
            if k >= 0 and j < 0:                        # north west quadant
                tile = 'N' + str(k).zfill(2) + 'W' + str(-j).zfill(3)
            if k < 0 and j >= 0:                        # south east quadrant
                tile = 'S' + str(-k).zfill(2) + 'E' + str(j).zfill(3)
            if k < 0 and j < 0:                         # south east quadrant
                tile = 'S' + str(-k).zfill(2) + 'W' + str(-j).zfill(3)


           # 1: Try to a) open it if already downloaded b) download it c) make a blank tile of null values
            try:
                print(f"{tile} : Trying to open locally...", end = "")
                tile_elev = read_hgt_file(path_tiles + '/' + tile + '.hgt', samples+1)                                 # look for tile in tile folder
                print(" Done!")
            except:                                                                                                    # if we can't find it, try and download it, or fill with 0s
                print(" Failed.")
                if download == 1:
                    print(f"{tile} : Trying to download it...  ", end = "" )
                    for region in regions:                                                                             # Loop through all the SRTM regions (as from a lat and lon it's hard to know which folder they're in)
                        if download_success is False:
                            try:
                                tile_downloader(region , tile)                                                    # srtm data is in different folders for each region.  Try all alphabetically until find the ones it's in
                                download_success = True
                                print(" Done! ")
                            except:
                                download_success = False
                                        
                    if download_success:
                        with zipfile.ZipFile(tile + '.hgt.zip' ,"r") as zip_ref:                                # if we downloaded a file, need to now delete it
                            zip_ref.extractall(path_tiles)
                        os.remove(tile + '.hgt.zip')                                                                # delete the downloaded zip file
                        tile_elev = read_hgt_file(path_tiles + '/' + tile + '.hgt', samples+1)                  # look for tile in tile folder
                        void_fill_skip = False
                    else:
                        print(" Failed. ")
                else:
                    pass
                if (download == 0) or (download_success == False):
                    print(f"{tile} : Replacing with null values (the tile probably doesn't exist and covers only water)...  " , end = "")
                    tile_elev = null * np.ones((samples,samples))
                    void_fill_skip = True                                                                  # it's all a void, so don't try and fill it
                    print(" Done! ")
#                    import ipdb; ipdb.set_trace()
                else:
                    pass


            #2: if required, fill voids in the tile
            if void_fill is True and np.min(tile_elev) == (-32768) and void_fill_skip is False:                             # if there is a void in the tile, and we've said that we want to fill voids.  
                print(f"{tile} : Filling voids in the tile... ", end = "")
                grid_x, grid_y = np.mgrid[0:1200:1  ,0:1200:1 ]
                valid_points = np.argwhere(tile_elev > (-32768))                                               # void are filled with -32768 perhaps?  less than -1000 should catch these
                tile_elev_at_valid_points = tile_elev[valid_points[:,0], valid_points[:,1]]
                tile_elev = griddata(valid_points, tile_elev_at_valid_points, (grid_x, grid_y), method='linear')
                print(' Done!')

            #3: Stitch the current tile into the full dem
            dem[num_y_pixs-((k+1-lats[0])*samples) :num_y_pixs-((k-lats[0])*samples), (j-lons[0])*samples:(j+1-lons[0])*samples] = tile_elev[0:1200,0:1200]
    return dem, lons, lats




def dem_wrapper(crop, path_tiles, download_dems, void_fill, pix_in_m):
    """
    Inputs:
        crop | list | e.g [(40.84, 14.14), 20]                     # lat lon scene width(km)
        path_tils | string | absolute path to location of dem info.
        download_dems | boolean | if true, function will try to download.  If all tiles have already been downloaded, faster if set to False
        void_fill | boolean | if true, will fill voids in dem data

    """
    import numpy as np

    # 0: from centre of scene, work out how big the dem needs to be, and make it
    ll_extent = [np.floor(crop[0][1]-1), np.ceil(crop[0][1]+1), np.floor(crop[0][0]-1) , np.ceil(crop[0][0]+1)]             # west east south north - needs to be square!
    ll_extent = [int(i) for i in ll_extent]                                                                                 # convert to intergers
    
    # 1: Make the DEM
    dem, lons, lats = dem_make(ll_extent[0], ll_extent[1], ll_extent[2], ll_extent[3], path_tiles, download_dems, void_fill)                       # make the dem

    # 2: Crop the DEM to the required size
    dem_crop, _ = crop_matrix_with_ll(dem, (ll_extent[2], ll_extent[0]), 1200, crop[0], crop[1])

    # 3: make a matrix of the x and y positions of each pixel of the cropped DEM.    
    x_pixs = (ll_extent[1] - ll_extent[0])*1200                                                               # coordinated of points in matrix form (ie 00 is top left)
    y_pixs = (ll_extent[3] - ll_extent[2])*1200
    X, Y = np.meshgrid(np.arange(0, x_pixs, 1), np.arange(0,y_pixs, 1))
    ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                          # pairs of coordinates of everywhere we have data
    ijk = np.vstack((ij, np.zeros((1,len(X)**2))))                                                                   #xy and 0 depth
    ijk_m = pix_in_m  * ijk                                                                                          # convert from pixels to metres

    return dem, dem_crop, ijk_m, ll_extent



#%%

def signal_atmosphere_turb(n_atms, water_mask, n_pixs, Lc = None, difference = False, verbose = False,
                    interpolate_threshold = 100, mean_cm = 2):
    """ A function to create synthetic turbulent atmospheres based on the 
    methods in Lohman Simmonds (sic?) 2005.  Note that due to memory issues,
    largers ones are made by interpolateing smaller ones.  Can return atmsopheres
    for an individual acquisition, or as the difference of two (as per an 
    interferogram).  
    
    Inputs:
        n_atms | int | number of atmospheres to generate
        n_pixs | int | side length (squares) for atmospheres in pixels
        Lc     | int | length scale, default is random and different for each one
        interpolate_threshold | int | if n_pixs is greater than this, images will be generated at size interpolate_threshold
                                        and then interpolate to the larger size.  This is as the distance matrix (which is
                                        of size n_pixs**2 x n_pixs**2 can get huge)
        max_cm | float | maximum strength of atmosphere, in cm.  Strength is chosen from a uniform distribution.  
    
    Outputs:
        ph_turb | r3 array | n_atms x n_pixs x n_pixs, UNITS ARE M
        Lc      | r1 array | length scales used for each atmosphere        
        
    2019/09/13 | MEG | adapted extensively from a simple script
    """
    
    import numpy as np
    import numpy.ma as ma
    from scipy.spatial import distance
    import scipy
    
    def generate_correlated_noise(pixel_distances, Lc, n_pixs_generate):
        """ given a matrix of pixel distances and a length scale for the noise,
        generate some 2d noise.  
        """
        
        Cd = np.exp((-1 * pixel_distances)/(Lc))                                     # from the matrix of distances, convert to covariances using exponential equation
        Cd_L = np.linalg.cholesky(Cd)                                               # ie Cd = CD_L @ CD_L.T
        x = np.random.randn(n_pixs_generate**2)                               # Parsons 2007 syntax - x for uncorrelated noise
        y = Cd_L @ x                                                            # y for correlated noise
        y_2d = np.reshape(y, (n_pixs_generate,n_pixs_generate))                 # turn back to rank 2
        
        return y_2d
    
    def rescale_atmosphere(atm, mean_cm):
        """ a function to rescale a 2d atmosphere with any scale to a mean centered
        one with a min and max value drawn from a normal distribution.  
        """
        atm -= np.mean(atm)
        if np.abs(np.min(atm)) > np.abs(np.max(atm)):                               # if range of negative numbers is larger
            atm *= ((np.random.randn(1) + mean_cm) / np.abs(np.min(atm)))             # strength is random up to max of max_cm drawn from uniform distribution
        else:
            atm *= ((np.random.randn(1) + mean_cm)/ np.max(atm))
        
        return atm
    
    #1: determine if linear interpolate is required
    if n_pixs > interpolate_threshold:
        if verbose:
            print(f"'n_pixs' is larger than 'interpolate_threshold' so images will be created "
                  f"at a size of {interpolate_threshold} and interpolated to a size of {n_pixs}.  ")
        n_pixs_generate = interpolate_threshold                                         # images will be generated at size n_pixs_generate
        interpolate = True
    else:
        n_pixs_generate = n_pixs
        interpolate = False
    
    #1a: make an array of the lenghth scales for each atmosphere
    if Lc is None:
        Lc = 10**(0.5 + 3*np.random.rand(n_atms))                                  # so Lc is in range 10^2 to 10^5
    else:
        Lc = np.repeat(Lc, n_atms)                                              # or if a single value is given, repeat so similar to above
    
    #2: calculate distance function between points
    ph_turb = np.zeros((n_atms, n_pixs_generate,n_pixs_generate))                                  # initiate output
    X, Y = np.meshgrid(np.arange(0,n_pixs_generate), np.arange(0,n_pixs_generate))                    # top left is (0,0)
    xy = np.hstack((np.ravel(X)[:,np.newaxis], np.ravel(Y)[:,np.newaxis]))      # convert into piels x 2 column vector
    pixel_distances = distance.cdist(xy,xy, 'euclidean')                        # calcaulte all pixelwise pairs - slow as (pixels x pixels)       
           
    #3: generate atmospheres
    if difference is False:
        for i in range(n_atms):
            ph_turb[i,:,:] = generate_correlated_noise(pixel_distances, Lc[i], n_pixs_generate)
            if verbose:
                print(f'Generated {i} of {n_atms} single acquisition atmospheres.  ')
    elif difference is True:
        for i in range(n_atms):
            y_2d_1 = generate_correlated_noise(pixel_distances, Lc[i], n_pixs_generate)
            y_2d_2 = generate_correlated_noise(pixel_distances, Lc[np.random.randint(0, Lc.shape[0])], n_pixs_generate)         # pick anotehr length scale at random
            ph_turb[i,:,:] = y_2d_1 - y_2d_2                                        # difference between the two atmospheres
            if verbose:
                print(f'Generated {i} of {n_atms} interferogram atmospheres.  ')
    else:
        print("'difference' must be either True or False.  Quitting.  ")
        import sys; sys.exit()    
    
    
    #4: possibly interplate to bigger size
    if interpolate:
        if verbose:
            print('Interpolating to the larger size...', end = '')
        ph_turb_output = np.zeros((n_atms, n_pixs,n_pixs))                                  # initiate output
        for atm_n, atm in enumerate(ph_turb):
            f = scipy.interpolate.interp2d(np.arange(0,n_pixs_generate), np.arange(0,n_pixs_generate), atm, kind='linear')
            ph_turb_output[atm_n,:,:] = f(np.linspace(0, n_pixs_generate, n_pixs), np.linspace(0, n_pixs_generate, n_pixs))
        if verbose:
            print('Done!')
        
    else:
        ph_turb_output = ph_turb
    
    #5: rescale to correct range (i.e. a couple of cm)
    ph_turb_cm = np.zeros(ph_turb_output.shape)
    for atm_n, atm in enumerate(ph_turb_output):
        ph_turb_cm[atm_n,] = rescale_atmosphere(atm, mean_cm)
    
    ph_turb_output *= 0.01                                                      # convert from cm to m
    
    
    water_mask_r3 = ma.repeat(water_mask[np.newaxis,], ph_turb_output.shape[0], axis = 0)
    ph_turb_output_ma = ma.array(ph_turb_output, mask = water_mask_r3)
    
    return ph_turb_output_ma, Lc


#%% other less exciting functions 


def quick_dem_plot(dem, title):
    """ Plot dems quickly
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig1, ax = plt.subplots()                                                       # make a figure to show it
    fig1.canvas.set_window_title(title)
    fig1.suptitle(title)
    matrixPlt = ax.imshow(dem, vmin = 0, vmax = np.max(dem))                                              # best to set lower limit to 0 as voids are filled with -32768 so mess the colours up
    fig1.colorbar(matrixPlt,ax=ax)
    
def crop_matrix_with_ll(im, im_ll, pixs2deg, centre, square):
    """
    A script to crop an image (stored as an array) to have side length 'square' (in km)
    , with 'centre' in the middle.  
    Input:
        im | array | the gridded image as a an array
        im_ll | tuple | lat lon of bottom left corner of im
        pixs2deg | int  | number of pixels in im that are 1deg long 
        centre | tuple | lat lon of centre of crop
        square | int | lenght (in km) of side of crop
        
    Output:
        im_crop | array | cropped image
        [ll, ur] | list of tuples | lat lon of lower left and upper right corner of crop.  
        
    Dependencies:
        ll2xy
    
    2017/03/01 | return the lower left and upper right corner of the cropped dem
    2017/03/02 | fix a bug that caused the area returned by the ll of the corners was twice as long in each dimension as it should be 
        
    """

    import numpy as np
    km_per_deg = 110.9
    pixs2km = pixs2deg/km_per_deg                                                # take 1 deg at 110.9 km - compromise between lat (110.6) and lon (111.3)
    y_pixs, x_pixs = im.shape                                   # get size of image that we're going to crop

    summit_xy = ll2xy(im_ll, pixs2deg, np.array([[centre[0], centre[1]]]))                    #xy from bottom left corner
    summit_xy[0,1] = y_pixs - summit_xy[0,1]                                                    # xy from top left (ie more matrix like notation, but still xy and not yx)                                   
    
    x_low = summit_xy[0,0] - pixs2km*(square/2.0)              # get the max/min coords for that distance from point of interest
    if x_low < 0:                                               # check that not falling outside dem that we have
        x_low = 0
    x_high = summit_xy[0,0] + pixs2km*(square/2.0)
    if x_high > x_pixs:                                         # as above
        x_high = x_pixs
    y_low = summit_xy[0,1] - pixs2km*(square/2.0)
    if y_low < 0:                                               # as above
        y_low = 0
    y_high = summit_xy[0,1] + pixs2km*(square/2.0)
    if y_high > y_pixs:                                         # as above
        y_high = y_pixs                                                  
    
    im_crop = im[int(y_low):int(y_high), int(x_low):int(x_high)]                    # do the cropping
    
    ll_lowleft = (centre[0] - (square/2.0)/km_per_deg, centre[1] - (square/2.0)/km_per_deg)
    ll_upright = (centre[0] + (square/2.0)/km_per_deg, centre[1] + (square/2.0)/km_per_deg)
    
    return im_crop, [ll_lowleft, ll_upright]



   
    

def ll2xy(bottom_left, pix2deg, points):
    """    
    Input:
        bottom_left | 1x2 np.array (lat lon) |lat long of bottom left pixel of xy space
        deg2pix | 1 |number of pixels in 1 deg (e.g. 1200 for SRTM3)
        points  | nx2   np.array (lat lon)| n >= 1 for it to work (ie no 1d arrays, must be at least 1x2)
    Output:
        points_xy | nx2 | (x y) in pixels from lower left corner as intergers 
                                Careful, as matrix indices are from top left forner 
        
    xy space has to be orientated so that north is vertical (ie angles are not supported)
    
    2016/12/14 | written
    @author: Matthew Gaddes
    """
    import numpy as np
    
    n_data, dims = points.shape
    points_diff = points - bottom_left              # difference in degrees from bottom left 
    points_diff_pix = points_diff * pix2deg
    points_xy = np.roll(points_diff_pix, 1, 1)          # lat lon is yx, switch to xy
    points_xy = points_xy.astype(int)                   # pixels must be integers 
    return points_xy                      
 
    
 
def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.  
    Useful when converting between vectors used by BSS methods results that are to be plotted
    
    Inputs:
        col | rank 1 array | 
        pixel_mask | array mask (rank 2)
        
    Outputs:
        source | rank 2 masked array | colun as a masked 2d array
    
    2017/10/04 | collected from various functions and placed here.  
    
    """
    import numpy.ma as ma 
    import numpy as np
    
    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()   
    return source

#%%

