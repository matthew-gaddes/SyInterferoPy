#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:22:56 2020

@author: matthew
"""
#%%


def deformation_wrapper(dem, dem_ll_extent, deformation_ll, source, m_in_pix = 92.6,
                        asc_or_desc = 'asc', incidence = 23, **kwargs):
    """ A function to prepare grids of pixels and deformation sources specified in lon lat for use with 
    deformation generating functions that work in metres.  
    
    Inputs:
        dem | rank 2 masked array | The dem, with water masked.   
        dem_ll_extent | list of tuples | [(lon lat lower left corner), (lon lat upper right corner)]
        deformation_ll | tuple | (lon, lat) of centre of deformation source.  
        source | string | mogi or quake or dyke or sill
        m_in_pix | float |x number of metres in 1 pixel.  92.6 for 3 arc second (i.e. SRTM3)
        asc_or_dec | string | 'asc' or 'desc' or 'random'.  If set to 'random', 50% chance of each.  
        incidence | float | satellite incidence angle.  
        **kwargs | various parameters required for each type of source.  E.g. opening, as opposed to slip or volumne change etc.  
    
    Returns:
        los_grid | rank 2 array | displacment in satellite LOS at each location.  
        x_grid | rank 2 array | x displacement at each location
        y_grid | rank 2 array | y displacement at each location
        z_grid | rank 2 array | z displacement at each location
    
    History:
        2020/08/07 | MEG | Written.  
    """
    
    import numpy as np    
    import numpy.ma as ma
    from auxiliary_functions import ll2xy
    
         
    # 0: Make a satellite look vector.  
    if asc_or_desc == 'asc':
        heading = 192.04
    elif asc_or_desc == 'desc':
        heading = 012.04
    elif asc_or_desc == 'random':
        if (-0.5+np.random.rand()) < 0.5:
            heading = 192.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are descending
        else:
            heading = 012.04                                                            # Heading (azimuth) of satellite measured clockwise from North, in degrees, half are ascending
    else:
        raise Exception(f"'asc_or_desc' must be either 'asc' or 'desc' or 'random', but is currently {asc_or_desc}.  Exiting.   ")

    # matlab implementation    
    deg2rad = np.pi/180
    sat_inc = 90 - incidence
    sat_az  = 360 - heading
#    sat_inc=Incidence                                                      # hmmm - why the different definition?
 #   sat_az=Heading;        
    los_x=-np.cos(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_y=-np.sin(sat_az*deg2rad)*np.cos(sat_inc*deg2rad);
    los_z=np.sin(sat_inc*deg2rad);
    los_vector = np.array([[los_x],
                            [los_y],
                            [los_z]])                                          # Unit vector in satellite line of site    
    # my Python implementaiton
    # look = np.array([[np.sin(heading)*np.sin(incidence)],
    #               [np.cos(heading)*np.sin(incidence)],
    #               [np.cos(incidence)]])                                                                     # ground to satelite unit vector

    
    # 1: Make a grid of points in metres to be passed to deformation functions.  
    x = m_in_pix * np.arange(0, dem.shape[1])
    y = m_in_pix * np.arange(0, dem.shape[0])
    xx, yy = np.meshgrid(x, y)
    yy = np.flipud(yy)
    # import matplotlib.pyplot as plt
    # f, axes = plt.subplots(1)
    # axes.imshow(yy)


    xx = np.ravel(xx)
    yy = np.ravel(yy)
    zz = np.zeros(xx.shape)                                                                  # observations of deformation will be at the surface.  
    xyz_m = np.vstack((xx[np.newaxis,:], yy[np.newaxis,:], zz[np.newaxis,:]))                        # x is row1, y is row 2, z is row 3
    
    

    

    # 2: calculate deformation location in the new metres from lower left coordinate system.  
    deformation_xy = ll2xy(np.asarray(dem_ll_extent[0])[np.newaxis,:], 1201,  
                           np.asarray(deformation_ll)[np.newaxis,:])            #1x2 array of lon lat of bottom left corner and of points of interest (the deformation lon lat), and the number of pixels in a degree
    
    #deformation_xy[0,1] = np.size(dem, axis=0) - deformation_xy[0,1]                        # originally xy was from bottom left (is as per axes) but now convert to matrix style where 0,0 is top left
    deformation_m = deformation_xy * m_in_pix                                               # convert from number of pixels from lower left corner to number of metres from lower left corner, 1x2 array.  
    

    # 3: Calculate the deformation:
    if source == 'mogi':
        model_params = np.array([deformation_m[0,0], deformation_m[0,1], kwargs['depth'], kwargs['volume_change']])[:,np.newaxis]
        U = deformation_Mogi(model_params, xyz_m, 0.25,30e9)                                                                   # 3d displacement, xyz are rows, each point is a column.  
    elif (source == 'quake') or (source == 'dyke') or (source == 'sill'):
        U = deformation_eq_dyke_sill(source, (deformation_m[0,0], deformation_m[0,1]),
                                     xyz_m, n_pixs = dem.shape[0], m_in_pix = m_in_pix,  **kwargs)
    else:
        raise Exception(f"'source' can be eitehr 'mogi', 'quake', 'dyke', or 'sill', but not {source}.  Exiting.  ")
    
    # 4: convert the xyz deforamtion in movement in direction of satellite LOS (ie. upwards = positive, despite being range shortening)    
    x_grid = np.reshape(U[0,], (len(y), len(x)))
    y_grid = np.reshape(U[1,], (len(y), len(x)))
    z_grid = np.reshape(U[2,], (len(y), len(x)))
    los_grid = x_grid*los_vector[0,0] + y_grid*los_vector[1,0] + z_grid*los_vector[2,0]
    
    return los_grid, x_grid, y_grid, z_grid
 




#%%


def deformation_eq_dyke_sill(source, source_xy_m, xyz_m, n_pixs = 324, m_in_pix = 90,  **kwargs):    
    """
    A function to create deformation patterns for either an earthquake, dyke or sill.  
    This functions calls disloc3d4, which then calls dc3d4 (earthquake), dc3d5 (dyke), or dc3d6 (sill).  
    To aid in readability, different sources take different parameters (e.g. slip for a quake, opening for a dyke)
    are passed separately as kwargs, even if they ultimately go into the same field in the model parameters.  
    
    A quick recap on definitions:
        strike - measured clockwise from 0 at north, 180 at south.  fault dips to the right of this.  hanging adn fo
        dip - measured from horizontal, 0 for horizontal, 90 for vertical.  
        rake - direction the hanging wall moves during rupture, measured relative to strike, anticlockwise is positive, so:
            0 for left lateral ss 
            180 (or -180) for right lateral ss
            -90 for normal
            90 for thrust

    Inputs:
        source | string | quake or dyke or sill
        source_xy_m | tuple | x and y location of centre of source, in metres.  
        xyz_m | rank2 array | x and y locations of all points in metres.  0,0 is top left?  
        n_pixs | int | rank 2 arrays returned will be square, and with this many pixels on each side.  
        m_in_pix | int | sets the size of a pixel.  defaults is 90m for SRTM3 resolution.  
        
        examples of kwargs:
            
        quake_normal = {'strike' : 0,
                        'dip' : 70,
                        'length' : 5000,
                        'rake' : -90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_thrust = {'strike' : 0,
                        'dip' : 30,
                        'length' : 5000,
                        'rake' : 90,
                        'slip' : 1,
                        'top_depth' : 4000,
                        'bottom_depth' : 8000}
        
        quake_ss = {'strike' : 0,
                    'dip' : 80,
                    'length' : 5000,
                    'rake' : 0,
                    'slip' : 1,
                    'top_depth' : 4000,
                    'bottom_depth' : 8000}
        
        dyke = {'strike' : 0,
                'top_depth' : 1000,
                'bottom_depth' : 3000,
                'length' : 5000,
                'dip' : 80,
                'opening' : 0.5}
        
        sill = {'strike' : 0,
                'depth' : 3000,
                'width' : 5000,
                'length' : 5000,
                'dip' : 1,
                'opening' : 0.5}
        
    Returns:
        x_grid | rank 2 array | displacment in x direction for each point (pixel on Earth's surface)
        y_grid | rank 2 array | displacment in y direction for each point (pixel on Earth's surface)
        z_grid | rank 2 array | displacment in z direction for each point (pixel on Earth's surface)
        los_grid | rank 2 array | change in satellite - ground distance, in satellite look angle direction. Need to confirm if +ve is up or down.  
        
    History:
        2020/08/05 | MEG | Written
    """    
    import numpy as np
    from oct2py import Oct2Py  
    
    oc = Oct2Py(temp_dir="./m_files/octave_temp_dir/")
    oc.addpath("./m_files/")
    
    # 1:  Setting for elastic parameters.  
    lame = {'lambda' : 2.3e10,                                                         # elastic modulus (Lame parameter, units are pascals)
        'mu'     : 2.3e10}                                                         # shear modulus (Lame parameter, units are pascals)
    #v = lame['lambda'] / (2*(lame['lambda'] + lame['mu']));                         #  calculate poisson's ration
        
        
    # 2: set up regular grid
    min_max = int((m_in_pix * (n_pixs-1)) / 2)                                    # clcaulte number of m to go in each direction to get the right number of pixels
    x = np.arange(-min_max, min_max+1, m_in_pix)                                  # when the grid is set like this, 0 is in the middle (and it goes from -10000 to 10000 for a scene of width 20km)
    y = np.arange(-min_max, min_max+1, m_in_pix)                                            
    xx, yy = np.meshgrid(x, y)
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    coords = np.vstack((xx[np.newaxis,:], yy[np.newaxis,:]))                        # x is top row, y is bottom row
       
    # import matplotlib.pyplot as plt
    # both_arrays = np.hstack((np.ravel(coords), np.ravel(xyz_m)))
    # f, axes = plt.subplots(1,2)
    # axes[0].imshow(coords, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                 # goes from -1e4 to 1e4
    # axes[1].imshow(xyz_m, aspect = 'auto', vmin = np.min(both_arrays), vmax = np.max(both_arrays))                  # goes from 0 to 2e4
    
    
    # 3: set 10x1 of model parameters needed by disloc3d4, and call
    model = np.zeros((10,1))
    model[0,0] = source_xy_m[0]                                                  # some parameters are shared for all three types and can be set here
    model[1,0] = source_xy_m[1]
    model[2,0] = kwargs['strike']
    model[3,0] = kwargs['dip']
    model[6,0] = kwargs['length']
    if source == 'quake':                                           # otherwise they need to be set for each type
        model[4,0] = kwargs['rake']
        model[5,0] = kwargs['slip']
        model[7,0] = kwargs['top_depth']
        model[8,0] = kwargs['bottom_depth']
        model[9,0] = 1
    elif source == 'dyke':
        model[4,0] = 0
        model[5,0] = kwargs['opening']
        model[7,0] = kwargs['top_depth']
        model[8,0] = kwargs['bottom_depth']
        model[9,0] = 2    
    elif source == 'sill':
        model[4,0] = 0
        model[5,0] = kwargs['opening']
        model[7,0] = kwargs['depth']
        model[8,0] = kwargs['width']
        model[9,0] = 3        
    else:
        raise Exception(f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting.")
    
    U = oc.disloc3d4(model, xyz_m[:2,], lame['lambda'], lame['mu'])                           # U is 3xnpoints and is the displacement in xyz directions.  
                                                                                              # Note that disloc3d4 only wants xy and assumes z is 0 (hence only taking first two rows).  
       
    return U





#%%

def deformation_Mogi(m,xloc,nu,mu):
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
    #Lambda=2*mu*nu/(1-2*nu)
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

#%%
    
def atmosphere_topo(dem_m, strength_mean = 56.0, strength_var = 2.0, difference = False):
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
    ph_topo_m -= ma.mean(ph_topo_m)                                                 # mean centre the signal

    return ph_topo_m

#%%

def atmosphere_turb(n_atms, water_mask, n_pixs, Lc = None, difference = False, verbose = False,
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




