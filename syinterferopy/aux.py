#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:14:05 2020

@author: matthew
"""

import pdb

#%%

def rescale_defo(defos, magnitude = 1.):
    """ Given deformation patterns in metres, rescale so that the maximum value is "magnitude".
    Inputs:
        defos | r3 ma | n_times x ny x nx, water masked.  
        magnitude | float | new maximum signals
    Retrurns:
        defos | r3 ma | As above, with new magnitude. 
    History:
        2023_08_25 | MEG | Written. 
    """
    import numpy.ma as ma
    defo_maxs = ma.max(defos, axis = (1,2))         
    for defo_n, defo in enumerate(defos):
        defo /= defo_maxs[defo_n]
    return defos

#%%

def lon_lat_to_ijk(lons_mg, lats_mg):
    """ Given a meshgrid of the lons and lats of the lower left corner of each pixel, 
    find their distances (in metres) from the lower left corner.  
    Inputs:
        lons_mg | rank 2 array | longitudes of the lower left of each pixel.  
        lats_mg | rank 2 array | latitudes of the lower left of each pixel.  
    Returns:
        ijk | rank 2 array | 3x lots.  The distance of each pixel from the lower left corner of the image in metres.  
        pixel_spacing | dict | size of each pixel (ie also the spacing between them) in 'x' and 'y' direction.  
    History:
        2020/10/01 | MEG | Written 
    """
    from geopy import distance
    import numpy as np
    
    ny, nx = lons_mg.shape
    pixel_spacing = {}
    pixel_spacing['x'] = distance.distance((lats_mg[-1,0], lons_mg[-1,0]), (lats_mg[-1,0], lons_mg[-1,1])).meters                  # this should vary with latitude.  Usually around 90 (metres) for SRTM3 resolution.  
    pixel_spacing['y'] = distance.distance((lats_mg[-1,0], lons_mg[-1,0]), (lats_mg[-2,0], lons_mg[-1,0])).meters                  # this should be constant at all latitudes.  
   
    X, Y = np.meshgrid(pixel_spacing['x'] * np.arange(0, nx), pixel_spacing['y'] * np.arange(0,ny))                       # make a meshgrid
    Y = np.flipud(Y)                                                                                                      # change 0 y cordiante from matrix style (top left) to axes style (bottom left)
    ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                                    # pairs of coordinates of everywhere we have data   
    ijk = np.vstack((ij, np.zeros((1, ij.shape[1]))))                                                                     # xy and 0 depth, as 3xlots
    
    return ijk, pixel_spacing

#%%



# def dem_wrapper(dem_ll_width, path_tiles, download_dems, void_fill, m_in_pix, srtm_dem_tools_bin, water_mask_resolution = 'i'):
#     """
#     A function to quickly create a small (tightly cropped dem).  This is done efficiently by not maskign the water bodies in the full DEM, and only 
#     doing so in the cropped dem.  
    
#     Inputs:
#         dem_ll_width | list | e.g [(14.14, 40.84,), 20]                     # lon lat (deg), scene width(km)
#         path_tils | string | absolute path to location of dem info.
#         download_dems | boolean | if true, function will try to download.  If all tiles have already been downloaded, faster if set to False
#         void_fill | boolean | if true, will fill voids in dem data
#         m_in_pix | float | number of metres in one pixel.  e.g. 92.6 for 3 arc second pixels.  
#         srtm_dem_tools_bin | path | path to where the SRTM_tools package is located.
#         water_mask_resolution | string | sets the resolution of the mask of the water pixels.   c (crude), l (low), i (intermediate), h (high), f (full) 
        
#     Returns:
#         dem_ma | masked array | large dem (whole number of tiles)
#         dem_ma_crop | masked array | cropped to the scene width set in "crop"
#         ijk_m | numpy array | x and y position of each pixel in metres
#         ll_extent | list of tuples | [(lon lat lower left corner), (lon lat upper right corner)]
#         ll_extent_crop | list of tuples | [(lon lat lower left corner), (lon lat upper right corner)]
        
#     History:
#         2018/??/?? | MEG | Written
#         2020/05/06 | Overhauled for Github.  

#     """
#     import numpy as np
#     import numpy.ma as ma
#     from auxiliary_functions import crop_matrix_with_ll
    
#     import sys
#     sys.path.append(srtm_dem_tools_bin)
#     from dem_tools_lib import SRTM_dem_make, water_pixel_masker
#     pixs_in_deg = 1201                                                                      # for srtm3
    
#     # 0: from centre of scene, work out how big the dem needs to be
#     ll_extent = [(np.floor(dem_ll_width[0][0]-1).astype(int), np.floor(dem_ll_width[0][1]-1).astype(int)),
#                  (np.ceil(dem_ll_width[0][0]+1).astype(int), np.ceil(dem_ll_width[0][1]+1).astype(int))]                           # (lon lat of lower left corner), (lon lat of upper right corner)

    
      
#     # 1: Make the DEM
#     dem, lons, lats = SRTM_dem_make(ll_extent[0][0], ll_extent[1][0], ll_extent[0][1], ll_extent[1][1], SRTM1_or3 = 'SRTM3',
#                                     SRTM3_tiles_folder = path_tiles, water_mask_resolution = None,                  
#                                     download = download_dems, void_fill = void_fill)                                    # make the dem, note taht water_mask_resolution is set to None so that no (time consuming) masking of water bodies happens.  

#     # 2: Crop the DEM to the required size
#     dem_crop, ll_extent_crop = crop_matrix_with_ll(dem, ll_extent[0], pixs_in_deg, dem_ll_width[0], dem_ll_width[1])      # crop the dem.  converted to lon lat format.  ll_extent_crop is [(lower left lon lat),(upper right lon lat)]
    
    
#     # 3: Mask the water in the cropped DEM
#     print(f"Masking the water bodies in the cropped DEM...", end = '')
#     mask_water = water_pixel_masker(dem_crop, ll_extent_crop[0], ll_extent_crop[1], water_mask_resolution, verbose = False)        # make the mask for the cropped DEM, extent is given as lonlat_lowerleft and lonlat_upperright
#     dem_crop_ma = ma.array(dem_crop, mask =  mask_water)                                                                             # apply the mask
#     print(f" Done!")

#     # 4: make a matrix of the x and y positions of each pixel of the cropped DEM.    
#     x_pixs = (ll_extent[1][0] - ll_extent[0][0])*1201                                                               # coordinated of points in matrix form (ie 00 is top left)
#     y_pixs = (ll_extent[1][1] - ll_extent[0][1])*1201
#     X, Y = np.meshgrid(np.arange(0, x_pixs, 1), np.arange(0,y_pixs, 1))
#     ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                          # pairs of coordinates of everywhere we have data
#     ijk = np.vstack((ij, np.zeros((1,len(X)**2))))                                                                   #xy and 0 depth
#     ijk_m = m_in_pix * ijk                                                                                          # convert from pixels to metres

#     return dem, dem_crop_ma, ijk_m, ll_extent, ll_extent_crop






#%% other less exciting functions 

def griddata_plot(griddata, lons_mg, lats_mg, title, dem_mode = True):
    """ Plot dems quickly, using lats and lons for tick labels.  
    
    Inputs:
        griddata | rank 2 array | dem, can be a masked array or a normal array.  
        lons_mg | rank 2 array | longitudes of each pixel in griddata.
        lats_mg | rank 2 array | latitudes of each pixel in griddata.  
        title | string | figure title.  
        dem_mode | boolean | If True, uses matplotlib terrain colourmap (with blues removed)
    Returns:
        figure
    History:
        2020/07/?? | MEG | Written
        2020/08/10 | MEG | Add tick labels in lon lat style.  
        20e0/10/01 | MEG | update to use meshgrid of lons and lats.  
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
  
    
    if dem_mode:
        cmap = plt.get_cmap('terrain')                                                          # makes sense for DEMs
        cmap = truncate_colormap(cmap, 0.2, 1)                                                  # but by deafult it starts at blue, so crop (truncate) that part off.  
    else:
        cmap = plt.get_cmap('coolwarm')
        cmap_mid = 1 - np.max(griddata)/(np.max(griddata) + abs(np.min(griddata)))          # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
        if cmap_mid > 0.5:
            cmap = remappedColorMap(cmap, start=0.0, midpoint=cmap_mid, stop=(0.5 + (1-cmap_mid)), name='shiftedcmap')
        else:
            cmap = remappedColorMap(cmap, start=(0.5 - cmap_mid), midpoint=cmap_mid, stop=1, name='shiftedcmap')
    
    
    fig1, ax = plt.subplots()                                                       # make a figure to show it
    fig1.canvas.manager.set_window_title(title)
    ax.set_title(title)
    if dem_mode:
        matrixPlt = ax.imshow(griddata, vmin = 0, vmax = np.max(griddata), cmap = cmap)             # best to set lower limit to 0 as voids are filled with -32768 so mess the colours up
    else:
        matrixPlt = ax.imshow(griddata, cmap=cmap)                                                  # if there are no voids, can just let 
    fig1.colorbar(matrixPlt,ax=ax)                                                                  # add a colour bar
    
    # create the tick labels which are lons and lats
    lons = lons_mg[-1,:]                                                                            # just get a rank 1 of the longitudes of each pixel.  
    xtick_pixel_n = np.linspace(0, griddata.shape[1]-1, 10).astype(int)                             # only plot every 10th?
    plt.xticks(xtick_pixel_n, np.round(lons[xtick_pixel_n],2), rotation = 'vertical')               # update xticks to longitudues, 2dp for clarity.  
    ax.set_xlabel('Longitude (degs)')
    lats = lats_mg[:,0]                                                                             # get a rank 1 of the latitudes of each pixel
    ytick_pixel_n = np.linspace(0, griddata.shape[0]-1, 10).astype(int)                             # only plot every 10th? 
    plt.yticks(ytick_pixel_n, np.round(lats[ytick_pixel_n],2)[::1])                                # update yticks to latitudes.  have removed - from before 1 to not reverse
    ax.set_ylabel('Latitude (degs)')
    
    fig1.tight_layout()    
    
#%%
    
    
# def crop_matrix_with_ll(im, im_ll, pixs2deg, centre_ll, square):
#     """
#     A script to crop an image (stored as an array) to have side length 'square' (in km)
#     , with 'centre' in the middle.  
#     Input:
#         im | rank2 array | the gridded image as a an array
#         im_ll | tuple | lon lat of bottom left corner of im
#         pixs2deg | int  | number of pixels in im that are 1deg long 
#         centre_ll | tuple | lon lat of centre of crop
#         square | int | lenght (in km) of side of crop
        
#     Output:
#         im_crop | array | cropped image
#         [ll, ur] | list of tuples | lat lon of lower left and upper right corner of crop.  
        
#     Dependencies:
#         ll2xy
    
#     2017/03/01 | MEG |  return the lower left and upper right corner of the cropped dem
#     2017/03/02 | MEG |  fix a bug that caused the area returned by the ll of the corners was twice as long in each dimension as it should be 
#     2020/10/07 | MEG | change to lon lat format.  
#     """


#     import numpy as np
#     km_per_deg = 110.9                                                           #  
#     pixs2km = pixs2deg/km_per_deg                                                # take 1 deg at 110.9 km - compromise between lat (110.6) and lon (111.3)
#     y_pixs, x_pixs = im.shape                                                    # get size of image that we're going to crop

#     centre_xy = ll2xy(im_ll, pixs2deg, np.asarray(centre_ll)[np.newaxis,:])      # xy from bottom left corner, instead of as a lonlat
#     centre_xy[0,1] = y_pixs - centre_xy[0,1]                                     # xy from top left (ie more matrix like notation, but still xy and not yx)                                   
    
#     x_low = centre_xy[0,0] - pixs2km*(square/2.0)               # get the max/min coords for that distance from point of interest
#     if x_low < 0:                                               # check that not falling outside dem that we have
#         x_low = 0
#     x_high = centre_xy[0,0] + pixs2km*(square/2.0)
#     if x_high > x_pixs:                                         # as above
#         x_high = x_pixs
    
#     y_low = centre_xy[0,1] - pixs2km*(square/2.0)
#     if y_low < 0:                                               # as above, but for y
#         y_low = 0
#     y_high = centre_xy[0,1] + pixs2km*(square/2.0)
#     if y_high > y_pixs:                                         # as above
#         y_high = y_pixs                                                  
   
#     im_crop = im[int(y_low):int(y_high), int(x_low):int(x_high)]                    # do the cropping
#     ll_lowleft = (centre_ll[0] - (square/2.0)/km_per_deg, centre_ll[1] - (square/2.0)/km_per_deg)                   # work out the lonlats of the corners of the cropped dem
#     ll_upright = (centre_ll[0] + (square/2.0)/km_per_deg, centre_ll[1] + (square/2.0)/km_per_deg)
    
#     return im_crop, [ll_lowleft, ll_upright]

#%%

def localise_data(r2_data, centre = None, threshold = False):
    """ Return a region that contains deformation above the threshold 
        (both positive and negative are considered)
    Inputs:
        r2_data | rank2 array | image / defomration map in m
        centre | tuple | centre of deformation signal, appears to be in matrix notation (ie 0,0 is top left)
        threshold | float |  value above which deformation is selected.  If False, set to 20% of maximum absolute deformation
        
    2019/02/?? - Written
    2019/03/18 | Convert output to be location of max deformation and half width in x and y direction (to allow for drawing of box around deformaiton)
    2019/03/19 | centre is now calculated as the centre of mass of the absolute deformation.  
    """
    import numpy as np
    from scipy import ndimage                                            # for centre of mass

    if threshold is False:                                               # if not passed to function, set it specific to this data
        threshold = 0.2 * np.max(np.abs(r2_data))

    #(centre_x, centre_y) = ndimage.measurements.center_of_mass(np.abs(r2_data))        # doesn't seem to work well

    if centre is None:
        centre_x = np.unravel_index(np.argmax(r2_data), r2_data.shape)[1]
        centre_y = np.unravel_index(np.argmax(r2_data), r2_data.shape)[0]
    else:
        centre_x = centre[0]
        centre_y = centre[1]
    
    def_args = np.argwhere(np.abs(r2_data) > threshold)                             # a matrix of the pixels that have a magnitude above the threshold (ie both positive and negative deformation)
    x_start = np.min(def_args[:,1])                                         # column 1 is for xs
    x_stop = np.max(def_args[:,1])
    y_start = np.min(def_args[:,0])                                         # column 0 is for ys
    y_stop = np.max(def_args[:,0])
    x_half_width = int(np.ceil(sum([centre_x-x_start, x_stop-centre_x])/2))               # the size of the pattern in x direction is the mean of the distance from the centre to each edge
    y_half_width = int(np.ceil(sum([centre_y-y_start, y_stop-centre_y])/2))               # ditto for y
    
    return [(centre_x, centre_y), (x_half_width, y_half_width)]             # return as a list of tuples
    

#%%
    

def ll2xy(bottom_left_ll, pix2deg, points_ll):
    """    
    Input:
        bottom_left_ll | 1x2 np.array | lon lat of bottom left pixel of xy space
        deg2pix | int | number of pixels in 1 deg (e.g. 1201 for SRTM3)
        points_ll  | nx2   np.array | n >= 1 for it to work (ie no 1d arrays, must be at least 1x2).  lons in column 1, lats in column 2
    Output:
        points_xy | nx2 | (x y) in pixels from lower left corner as intergers 
                                Careful, as matrix indices are from top left forner 
        
    xy space has to be orientated so that north is vertical (ie angles are not supported)
    
    2016/12/14 | MEG | written
    2020/08/06 | MEG | Change so that ll is lonlat.
    """
    import numpy as np
    
    n_data, dims = points_ll.shape                          # each row is a new pair of coordinates.  
    points_diff = points_ll - bottom_left_ll              # difference in degrees from bottom left 
    points_xy = points_diff * pix2deg
    #points_xy = np.roll(points_diff_pix, 1, 1)          # lat lon is yx, switch to xy
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



def remappedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range (i.e. truncate the colormap so that it isn't
    compressed on the shorter side) . Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.
    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax)
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin))
          Only got this to work with:
              1 - vmin/(vmax + abs(vmin))
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin))

      2017/??/?? | taken from stack exchange
      2017/10/11 | update so that crops shorter side of colorbar (so if data are in range [-1 100],
                   100 will be dark red, and -1 slightly blue (and not dark blue))
      '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    if midpoint > 0.5:                                      # crop the top or bottom of the colourscale so it's not asymetric.
        stop=(0.5 + (1-midpoint))
    else:
        start=(0.5 - midpoint)


    cdict = { 'red': [], 'green': [], 'blue': [], 'alpha': []  }
    # regular index to compute the colors
    reg_index = np.hstack([np.linspace(start, 0.5, 128, endpoint=False),  np.linspace(0.5, stop, 129)])

    # shifted index to match the data
    shift_index = np.hstack([ np.linspace(0.0, midpoint, 128, endpoint=False), np.linspace(midpoint, 1.0, 129)])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))
    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #plt.register_cmap(cmap=newcmap)
    return newcmap



def plot_ifgs(ifgs, pixel_mask, title, n_rows = 3):
    """
    Function to plot a time series of S1 ifgs in a single figure.  
    Useful for quickly visualising the full training or testing time series.  
    Inputs:    
        ifgs | pxt matrix of ifgs as columns (p pixels, t times)
        pixel_mask | mask to turn spaital maps back to regular grided masked arrays
        n_rows | number of columns for ifg plot to have.   
    Ouptuts:
        Figure
    2018/04/18 | taken from ifg_plot_v2
    2020/05/07 | Updated
    
    """ 
    import numpy as np
    import matplotlib.pyplot as plt
    from syinterferopy.aux import remappedColorMap
   
    # 1: colour map stuff
    ifg_colours = plt.get_cmap('coolwarm')
    cmap_mid = 1 - np.max(ifgs)/(np.max(ifgs) + abs(np.min(ifgs)))          # get the ratio of the data that 0 lies at (eg if data is -15 to 5, ratio is 0.75)
    if cmap_mid > 0.5:
        ifg_colours_cent = remappedColorMap(ifg_colours, start=0.0, midpoint=cmap_mid, stop=(0.5 + (1-cmap_mid)), name='shiftedcmap')
    else:
        ifg_colours_cent = remappedColorMap(ifg_colours, start=(0.5 - cmap_mid), midpoint=cmap_mid, stop=1, name='shiftedcmap')
    
    
    # 2: Set-up the plot
    n_cols = int(np.ceil(ifgs.shape[0]/float(n_rows)))
    f, axes = plt.subplots(n_rows, n_cols)
    f.suptitle(title, fontsize=14)
    f.canvas.manager.set_window_title(title)

    # 3: loop through plotting ifgs
    for ifg_n, axe in enumerate(np.ravel(axes)):
        axe.set_yticks([])
        axe.set_xticks([])
        try:
            ifg = col_to_ma(ifgs[ifg_n,:], pixel_mask = pixel_mask)                                        # convert row vector to rank 2 masked array
            im = axe.imshow(ifg, cmap = ifg_colours_cent, vmin=np.min(ifgs), vmax=np.max(ifgs))            # 
        except:
            f.delaxes(axe)

    # 4: shared colorbar
    f.subplots_adjust(right=0.87)            
    cbar_ax = f.add_axes([0.87, 0.01, 0.02, 0.3])
    cbar = f.colorbar(im, cax=cbar_ax)
    cbar.set_label('Combined Signal (m)', fontsize = 8)

#%%

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    import matplotlib.colors as colors
    import numpy as np
    
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap 