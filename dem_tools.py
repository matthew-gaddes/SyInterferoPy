#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:54:45 2020

@author: matthew
"""


#%%

def dem_wrapper(crop, path_tiles, download_dems, void_fill, pix_in_m, mask_resolution = 'i'):
    """
    Inputs:
        crop | list | e.g [(40.84, 14.14), 20]                     # lat lon scene width(km)
        path_tils | string | absolute path to location of dem info.
        download_dems | boolean | if true, function will try to download.  If all tiles have already been downloaded, faster if set to False
        void_fill | boolean | if true, will fill voids in dem data
        mask_resolution | string | sets the resolution of the mask of the water pixels.  'i' for intermediate, 'f' for fine.  Check basemap manual for other values
        
    Returns:
        dem_ma | masked array | large dem (whole number of tiles)
        dem_ma_crop | masked array | cropped to the scene width set in "crop"
        ijk_m | numpy array | x and y position of each pixel in metres
        ll_extent | list | lat lons of large dem taht was made
        ll_extent | list | lat lons of cropped dem that was made
        
    History:
        2018/??/?? | MEG | Written
        2020/05/06 | Overhauled for Github.  

    """
    import numpy as np
    import numpy.ma as ma
    from auxiliary_functions import crop_matrix_with_ll

    # 0: from centre of scene, work out how big the dem needs to be, and make it
    ll_extent = [np.floor(crop[0][1]-1), np.ceil(crop[0][1]+1), np.floor(crop[0][0]-1) , np.ceil(crop[0][0]+1)]             # west east south north - needs to be square!
    ll_extent = [int(i) for i in ll_extent]                                                                                 # [lons lats], convert to intergers
      
    # 1: Make the DEM
    dem, lons, lats = dem_make(ll_extent[0], ll_extent[1], ll_extent[2], ll_extent[3], path_tiles, download_dems, void_fill)                       # make the dem

    # 2: Crop the DEM to the required size
    dem_crop, ll_extent_crop_t = crop_matrix_with_ll(dem, (ll_extent[2], ll_extent[0]), 1200, crop[0], crop[1])          # crop the dem.  Note that ll_extet_crop is lat lon of lower left and upper right corner of crop.           
    ll_extent_crop = [ll_extent_crop_t[0][1], ll_extent_crop_t[1][1], ll_extent_crop_t[0][0], ll_extent_crop_t[1][0]]                # convert to [lons lats] format
    del ll_extent_crop_t

    # 3: Mask the water in the cropped DEM
    print(f"Masking the water bodies in the cropped DEM...", end = '')
    mask_water = water_pixel_masker(dem_crop, (ll_extent_crop[0], ll_extent_crop[1]), 
                                              (ll_extent_crop[2], ll_extent_crop[3]), mask_resolution, verbose = False)                                                           # make the mask for the cropped DEM
    dem_crop_ma = ma.array(dem_crop, mask =  mask_water)                                                                                              # apply the mask
    print(f" Done!")

    # 4: make a matrix of the x and y positions of each pixel of the cropped DEM.    
    x_pixs = (ll_extent[1] - ll_extent[0])*1200                                                               # coordinated of points in matrix form (ie 00 is top left)
    y_pixs = (ll_extent[3] - ll_extent[2])*1200
    X, Y = np.meshgrid(np.arange(0, x_pixs, 1), np.arange(0,y_pixs, 1))
    ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))                                          # pairs of coordinates of everywhere we have data
    ijk = np.vstack((ij, np.zeros((1,len(X)**2))))                                                                   #xy and 0 depth
    ijk_m = pix_in_m  * ijk                                                                                          # convert from pixels to metres

    return dem, dem_crop_ma, ijk_m, ll_extent, ll_extent_crop


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

#%%


def water_pixel_masker(data, lons_range, lats_range, coast_resol = 'i', verbose = False):
    """
       A function to creat a mask of pixels over water. This can be very slow for big DEMs
    Inputs:
        data | rank 2 array | gridded data (e.g. a dem)
        lons_range | tuple | west and east extent of dem
        lats_range | tuple | south and north extent of dem
        coast_resol | str | resolution of vector coastlines: c l i h f 
       
    Output:
        result_ary | rank 2 array | array to be used as pixel mask
        
    2017/03/01 | adapterd from dem_show_oceans_detailed_2
    """
     
    from mpl_toolkits.basemap import Basemap
    from matplotlib.path import Path
    import numpy as np
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    
    if verbose:
        print('Creating a mask of pixels that lie in water (sea and lakes).  This can be slow')
    ll_extent = [lons_range[0], lons_range[-1], lats_range[0], lats_range[-1]]
    ny = data.shape[0]; nx = data.shape[1]
    plt.figure()                                                                # need a figure instance for basemap
    map = Basemap(projection='cyl', llcrnrlat=ll_extent[2],urcrnrlat=ll_extent[3],llcrnrlon=ll_extent[0],urcrnrlon=ll_extent[1], resolution=coast_resol)
    map.drawcoastlines()
    
    
    mesh_lons, mesh_lats = map.makegrid(nx, ny)               # get lat/lons of in evenly spaced grid with increments to match data matrix
    lons = np.ravel(mesh_lons)
    lats = np.ravel(mesh_lats)
    x, y = map(lons, lats)
    locations = np.c_[x, y]                                                     # concatenate to one array
    
    result = np.zeros(len(locations), dtype=bool)                                  # initialise as false
    land_polygons = [Path(p.boundary) for p in map.landpolygons]                    # check if land
    for polygon in land_polygons:
        result += np.array(polygon.contains_points(locations))
    lake_polygons = [Path(p.boundary) for p in map.lakepolygons]                    # check if in lake
    for polygon in lake_polygons:
        result = np.invert(np.array(polygon.contains_points(locations)))              # pixels in lakes are 1s, so subtract these.  
    result = np.invert(result)                                                       # true if in see, so masked out
    water_mask = np.flipud(np.reshape(result, (ny, nx)))
    plt.close()                                                                 # close figure instance
    return water_mask









