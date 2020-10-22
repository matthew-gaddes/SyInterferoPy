# SyInteferoPy
Generate synthetic interferograms that are similar to those produced by the Sentinel-1 satellites.  

Users can select any subaerial region that is covered by the SRTM3 digital elevation model (DEM), and create an interferogram for there containing deformation due to a selection of simple sources, a topogrpahically correlated atmospheric phase screen (APS), and a turbulent APS.  The temporal behaviour of the  signals can also be set to create time series of interferograms, and regions of incoherence can also be synthesised.  

If you use this software in your research, please cite the [2019 JGR:SE paper](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) that describes it. 

There are four examples included in the repository which are intended to provide easy to follow demonstrations of the code's functionality.  After the installation notes, they are described below.  

<br>

<h5>Installation:</h5>
A suitable python environment can be created using the conda command:<br>
<code>conda env create --file SyInterferoPy.yml</code>

To synthesise topographically correlated atmospheric phase screens (APSs), DEMs are required.  These could be generated using many methods, but the SyInterferoPy functions were designed to work with the [SRTM-DEM-tools](https://github.com/matthew-gaddes/SRTM-DEM-tools) package.  All the dependencies for this package are included in the SyInterferoPy.yml file.  To allow the SRTM-DEM-tools to be used by SyInterferoPy, please download the SRTM-DEM-tools and add them to your path (this line is included in each example and need to be modified by a user).  

<br>

<h5>Example 1: 01_example_atmosphere_and_deformation.py:</h5>

To create a DEM of Campi Flegrei, we set the DEM's centre (lon then lat), and the side length (x, y) in metres: <br>
<code> dem_loc_size = {'centre'        : (14.14, 40.84), 'side_length'   : (20e3,20e3)}        </code> <br>
Before creating the DEM: <br>
<code> dem, lons_mg, lats_mg = SRTM_dem_make(dem_loc_size, **dem_settings) </code> <br>

![02_DEM_Cropped_to_area_of_interest,_with_water_masked_(as_white)](https://user-images.githubusercontent.com/10498635/89804030-88000200-db2b-11ea-8bad-b788418c3740.png)


And convert the Campi Flegrei DEM into a topographically correlated APS: <br>
<code>  signals_m['topo_correlated_APS'] = atmosphere_topo(dem, strength_mean = 56.0, strength_var = 12.0, difference = True) </code><br>
![Topographically_correlated_APS](https://user-images.githubusercontent.com/10498635/89804099-a0701c80-db2b-11ea-8ce2-3d1ce33daddc.png)

We can add deformation from a point (Mogi) source by first setting its latitude and longitude, and then its depth and volume change (in m and m^3m, respectively):\
<code>deformation_ll = (14.14, 40.84,)     </code>\
<code> mogi_kwargs = {'volume_change' : 1e6, 'depth': 2000} </code>\
<code> los_grid, x_grid, y_grid, z_grid = deformation_wrapper(lons_mg, lats_mg, deformation_ll, 'mogi', dem, **mogi_kwargs) </code>\
![06_Deformaiton_signal](https://user-images.githubusercontent.com/10498635/89804198-be3d8180-db2b-11ea-8ecf-0571c4f87cf7.png)


We can also generate a turbulent APS:\
<code> ph_turb = atmosphere_turb(1, lons_mg, lats_mg, water_mask, Lc = Lc, verbose=True, interpolate_threshold = 5e3, mean_m = 0.02) </code> \

![09_Turbulent_APS_-_just_spatially_correlated_noise](https://user-images.githubusercontent.com/10498635/89804223-c72e5300-db2b-11ea-9a66-cbe1edf83b47.png)


Before combining the signals to make an interferogram:\
![_Combined](https://user-images.githubusercontent.com/10498635/81292410-765a1a80-9063-11ea-9c9c-e02684adb437.png)


Or set the constituent signals' temporal behaviour to make a time series:\
![Synthetic_Interferograms](https://user-images.githubusercontent.com/10498635/81292573-bae5b600-9063-11ea-84cb-fc028c1eed07.png)


<h5>Example 2: 02_example_deformations.py</h5>
Users can now generate deformation patterns for opening dykes and sills (i.e. Using Okada's results).  \
The location of the deformation is set as a (lon,lat) tuple, as in the previous example:
<code> deformation_ll = (14.14, 40.84,)   </code>
And with arguments specific to the type of deformation pattern.  E.g.: 
  
<code> dyke_kwargs = {'strike' : 0,
        'top_depth' : 1000,
        'bottom_depth' : 3000,
        'length' : 5000,
        'dip' : 80,
        'opening' : 0.5} </code> 


![03:_Opening_Dyke](https://user-images.githubusercontent.com/10498635/89890070-a5d07400-dbca-11ea-8c17-c3bb9a8fa35c.png)

        
Or, E.g.: 
<code>sill_kwargs = {'strike' : 0,
               'depth' : 3000,
               'width' : 5000,
               'length' : 5000,
               'dip' : 1,
               'opening' : 0.5} </code>

![04:_Inflating_Sill](https://user-images.githubusercontent.com/10498635/89890071-a6690a80-dbca-11ea-9ff9-c207e9086f1c.png)

Examples of the deformation patterns for earthquakes with uniform slip have also been included, but the results of these have not been tested.  



<h5>Example 3: 03_example_random_ifgs.py</h5>
The SyInterferoPy package has been developed to generate large datasets of synthetic interferograms for training deep learning models.  Example 3 showcases this, and creates an dataset of seven synthetic interferograms at a collection of subaerial volcanoes.  Deformation patterns include those due to dykes, sills, and point (Mogi) sources, but interferograms without deformation can also be synthesised.   \

During creation of the interferograms, a diagnostic figure can be produced to determine the steps being taken during creation of the interferograms.  The bottom row shows the DEM for the volcano of choice, and a mask of incoherent areas.  In the top row, random crops of the DEM are taken and a deformation pattern is generated, and this step is repeated until the majority of the deformation pattern is on land and areas that are not being classed as incoherent.  In the second row, turbulent APSs and topographically correlated APSs are generated, and the visibility of the deformation pattern is checked by computing the signal-to-noise-ratio (SNR): \

![1_Volcano:Taal](https://user-images.githubusercontent.com/10498635/96578458-23041d80-12cd-11eb-8f36-111a868a0dc4.png)

The .gif below shows the algorithm running for several volcanoes in real time:

![SyInterferoPy_random_ifgs](https://user-images.githubusercontent.com/10498635/96578459-239cb400-12cd-11eb-85a9-30ca6cc4494a.gif)

When using deep learning models, it is common to use transfer learning and therefore encounter models that have been trained for use with RGB (red-green-blue) three channel inputs (e.g. each image is of size 224x224x3).  The SyInterferoPy package includes the option to create three channel data using the "outputs" argument, and by default can create the three channel data that is described in Gaddes et al (in prep.).  The figure below shows five possible arrangments, in which the wrapped phase, unwrapped phase, DEM, real component of complex interferogram, and imaginary components of complex interferogram are combined in a variety of ways to create three channel data.  Sadly, the conclusion of the experiment in Gaddes et al. (in prep.) was that performance is degraded when using anything other than the wrapped or unwrapped phase repeated across three channels.  

![Channel_format_of_data](https://user-images.githubusercontent.com/10498635/96867085-2b429100-1464-11eb-9be0-ebcb1445b900.png)



<h5>Example 4: 04_turbulent_APS_example.py</h5>
The SyInterferoPy package generates spatially correlated noise for use in either turbulent APSs and as incoherence masks.  This is done following the algorithm presented in Lohman and Simons (2005), and by setting a different length scale, different styles of signals can be created: \

![Figure_1](https://user-images.githubusercontent.com/10498635/96580192-d1a95d80-12cf-11eb-92d6-6050c8d05e2d.png)

