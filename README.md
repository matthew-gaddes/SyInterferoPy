# SyInteferoPy
Generate synthetic interferograms that are similar to those produced by the Sentinel-1 satellites.  

Users can select any subaerial region that is covered by the SRTM3 digital elevaiton model (DEM), and create an interferogram for there containing deformation due to a selection of simple sources, a topogrpahically correlated atmospheric phase screen (APS), and a turbulent APS.  The temporal behaviour of the  signals can also be set to create time series of interferograms, and regions of incoherence can also be synthesised.  

If you use this software in your research, please cite the [2019 JGR:SE paper](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) that describes it.  

<br>

<h5>Installation:</h5>
A suitable python environment can be created using the conda command:<br>
<code>conda env create --file synthetic_interferograms.yml</code>

<br>

<h5>Examples:</h5>

We can set a scene of 20km side length centered on Campi Flegrei using just its latitude and longitude:\
<code>scene_centre = [(40.84, 14.14), 20]                                                </code>

![02_DEM_Cropped_to_area_of_interest,_with_water_masked_(as_white)](https://user-images.githubusercontent.com/10498635/89804030-88000200-db2b-11ea-8bad-b788418c3740.png)

Or just as easily switch to a different volcano, such as Vesuvius:\
<code>scene_centre = [(40.82, 14.43), 20]</code>
![03_DEM_cropped_to_area_of_interest_for_new_location_(Vesuvius](https://user-images.githubusercontent.com/10498635/89804084-9817e180-db2b-11ea-9d10-1e63dd151112.png)


And convert the Campi Flegrei DEM into a topographically correlated APS:
![Topographically_correlated_APS](https://user-images.githubusercontent.com/10498635/89804099-a0701c80-db2b-11ea-8ce2-3d1ce33daddc.png)

We can add deformation from a point (Mogi) source at a latitude and longitude of our choice, and set its depth and volume change (in m/m^3):\
<code>deformation_centre = [(40.84, 14.14), 2000 , 1e6]            </code>
![06_Deformaiton_signal](https://user-images.githubusercontent.com/10498635/89804198-be3d8180-db2b-11ea-8ecf-0571c4f87cf7.png)


And also generate a turbulent APS:

![09_Turbulent_APS_-_just_spatially_correlated_noise](https://user-images.githubusercontent.com/10498635/89804223-c72e5300-db2b-11ea-9a66-cbe1edf83b47.png)


Before combining the signals to make an interferogram:
![_Combined](https://user-images.githubusercontent.com/10498635/81292410-765a1a80-9063-11ea-9c9c-e02684adb437.png)


Or set the constituent signals' temporal behaviour to make a time series:
![Synthetic_Interferograms](https://user-images.githubusercontent.com/10498635/81292573-bae5b600-9063-11ea-84cb-fc028c1eed07.png)


<h5>Deformation Examples (New in August 2020)</h5>

If a user has Octave installed, the oct2py package can be used to call .m functions that describe deformation patterns associated with inflating sills, opening dykes, and earthquakes with a variety of source parameters.  The new example script (02_example_deformations.py) shows these in action:  

![03:_Opening_Dyke](https://user-images.githubusercontent.com/10498635/89804374-0361b380-db2c-11ea-95c3-d40c13a8415e.png)

![04:_Inflating_Sill](https://user-images.githubusercontent.com/10498635/89804379-03fa4a00-db2c-11ea-9ea1-58db049e7f9d.png)

![05:_SS_fault_earthquake](https://user-images.githubusercontent.com/10498635/89804380-0492e080-db2c-11ea-8e16-260668f2a74a.png)

![06:_Normal_fault_earthquake](https://user-images.githubusercontent.com/10498635/89804382-052b7700-db2c-11ea-866a-aeb2a39d5c39.png)

![07:_Thurst_fault_earthquake](https://user-images.githubusercontent.com/10498635/89804384-052b7700-db2c-11ea-9613-84ca68fb9fe4.png)

