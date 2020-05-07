# Synthetic-interferograms
Generate synthetic interferograms that are similar to those produced by the Sentinel-1 satellites.  

Users can select any subaerial volcano that is covered by the SRTM3 digital elevaiton model (DEM), and create an interferogram for there containing deformation due to a point (Mogi) source, a topogrpahically correlated atmospheric phase screen (APS), and a turbulent APS.  Temporal behaviour for these signals can also be set to create time series of interferograms.  

If you use this software in your research, please cite the [2019 JGR:SE paper](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519) that describes its generation.  

<br>

<h5>Installation:</h5>
A suitable python environment can be created using the conda command:<br>
<code>conda env create --file synthetic_interferogams.yml</code>

<br>

<h5>Examples:</h5>

We can set a scene of 20km side length centered on Campi Flegrei using just its latitude and longitude:\
<code>scene_centre = [(40.84, 14.14), 20]                                                </code>

![02_DEM_Cropped_to_area_of_interest](https://user-images.githubusercontent.com/10498635/81289416-4f4d1a00-905e-11ea-8ea0-79bdd7ebe365.png)

Or just as easily switch to a different volcano, such as Vesuvius:\
<code>scene_centre = [(40.82, 14.43), 20]</code>
![03_DEM_cropped_to_area_of_interest_for_new_location_(Vesuvius](https://user-images.githubusercontent.com/10498635/81289455-5c6a0900-905e-11ea-904d-dff238823018.png)

And convert the Campi Flegrei DEM into a topographically correlated APS:
![08_Topographically_correlated_APS](https://user-images.githubusercontent.com/10498635/81289548-84596c80-905e-11ea-8100-27d6e9c21c01.png)

We can add deformation from a pont (Mogi) source at a latitude and longitude of our choice, and set its depth and volume change (in m):\
<code>deformation_centre = [(40.84, 14.14), 2000 , 1e6]            </code>
![06_Deformaiton_signal](https://user-images.githubusercontent.com/10498635/81289526-7ad00480-905e-11ea-87c2-59ef66eec217.png)

And also generate a turbulent APS:

![09_Turbulent_APS_-_just_spatially_correlated_noise](https://user-images.githubusercontent.com/10498635/81289549-84f20300-905e-11ea-9731-acbd1f73a865.png)

Before combining the signals to make an interferogram:
![_Combined](https://user-images.githubusercontent.com/10498635/81292410-765a1a80-9063-11ea-9c9c-e02684adb437.png)


Or set the constituent signals' temporal behaviour to make a time series:
![Synthetic_Interferograms](https://user-images.githubusercontent.com/10498635/81292573-bae5b600-9063-11ea-84cb-fc028c1eed07.png)

