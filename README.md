# Synthetic-interferograms
Generate synthetic interferograms that are similar to those produced by the Sentinel-1 satellites.  

Users can select any subaerial volcano that is covered by the SRTM3 digital elevaiton model (DEM), and create an interferogram for there containing deformation due to a point (Mogi) source, a topogrpahically correlated atmospheric phase screen (APS), and a turbulent APS.  Temporal behaviour for these signals can also be set to create time series of interferograms.  

Installation:
A suitable python environment can be created using the conda command:
conda env create --file synthetic_interferogams.yml

Examples:

![02_DEM_Cropped_to_area_of_interest](https://user-images.githubusercontent.com/10498635/81289416-4f4d1a00-905e-11ea-8ea0-79bdd7ebe365.png)
