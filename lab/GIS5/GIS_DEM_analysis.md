# GIS-5: DIGITAL TERRAIN MODELS  {-}
 
In terrain modelling, we often need to generate a grid-based DEM. This can be done from 
aerial photographs, but often we generate the DEM from existing contours and/or point 
data. When gridding contour lines one common problem is that, we have a lot of data 
along the lines, and no data in between. When trying to interpolate the data to a 
continuous surface we often end up with stepped surfaces, which produce numerous 
errors when used quantitatively. 
 
In this assignment, we will generate DEMs based on contour lines using two different 
approaches. You should compare both approaches and assess the data quality. The scale 
and the quality of your input data will be inherited to your DEM. It is therefore important 
that you describe both the input data and the quality and error of the resulting DEM. From 
the DEM we can calculate terrain parameters and do different types of analysis. Terrain 
parameters are derivatives of the elevation, the simplest ones, such as slope (first 
derivative) and curvatures (second derivatives) are the most commonly used.

# Data
Found in `GIS5_datafiles/`

# TECHNICAL TASKS    
## DEM GENERATION   
The first approach is to create a TIN from the contours, and then interpolate a grid. You use 
three data layers, (1) contour lines (“contour_arc”), (2) river lines (“rivers_arc”), (3) lakes 
“lakes_polygon”). Contour lines contain your information about altitudes, rivers are break 
lines indicating local minima in the terrain, and lakes are pits, indicating a flat area with 
constant altitude. 
 
### TIN FROM CONTOURS: 
 
1)  Add your datasets (contour lines, river lines and lakes) into a new project.   
2)  Go to View → Geoprocessing → Toolboxes → 3D Analyst Tools → TIN Dataset → 
Create TIN. Select layers you want to include in the TIN generation, add each layer  
as Input Feature Class separately  . The "Height field" is the field that 
specifies the source of elevation values for the features; "HOEYDE" for elevation 
(means elevation), use <none> for rivers and lakes. The "Type" (surface feature 
type) defines how the geometry imported from the features is incorporated into the 
triangulation for the surface. The contour lines should be added as "mass points", 
the streams as "hard line" and the lakes as "hard replace". 
3)  Convert to raster: The resulting TIN is then converted into a grid by the command 
"TIN to Raster" under "Toolboxes→ 3D Analyst Tools → TIN Dataset → Conversion 
→ TIN to Raster". Select your TIN as Input TIN, save the grid under the GEO3460 
folder in your home area, Sampling distance = Cell Size, Sampling Value = 50, Z 
Factor = 1, else default values. It is recommended to add extension ".tif" to the 
output raster, but note that you cannot save it in the geodatabase. 

### DEM USING TOPOGRID:  
 
Another approach to DEM generation is to 
use the ANUDEM-algorithm available from 
the  function  called  "Topo  to  Raster"  in 
"Toolboxes → 3D Analyst Tools → Raster → 
Interpolation → Topo to Raster". Use the 
same input feature data as before and output 
cell size of 50 m. It is recommended to add 
extension ".tif" to the output raster. (Note! To 
compare Topogrid and TIN you need to have 
the same extent. In the “Output Extrend" you 
need  to put the newly  created  TIN-raster 
using “Same As Layer”). 

The "Topo to raster" is based on an algorithm proposed by Hutchinson (1989) and produces 
a hydrologically consistent DEM (streams (rivers) are local minima, pits are removed). The 
interpolator is a local spline interpolator, iteratively adapting an elevation model to the data 
points. Remember to change field and type for each dataset. You can read more about the 
algorithm in the ArcGIS Pro documentation (https://pro.arcgis.com/en/pro-app/latest/tool-
reference/3d-analyst/how-topo-to-raster-works.htm)  and  in  the  following  paper: 
Hutchinson, M.F., 1989. A new procedure for gridding elevation and stream line data with 
automatic removal of spurious pits. Journal of Hydrology, 106: 211-232.

### CALCULATE RMSE, CREATE COUNTOURS, HILLSHADE AND USE RASTER CALCULATOR: 
 
1)   
Now you can assess the quality of 
your two (or more) DEMs by 
calculating the RMSE between your 
DEM and measured elevation points. 
Use the elevation points 
(“elevationp_point”) to do this. Use 
"Zonal statistics as Table" from the 
"Toolboxes → Spatial Analyst Tools → 
Zonal", to find the DEM value at each 
point for the TIN and the topoGRID 
raster. 
 
   
Then you can compare the DEM value 
with the measured elevation in the 
point. Join the elevationp_point data 
set on the tables using the "FID"-field. 
Right-click on elevationp_point layer, 
choose "Joins and Relates" and then 
"Add Join".

Export Table to Excel and compare 
each elevation value and calculate 
RMSE using Excel or other suitable 
software.  
 
  Note that you have to do Zonal Statistics and Join Data for both the TIN and 
  TopoGRID. Before you join with the new data, remove the join with the previous 
  table: right-click on elevation_point → “Joins and Relates” → “Remove All Joins”. 

2)  You can also calculate contour lines from the DEMs ("Toolboxes → Spatial Analyst 
Tools → Surface → Contour"), and compare the calculated contours with the 
original contours visually. In order to display only contours at each e.g. 100 m 
interval (and you chose 10 m as contour interval before), use the remainder 
(modulo, in ArcGIS Pro: "Mod") equal to 0: 1) right-click a chosen layer, go to 
"Properties" and "Definition Query", 2) Add an SQL query e.g. “Mod( Contour , 100) 
= 0” (i.e. remainder after division by 100 is 0).  
 
3)  Furthermore, you should calculate a hillshade map ("Toolboxes → Spatial Analyst 
Tools → Surface → Hillshade") and investigate the result visually for artifacts, stripes 
or other errors.  
 
4)  To compare two or more grids you can use map algebra. Open the "Raster 
Calculator" from the Toolboxes ("Spatial Analyst Tools → Map Algebra → Raster 
Calculator"). With this you can calculate new grids based on existing ones (e.g. 
subtracting one grid from another gives you the difference between the two e.g. 
between TINgrid and TOPOgrid). 


## CALCULATE SLOPE AND MAKE A PROFILE ANALYSIS    
 
### CALCULATE SLOPE: 
 
Calculate the terrain parameter slope. It is available from the "Spatial Analyst Tools → 
Surface → Slope". Include a figure of this in your report. Remember to zoom in on 
interesting areas before creating the figure.    
 
### PROFILE ANALYSIS: 
 
We often want to make a topographic profile of an area. First, create a line where you want 
the profile. You need to create a shapefile/feature class with a polyline geometry type and 
digitize a transect line. To extract profile lines, use "Stack Profile" in "Toolboxes → 3D 
Analyst Tools → 3D Intersections → Stack Profile". Run the tool separately for TINgrid and 
TOPOgrid. The tool yields a table, which can be exported to e.g. Excel where a profile graph 
can be made. Create some profiles in the main valley (e.g. Visdalen), or on a glacier (follow 
the contour lines up-glacier).

## REPORT 
 
As always, source your own data! The N50 kartdata should have everything you need if you 
chose an area in Norway. Crop the data to a small rectangular area of interest, as working 
with a whole region would be impractical. 
Write a report of maximum 5 pages, where you include the points below additionally to 
what have been explained earlier in the exercise: 
 
1)  Describe the two methods you used to generate your terrain models and include a 
quality assessment of your results (using RMSE from measured elevation points). 
Discuss issues related to input data, rivers as local minima?, interpolation, errors etc.   2)  Show where the models differ and discuss why.  
3)  Are there differences between slope calculated from the TIN-based and the 
TOPOGRID-based DEM? Evaluate. How does the profile analysis look for both 
terrain models? Explain challenges with the methods. Use figures and flow charts 
where appropriate and try to avoid specific references to the software (of the kind: 
"To generate a slope grid you must click the "Slope" button...").