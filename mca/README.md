# MCA Analysis

A multi-criteria analysis (MCA) focusing on improving cycling infrastructure in Oslo, Norway. 
The analysis looks for popular routes that are currently not well connected to the existing cycling infrastructure.


The research questions we'll aim to solve area:
- (General) Can we find ways to improve cycling infrastructure around Oslo using traffic & Strava data?
- (Specific) What are some examples of popular Strava segments that do not have bike lanes?
- (Specific) How does segment popularity vary between segments over bike lanes versus those not on bike lanes?
- (Specific) Is there a correlation between Strava segment popularity and cost? Traffic? Speed?


## Contents
- [Project description](GIS5_DEM_analysis.pdf)
- [Report](Halvorsen_GEO4460_GIS5_DEM_generation.pdf)
- [Source code](src/)
    - [Config](src/config.py): Configuration file with parameters for the project
    - [Tasks](tasks/): Each step of the project, incluing class and task wrapper function
    - [Workflow](src/workflow.py): Module that executes the tasks from the project description


## How to run
Navigate to `lab/GIS5` and run the following commands:
```bash
make install
make run
```

It would be best to run the pipeline in a virtual environment.

Using `conda`:
```bash
conda create -n mca python=3.11
conda activate mca
make install
make run
```

Or `uv`:
```bash
uv sync
uv run python -m src.workflow
```

The workflow leverages `dask` to parallelize the featurization process. 
The enables scalability and allows for faster processing of large datasets.

The pipeline is designed to be modular, allowing for easy addition or removal of tasks as needed.
Every feature can be built indivdually by calling the corresponding feature layer as a module:
```bash
python -m src.tasks.features.heatmap
```

Each layer produces a Folium map, to quickly inspect the results of the analysis.
We haven't automated viewing of these maps, for faster iterations.
However example results have been included in the `output/` folder.





## Workflow outline
1. Prepare data: convert data from `data/` to gdf w/ polylines or points
    1. Segments: gdf, get_metric(metric, id=“all”), get(id), len, …
    2. Heatmap: gdf, get_activity(id), len, …
    3. Traffic: gdf, get_metric(metric, id="all”, vehicle=["all”, “bike”, “car”]), get(id), len, … 
    4. Elevation: contour lines
    5. Roads: gdf, get_classification()
2. Generate feature layers:
    1. Roads:
        1. Roads polylines (needed for CDW analysis)
        2. Bike lane polylines (w/ lane classification if possible)
        3. Roads w/o bike lanes (final layer to be used downtream)
    2. Segment popularity rasters/lines
        1. Aggregate column data into relevant metrics:
            1. Athletes/age
            2. Stars/age
            3. Stars/athletes
        2. Aggregate metrics over all polylines (average)
        3. Create raster from aggregated metric polylines 
    3. Average speed raster (from personal heat map)
        1. Start w/ activities df including polylines w/ speed points
        2. Build speed points layer
        3. Create raster from speed points (doppler shift expected on two way roads up hill)
    4. Traffic buffers (for better segment intersection)
        1. Traffic stations as points w/ flux metrics
        2. Create buffers around traffic stations
        3. Create raster from traffic buffers
    5. Elevation & slope rasters:
        1. Contour lines as points
        3. Create elevation raster from contour points 
        4. Create slope raster from elevation raster (use in cost function analysis)
    6. Cost function raster
        1. Elevation for slope raster
        2. Roads for road polylines 
        3. (Questionable due to double representation) Heatmap for average speed raster (higher speed = more reward)
        Alternatives: (choose based on implementation)
            1. Randomized CDW 
                1. Select two random points in Oslo
                2. Find least cost routes
                    1. Transverse between points, traveling over all roads
                    2. Transverse between points, biased towards traveling over bike routes
                    3. Transverse between points, biased towards traveling over non-bike routes
            2. Simple cost funciton raster
                1. Build cost layer raster with 
                    1. slope as cost function
                    2. high speed as rewards
                    3. roads as restrictions
3. Combine features:
    1. Overlay A: Popular segments w/o biking lanes
        1. (Iterative preprocessing step) Roads - bike_lanes -> find segments around these spots
        2. All segment lines layer - bike_lanes (keep all segments with part outside of bike lanes)
    2. Overlay B: Popular segments w/o biking lanes w/ high avg speeds
        1. Overlay A + average speed raster
    3. Overlay C: Popular segments w/o biking lanes w/ high avg speeds + high traffic
        1. Overlay B + traffic buffer raster
    4. Overlay D: Popular segments w/o biking lanes w/ high avg speeds + high traffic + high cost
        1. Overlay C + cost function raster
4. Evaluate findings
    1. Compare segments w/ and w/o bike lanes
        1. Against: popularity metrics, average speed, average elevation difference
        2. Find top ranking remaining segments w/ different metrics
    2. Compare costs of popular segments
    3. Find segments to recommend from final map
5. Present results
    1. Display overlays: segments, lanes, roads, roads w/o lanes, most important CDW routes 
    2. Plot statistics comparing lane types for all metrics 
    3. Show best scoring bike-less segments/routes/roads
    4. Show best scoring bike-paths


## Data

- [x] Strava segments (geojson)
    - id, polyline 
    - `mca/data/segments/segments_oslo.geojson`
- [x]  Strava activities (geojson)
    - polyline, speed line, activity id
- [x]  N50 (fgdb, +++)
    - [x]  Roads
        - [x]  roads
        - [x]  bike lanes
    - [x]  Elevation
- [ ]  Traffic (json)
    - [ ]  Cars
    - [x]  Bikes 
        - `data/traffic/all-oslo-bikes-day_20240101T0000_20250101T0000.csv`
- [ ]  Bike lane classifications (maybe)

<!-- TODO: update w/ source references -->
<!-- TODO update with correct dataset names -->

## Helper functions
Display info on rasters (tif layers)
```bash
gdalinfo output/mca_20250421_1749_heatmap/average_speed.tif
```

Display layers and fields of N50 Oslo GeoDataBase
```bash
ogrinfo -so data/Basisdata_03_Oslo_25833_N50Kartdata_FGDB.gdb


ogrinfo -so data/Basisdata_03_Oslo_25833_N50Kartdata_FGDB.gdb "N50_Samferdsel_senterlinje"
```

Get unique values of a field in a layer
```bash
ogrinfo -sql "SELECT DISTINCT typeveg FROM N50_Samferdsel_senterlinje" data/Basisdata_03_Oslo_25833_N50Kartdata_FGDB.gdb
```
