# MCA Analysis

A multi-criteria analysis (MCA) focusing on improving cycling infrastructure in Oslo, Norway. 
The analysis looks for popular routes that are currently not well connected to the existing cycling infrastructure.
In particular we:
- Identify popular routes currently missing bike lanes
- Compare relative traffic of bike lane routes versus routes without bike lanes
- (Bonus task) Run cost distance analysis between random points in the city, comparing over lap with existing bike lanes versus popular routes


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

## Workflow outline
1. Load data: as polylines and points 
    1. Segments: gdf, get_metric(metric, id=“all”), get(id), len, …
    2. Heatmap: gdf, get_activity(id), len, …
    3. Traffic: gdf, get_metric(metric, id="all”, vehicle=["all”, “bike”, “car”]), get(id), len, … 
    4. Lanes: gdf, get_classification(id), get(id), len, …
    5. AQI: gdf, get_metric(metric, id="all”), get(id), len, …
    6. Elevation: contour lines
    7. Roads: gdf, get_classification()
2. Generate feature layers:
    1. Traffic buffers (for better segment intersection)
        1. Bike
        2. Car
    2. Lanes raster/lines
    3. AQI Influence (IDW)
    4. Average speed raster/lines (from personal heat map)
    5. Segment popularity rasters/lines
        1. Athletes/age
        2. Stars/age
        3. Stars/athletes
        4. …
    6. Generate roads polygon (cost distance)
    7. Elevation raster from contour points
3. Geoprocess
    1. Get popular segments w/o biking lanes
        1. (Iterative preprocessing step) Roads - lanes_classification_filtered -› find segments around these spots
        2. Full segment raster - lanes_classification_filtered (keep segments even if half has bike lane)
    2. Randomized CDW 
        1. Select two random points in Oslo
        2. Find least cost routes
            1. Transverse between points, traveling over all roads
            2. Transverse between points, biased towards traveling over bike routes
            3. Transverse between points, biased towards traveling over non-bike routes
4. Assessment 
    1. Compare segments w/ and w/o bike lanes
        1. Against: popularity metrics, average speed, average elevation difference
        2. Find top ranking remaining segments w/ different metrics
    2. Compare CDW results between all three options
        1. Compare distances
        2. Compare average popularities 
        3. Compare PASR (giving estimate for time)
    3. Extract segments/CDW routes/road sections with highest discrepancies (prioritize better scores for non-bike routes)
5. Present results
    1. Display overlays: segments, lanes, roads, roads w/o lanes, most important CDW routes 
    2. Plot statistics comparing lane types for all metrics 
    3. Show best scoring bike-less segments/routes/roads
    4. Show best scoring bike-paths