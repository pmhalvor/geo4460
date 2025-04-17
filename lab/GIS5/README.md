# GIS 5 - DEM analysis

Pipeline for DEM generation and analysis. 
Compares Natural Neighbor interpolation, TIN interpolation, and ANUDEM interpolation.
Evaluation includes a quantative method (RMSE) and visual analyses (hillshade, slope, profile).

Limitations to open source software hinder the data preparation and ANUDEM generation steps.
These steps were instead performed in isolation from the pipeline in ArcGIS Pro.

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
conda create -n gis5 python=3.11
conda activate gis5
make install
make run
```

Or `uv`:
```bash
uv sync
uv run python -m src.workflow
```



