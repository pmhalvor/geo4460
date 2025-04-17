"""
2. Generate rasters:
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
    7. Create Oslo mask (from random CDW)
    8. Elevation DEM
    9. Slope Raster 
    10. PASR: Predicted average speed raster (Co-Kriging or similar using slope as external variable)

"""
