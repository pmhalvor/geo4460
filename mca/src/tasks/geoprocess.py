"""
3. Get popular segments w/o biking lanes
    1. (Iterative preprocessing step) Roads - lanes_classification_filtered -› find segments around these spots
    2. Full segment raster - lanes_classification_filtered (keep segments even if half has bike lane)
4. Randomized CDW 
    1. Select two random points in Oslo
    2. Find least cost routes
        1. Transverse between points, traveling over all roads
        2. Transverse between points, biased towards traveling over bike routes
        3. Transverse between points, biased towards traveling over non-bike routes
"""
