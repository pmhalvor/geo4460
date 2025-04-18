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


def build_overlay_a(segments, roads):
    """
    Popular segments
    w/o biking lanes
    """


def build_overlay_b(overlay_a, heatmap):
    """
    Popular segments
    w/o biking lanes
    w/ high speeds
    """


def build_overlay_c(overlay_b, traffic):
    """
    Popular segments
    w/ biking lanes
    w/ high speeds
    w/ high traffic
    """


def build_overlay_d(overlay_c, slope):
    """
    Popular segments
    w/ biking lanes
    w/ high speeds
    w/ high traffic
    w/ lowest cost

    Randomized CDW calcuations should be done in this step,
    since it requires roads as restriction.
    """
