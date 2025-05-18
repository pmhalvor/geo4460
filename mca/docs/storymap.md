Further Improving Oslo's Cycling Infrastructure

A Suitability Analysis for Potential Future Bike Lanes

By Per Halvorsen

Draft

Oslo Kommune's map of cycling routes as of 2025 (original map)

Like most Scandinavian cities, Oslo is a thriving metropolis with growing ambitions to promote sustainable travel and living. Among these green initiatives is its widespread network of cycling lanes throughout the city. To further expand this infrastructure, Oslo's Street, Transport, and Parking department is looking for candidates for future bike lanes. The following suitability analysis aims to assist in this search.

Finding the Data
A handful of state-funded data sources that would traditionally serve as the foundation for such an analysis of Oslo are made freely accessible to the general public (see Kartverket, Statens Vegvesen, and Oslo Kommune). While great for surface-level analysis, these data are often quite general and sparse, failing to capture nuanced travel patterns containing specific routes or parts of the city. This can be problematic when trying to further expand on an already robust cycling infrastructure, because most of the general, low-hanging fruit have already been collected and optimized for. These data sources still serve as a great foundation for a suitability analysis, providing geographical context, such as traffic flow and elevation-based slope approximations.

Strava has over 35 million public segments around the globe.

In our analysis, we decided to also include crowd-sourced data from Strava. This gave us a more informative view of how Osloites actually cycle through their city. Granted, data from workouts posted to Strava also have their limitations. For example, not all people riding bikes through Oslo register their journeys as an activity on Strava. However, with roughly 10,000 activities being uploaded in the Oslo area on a daily basis, we assume these data to be the most conclusive of any publicly available data source, making them a great resource for our analysis. Strava makes it easier for developer-athletes to access public and private segments through their Developer API.

Building Feature Layers
Example of a raster showing average speed across Oslo.

Not all the data gathered for this analysis was ready to use, fresh out of the box. As with most GIS projects, each data set needed to be projected to the correct coordinate system and rasterized or vectorized into some geographical representation, whether that be rasters, polylines, or geo-points.

A raster is a 2D layer of a map where every pixel represents some value for that particular location. Typically, information like expected rainfall or cloud coverage is best represented as a raster. In our analysis, we built rasters to represent: average cycling speed, expected traffic, elevation, and slope.

Using the fine-grained activity details endpoint, we were able to extract the exact speeds traveled at points along all the activities of a single user over the last 5 years. These speed points were then used to build the average speed raster.

Thanks to Statens Vegvesen's Traffikdata API, we were also able to extract the hourly traffic flow for all the traffic registration points around the city. From these data, we could build a raster representing expected traffic per time of day. To the left, we see the raster created from the averaged traffic data between 16:00-24:00 in the month of May 2024.

The other important data format used for our analysis was polylines. These are lines connected to geographical locations with information along the lines. Our data had two polyline-based data sources: registered roads/bike lanes and Strava activities/segments.

The map presented at the top of this article showed the bike lanes of Oslo, along with some classifications. Opening the sidebar in that figure, you can see the different categories of bike lanes that are currently registered and maintained by Oslo Kommune. For our analysis, we considered bike lanes that were mixed with automotive traffic as candidates for future bike lanes, as well as roads with no registered bike lanes.

Segments colored by popularity, lighter ones being more popular.

Pulling segment data from the Strava API, we were able to build a polyline layer representing all the public segments downloaded for our project. Due to rate limits and engineering restrictions around the Strava API, not every single segment in Oslo could be considered for this analysis. We did, however, try our best to scrape as many public segments as possible, ensuring proper coverage of the city.

Combining the Data
With all the data in place, in each of their corresponding formats, we needed to join the information to make a data-driven decision on where future bike lanes should be prioritized. For simplicity, we decided that all the segments downloaded would be our potential candidates to one day be converted into a bike lane. The top-ranking candidates to be proposed would be found based on 5 requirements:

- There could not be a current bike lane overlapping with the segment.
- The segment would have to be "popular".
- The average speed heat map should show relatively high speeds from previous rides.
- There should be a considerably high amount of traffic in the area.
- The segment should relate to a slope-based low cost, i.e., few inclines & easily bikeable.

There were a few ways to define popularity. Every segment has a creation date, total number of athletes to complete the segment, total number of efforts for the segment, a Local Legend (the person who completed the segment the most times in the last 90 days), and a Leaderboard (athletes who have completed the segment ranked according to their fastest times). We decided a metric based on the total number of efforts relative to the segment's age gave the best representation of popularity to suit our use case.

All the layers were normalized, giving values between 0 and 1. This made tuning these parameters easier, helping decide which data should be considered more important. If thresholds were set too high, too many segments would have ended up in the final proposed list of candidates, giving inconclusive results for the analysis.

The most important feature we found ended up being the segment's popularity. This was a partially intentional decision, as we want the future bike lane candidates to be useful for as many people as possible. Speed was the second most important factor, followed by traffic, then cost. Below, you can find an interactive map with the candidate segments that made it through each requirement filter. You will also find the polylines representing all the segments considered for the analysis.

Suitability analysis results for future cycling lanes in Oslo, Norway.

What do you think? Are there any routes here you would also want to see a bike lane for in the future? Feel free to reach out if you have any questions or further recommendations!

This analysis was our 2025 semester project for GEO4460 at the University of Oslo.
Source code for this analysis can be found at: github.com/pmhalvor/geo4460/

N50 Map Data

Kartverket

Strava Segment & Activity Data

Strava Developer API

Traffic Registration Data

Statens Vegvesen

Bike Routes in Oslo

Oslo Kommune
