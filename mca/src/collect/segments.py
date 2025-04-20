import numpy as np

from src.strava.explore import explore_segments, store_segments
from src.strava.locations import locations
from src.traffic.stations import get_oslo_stations


def calculate_km_to_degree_at_latitude(km, latitude):
    """Convert kilometers to degrees at a given latitude"""
    # 1 degree of latitude is approximately 111.32 km
    latitude_degree = km / 111.32

    # 1 degree of longitude varies with latitude
    longitude_degree = km / (111.32 * np.cos(np.radians(latitude)))

    return float(latitude_degree), float(longitude_degree)


def create_bounds_around_point(point, areas_km2=[1, 2, 5]):
    """Create bounds of different sizes around a point"""
    lon, lat = point.x, point.y
    bounds_list = []

    for area in areas_km2:
        side_length = np.sqrt(area)

        lat_offset, lon_offset = calculate_km_to_degree_at_latitude(
            side_length / 2, lat
        )

        # Create bounds [min_lat, min_lng, max_lat, max_lng]
        bounds = [
            lat - lat_offset,  # min_lat
            lon - lon_offset,  # min_lng
            lat + lat_offset,  # max_lat
            lon + lon_offset,  # max_lng
        ]
        bounds_list.append(bounds)

    return bounds_list


def explore_segments_with_multiple_bounds(gdf, area_sizes=[1, 2, 5]):
    """Explore segments around points with multiple bound sizes"""
    all_segments = []
    unique_segment_ids = set()

    for idx, row in gdf.iterrows():
        point = row.points
        bounds_list = create_bounds_around_point(point, area_sizes)

        for i, bounds in enumerate(bounds_list):
            print(f"Exploring {area_sizes[i]} km² area around point {idx}")
            result = explore_segments(bounds).get("segments", [])

            new_segments = []
            for segment in result:
                segment_id = segment.get("id")
                if segment_id not in unique_segment_ids:
                    unique_segment_ids.add(segment_id)
                    new_segments.append(segment)

            print(f"Found {len(new_segments)} new segments")
            all_segments.extend(new_segments)

    return all_segments


if __name__ == "__main__":
    oslo_stations = get_oslo_stations()

    oslo_bounds = locations["oslo"]["bounds"]
    oslo_segments = explore_segments(oslo_bounds).get("segments", None)

    segments = explore_segments_with_multiple_bounds(
        oslo_stations, area_sizes=[3, 5, 10]
    )
    store_segments(segments)
    print(f"Nice exploring! 🔎" f" Found {len(segments)} new segments!")
