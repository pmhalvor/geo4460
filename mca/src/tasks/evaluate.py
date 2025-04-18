import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd

# Import plotting libraries if needed, e.g., matplotlib, seaborn
# import matplotlib.pyplot as plt
# import seaborn as sns

# Local imports
from src.config import AppConfig
from src.utils import load_vector_data, save_vector_data

logger = logging.getLogger(__name__)


# --- Docstring from original file (for reference) ---
"""
Evaluation Steps:

1. Evaluate findings
    1. Compare segments w/ and w/o bike lanes (Requires loading original segments/lanes)
        1. Against: popularity metrics, average speed, average slope, average cost
           (Requires these stats to be added during combine_features)
        2. Find top ranking segments w/ different metrics (from Overlay D)
    2. Compare CDW results (If CDW analysis was performed - currently not implemented)
        1. Compare distances
        2. Compare average popularities
        3. Compare PASR (Potential Anisotropic Shortest Route - if calculated)
    3. Find segments to recommend from final map (Overlay D, ranked)
2. Present results
    1. Display overlays: (Handled by external GIS software/visualization)
       - segments, lanes, roads, roads w/o lanes, final overlay D
    2. Plot statistics comparing lane types for all metrics
    3. Show best scoring bike-less segments/routes/roads (Save top N)
    4. Show best scoring bike-paths (Requires comparison group)
"""


# Helper function
def _get_full_output_path(settings: AppConfig, key: str) -> Path:
    """Helper to get a full output path from settings."""
    filename = getattr(settings.output_files, key)
    return settings.paths.output_dir / filename


# --- Evaluation Functions ---


def compare_segment_groups(settings: AppConfig, overlay_d_gdf: gpd.GeoDataFrame):
    """
    Compares segments with and without bike lanes based on available metrics.

    Requires loading original prepared segments and lanes data.
    Assumes metrics (popularity, speed, cost, slope) have been added to
    overlay_d_gdf during the combine_features step.

    Args:
        settings: Application config.
        overlay_d_gdf: GeoDataFrame from Overlay D (segments recommended for improvement).
    """
    logger.info(
        "--- Comparing Segment Groups (Overlay D vs All/Bike Lane Segments) ---"
    )

    # Load necessary comparison data
    try:
        all_segments_path = _get_full_output_path(settings, "prepared_segments_gpkg")
        lanes_path = _get_full_output_path(settings, "prepared_bike_lanes_gpkg")

        if not all_segments_path.exists():
            logger.warning(
                "Prepared segments file not found. Cannot perform full comparison."
            )
            return None  # Or return partial stats based only on overlay_d_gdf

        all_segments_gdf = load_vector_data(all_segments_path)
        # TODO: Add metrics (popularity, etc.) to all_segments_gdf if not already present

        bike_lane_segments_gdf = None
        if lanes_path.exists():
            lanes_gdf = load_vector_data(lanes_path)
            # Find segments that intersect with bike lanes
            # Ensure CRS match
            if all_segments_gdf.crs != lanes_gdf.crs:
                lanes_gdf = lanes_gdf.to_crs(all_segments_gdf.crs)
            bike_lane_segments_gdf = gpd.sjoin(
                all_segments_gdf, lanes_gdf, how="inner", predicate="intersects"
            )
            segment_id_field = settings.input_data.segment_id_field
            if segment_id_field in bike_lane_segments_gdf.columns:
                bike_lane_segments_gdf = bike_lane_segments_gdf.drop_duplicates(
                    subset=[segment_id_field]
                )
            logger.info(
                f"Identified {len(bike_lane_segments_gdf)} segments intersecting bike lanes."
            )
        else:
            logger.warning(
                "Bike lanes file not found. Cannot compare against bike lane segments."
            )

    except Exception as e:
        logger.error(f"Error loading data for comparison: {e}")
        return None

    # --- Perform Statistical Comparisons ---
    # Example: Compare average popularity metric
    # This requires the metric columns (e.g., 'avg_popularity_stat', 'avg_speed_stat')
    # to exist in the GeoDataFrames, added during combine_features.
    comparison_stats = {}
    metrics_to_compare = [
        "avg_popularity_stat",
        "avg_speed_stat",
        "avg_cost_stat",
        "avg_slope_stat",
    ]  # Example metric names

    for metric in metrics_to_compare:
        if metric in overlay_d_gdf.columns:
            overlay_d_mean = overlay_d_gdf[metric].mean()
            comparison_stats[f"overlay_d_{metric}_mean"] = overlay_d_mean
            logger.info(f"Overlay D Mean {metric}: {overlay_d_mean:.2f}")

            if metric in all_segments_gdf.columns:
                all_mean = all_segments_gdf[metric].mean()
                comparison_stats[f"all_segments_{metric}_mean"] = all_mean
                logger.info(f"All Segments Mean {metric}: {all_mean:.2f}")

            if (
                bike_lane_segments_gdf is not None
                and metric in bike_lane_segments_gdf.columns
            ):
                lane_mean = bike_lane_segments_gdf[metric].mean()
                comparison_stats[f"bike_lane_segments_{metric}_mean"] = lane_mean
                logger.info(f"Bike Lane Segments Mean {metric}: {lane_mean:.2f}")
        else:
            logger.warning(
                f"Metric '{metric}' not found in Overlay D GDF. Cannot compare."
            )

    # TODO: Add more sophisticated statistical tests (t-tests, etc.) if needed.

    # --- Save Comparison Statistics ---
    stats_df = pd.DataFrame([comparison_stats])  # Convert dict to DataFrame
    output_csv_path = _get_full_output_path(settings, "evaluation_stats_csv")
    try:
        stats_df.to_csv(output_csv_path, index=False)
        logger.info(f"Comparison statistics saved to: {output_csv_path}")
    except Exception as e:
        logger.error(f"Failed to save comparison stats CSV: {e}")

    # --- Generate Plots (Optional) ---
    # Example: Box plots comparing distributions
    # plots_dir = _get_full_output_path(settings, "evaluation_plots_dir")
    # plots_dir.mkdir(parents=True, exist_ok=True)
    # for metric in metrics_to_compare:
    #     # Create combined dataframe for plotting
    #     # ... plot generation logic using matplotlib/seaborn ...
    #     # plt.savefig(plots_dir / f"{metric}_comparison_boxplot.png")
    #     logger.warning(f"Plot generation for {metric} not implemented.")
    logger.warning("Comparison plot generation not implemented.")

    return comparison_stats


def rank_and_save_recommendations(settings: AppConfig, overlay_d_gdf: gpd.GeoDataFrame):
    """
    Ranks the segments in Overlay D based on specified criteria and saves the top N.

    Args:
        settings: Application config.
        overlay_d_gdf: GeoDataFrame from Overlay D.
    """
    logger.info("--- Ranking and Saving Top Recommended Segments ---")

    if overlay_d_gdf is None or overlay_d_gdf.empty:
        logger.warning(
            "Overlay D GeoDataFrame is empty or None. Cannot rank recommendations."
        )
        return

    # TODO: Define ranking criteria. This might involve combining multiple metrics.
    # Example: Rank by a primary metric (e.g., popularity) descending,
    #          then maybe by cost ascending as a tie-breaker.
    # Assumes metric columns exist from combine_features step.
    primary_metric = "avg_popularity_stat"  # Example
    secondary_metric = "avg_cost_stat"  # Example

    if primary_metric not in overlay_d_gdf.columns:
        logger.warning(
            f"Primary ranking metric '{primary_metric}' not found. Cannot rank."
        )
        # Save the unranked Overlay D as the recommendations for now
        ranked_gdf = overlay_d_gdf
    else:
        sort_columns = [primary_metric]
        ascending_order = [False]  # Descending for popularity

        if secondary_metric in overlay_d_gdf.columns:
            sort_columns.append(secondary_metric)
            ascending_order.append(True)  # Ascending for cost
        else:
            logger.warning(f"Secondary ranking metric '{secondary_metric}' not found.")

        logger.info(
            f"Ranking segments by: {sort_columns} (Ascending: {ascending_order})"
        )
        ranked_gdf = overlay_d_gdf.sort_values(
            by=sort_columns, ascending=ascending_order
        )

    # Select Top N recommendations
    top_n = settings.processing.top_n_recommendations
    recommended_gdf = ranked_gdf.head(top_n)
    logger.info(f"Selected top {len(recommended_gdf)} recommended segments.")

    # Save recommendations
    output_gpkg_path = _get_full_output_path(settings, "recommended_segments_gpkg")
    try:
        save_vector_data(recommended_gdf, output_gpkg_path, driver="GPKG")
        logger.info(f"Top recommended segments saved to: {output_gpkg_path}")
    except Exception as e:
        logger.error(f"Failed to save recommended segments GeoPackage: {e}")


# --- Main Task Function ---


def evaluate_task(combined_results: dict, settings: AppConfig):
    """
    Executes the evaluation workflow.

    Args:
        combined_results (dict): Dictionary containing results from combine_features_task.
                                 Expected key: 'overlay_d' (path to final overlay).
        settings (AppConfig): Application configuration object.
    """
    logger.info("--- Starting Task 3: Evaluate Results ---")

    overlay_d_path = combined_results.get("overlay_d")

    if not overlay_d_path or not overlay_d_path.exists():
        logger.error(
            "Final overlay (Overlay D) path not found in combined_results or file does not exist. Cannot perform evaluation."
        )
        return

    try:
        overlay_d_gdf = load_vector_data(overlay_d_path)
        logger.info(
            f"Loaded Overlay D GeoDataFrame with {len(overlay_d_gdf)} features for evaluation."
        )

        # Step 1: Compare segment groups (Overlay D vs All/Bike Lane segments)
        # Note: This requires metrics to be added in combine_features
        compare_segment_groups(settings, overlay_d_gdf)

        # Step 2: Rank and save top recommendations from Overlay D
        rank_and_save_recommendations(settings, overlay_d_gdf)

        # Step 3: Add any other evaluation steps (e.g., CDW comparison if implemented)
        logger.warning("CDW comparison not implemented.")

    except Exception as e:
        logger.error(
            f"An error occurred during the evaluation task: {e}", exc_info=True
        )

    logger.info("--- Task 3: Evaluate Results Completed ---")
