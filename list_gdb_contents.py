import arcpy
import os
import sys

try:
    # Construct the absolute path to the geodatabase
    gdb_path = os.path.abspath(os.path.join("lab", "GIS5", "GIS5_datafiles", "DEM_analysis_DATA.gdb"))
    
    # Check if the geodatabase exists
    if not arcpy.Exists(gdb_path):
        print("Error: Geodatabase not found at {}".format(gdb_path))
        sys.exit(1)

    # Set the workspace environment setting
    arcpy.env.workspace = gdb_path

    print("Feature classes in {}:".format(gdb_path))
    fcs = arcpy.ListFeatureClasses()
    
    # Check if any feature classes were found
    if not fcs:
        print("  No feature classes found.")
        # Attempt to list tables as well, in case elevation points are a table
        print("\nTables in {}:".format(gdb_path))
        tbls = arcpy.ListTables()
        if not tbls:
            print("  No tables found.")
        else:
            for tbl in tbls:
                print("  - {}".format(tbl))
        sys.exit(0) # Exit cleanly if no feature classes found
    else:
        # Print found feature classes
        for fc in fcs:
            print("  - {}".format(fc))

    # Attempt to identify the contour feature class and list its fields
    potential_contour_names = ['contour_arc', 'contours', 'contour_lines', 'contour']
    contour_fc_name = None
    for name in potential_contour_names:
        # Case-insensitive check might be useful depending on data source
        if name in fcs:
            contour_fc_name = name
            break
        # Check variations if needed (e.g., different capitalization)
        # This simple check assumes exact match for now

    if contour_fc_name:
        print("\nFields in '{}':".format(contour_fc_name))
        try:
            fields = arcpy.ListFields(contour_fc_name)
            for field in fields:
                # Check for potential elevation field names
                potential_height_field = ""
                if field.name.upper() == "HOEYDE" or field.name.upper() == "ELEVATION" or field.name.upper() == "Z" or field.name.upper() == "ALTITUDE":
                     potential_height_field = " <-- Potential Height Field?"
                print("  - {0} (Type: {1}){2}".format(field.name, field.type, potential_height_field))
        except Exception as e:
            print("  Error listing fields for {}: {}".format(contour_fc_name, e))
    else:
        print("\nCould not automatically identify a contour feature class using names: {}.".format(potential_contour_names))
        print("Please examine the list above and specify the correct contour feature class name and its height field.")
        # Optionally list fields for all feature classes if contour name is uncertain
        # print("\nListing fields for all feature classes:")
        # for fc in fcs:
        #     print("\nFields in '{}':".format(fc))
        #     try:
        #         fields = arcpy.ListFields(fc)
        #         for field in fields:
        #             print("  - {0} (Type: {1})".format(field.name, field.type))
        #     except Exception as e:
        #         print("  Error listing fields for {}: {}".format(fc, e))


except ImportError:
    print("Error: ArcPy module not found. Make sure ArcGIS Pro or ArcGIS Server is installed and the Python environment is configured correctly.")
    sys.exit(1)
except Exception as e:
    print("An unexpected error occurred: {}".format(e))
    sys.exit(1)
