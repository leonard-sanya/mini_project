"""
Access module for the fynesse framework.

This module handles data access functionality including:
- Data loading from various sources (web, local files, databases)
- Legal compliance (intellectual property, privacy rights)
- Ethical considerations for data usage
- Error handling for access issues

Legal and ethical considerations are paramount in data access.
Ensure compliance with e.g. .GDPR, intellectual property laws, and ethical guidelines.

Best Practice on Implementation
===============================

1. BASIC ERROR HANDLING:
   - Use try/except blocks to catch common errors
   - Provide helpful error messages for debugging
   - Log important events for troubleshooting

2. WHERE TO ADD ERROR HANDLING:
   - File not found errors when loading data
   - Network errors when downloading from web
   - Permission errors when accessing files
   - Data format errors when parsing files

3. SIMPLE LOGGING:
   - Use print() statements for basic logging
   - Log when operations start and complete
   - Log errors with context information
   - Log data summary information

4. EXAMPLE PATTERNS:
   
   Basic error handling:
   try:
       df = pd.read_csv('data.csv')
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
   
   With logging:
   print("Loading data from data.csv...")
   try:
       df = pd.read_csv('data.csv')
       print(f"Successfully loaded {len(df)} rows of data")
       return df
   except FileNotFoundError:
       print("Error: Could not find data.csv file")
       return None
"""

from typing import Any, Union
import pandas as pd
import logging
import os
import datetime
import osmnx as ox

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def data() -> Union[pd.DataFrame, None]:
    """
    Read the data from the web or local file, returning structured format such as a data frame.

    IMPLEMENTATION GUIDE
    ====================

    1. REPLACE THIS FUNCTION WITH YOUR ACTUAL DATA LOADING CODE:
       - Load data from your specific sources
       - Handle common errors (file not found, network issues)
       - Validate that data loaded correctly
       - Return the data in a useful format

    2. ADD ERROR HANDLING:
       - Use try/except blocks for file operations
       - Check if data is empty or corrupted
       - Provide helpful error messages

    3. ADD BASIC LOGGING:
       - Log when you start loading data
       - Log success with data summary
       - Log errors with context

    4. EXAMPLE IMPLEMENTATION:
       try:
           print("Loading data from data.csv...")
           df = pd.read_csv('data.csv')
           print(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
           return df
       except FileNotFoundError:
           print("Error: data.csv file not found")
           return None
       except Exception as e:
           print(f"Error loading data: {e}")
           return None

    Returns:
        DataFrame or other structured data format
    """
    logger.info("Starting data access operation")

    try:
        # IMPLEMENTATION: Replace this with your actual data loading code
        # Example: Load data from a CSV file
        logger.info("Loading data from data.csv")
        df = pd.read_csv("data.csv")

        # Basic validation
        if df.empty:
            logger.warning("Loaded data is empty")
            return None

        logger.info(
            f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns"
        )
        return df

    except FileNotFoundError:
        logger.error("Data file not found: data.csv")
        print("Error: Could not find data.csv file. Please check the file path.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        print(f"Error loading data: {e}")
        return None


def show_copyright_info() -> None:
    """
    Displays the MIT License for this project.
    """
    license_text = """
    ====================================================
                       MIT License
    ====================================================
    Copyright (c) 2025 Leonard Sanya

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    ====================================================
    """
    print(license_text)


def load_datasets(raw_url: str) -> pd.DataFrame:
    """
    Load a CSV dataset directly from a GitHub raw URL.
    """
    print(f"Loading dataset from {raw_url} ...")

    try:
        df = pd.read_csv(raw_url)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_url, encoding="ISO-8859-1")

    print("Dataset loaded:", df.shape)
    return df


def get_feature_vector(
    latitude, longitude, box_size_km=2, features=None, all_features=None
):
    """
    Return a consistent feature vector as a dict, even if OSM query fails
    or no features are found.
    """
    # Construct bbox
    box_width = box_size_km / 111
    box_height = box_size_km / 111
    north = latitude + box_height
    south = latitude - box_height
    west = longitude - box_width
    east = longitude + box_width
    bbox = (west, south, east, north)

    # Build tags dictionary
    tags = {k: True for k, _ in features} if features else {}

    # Master feature list for consistent schema
    if all_features is None:
        all_features = [f"{k}:{v}" if v else k for k, v in features]

    try:
        pois = ox.features_from_bbox(bbox, tags)

        if pois is None or pois.empty:
            print("[Info] No features found, returning zero vector.")
            return {feat: 0 for feat in all_features}

        pois_df = pois.reset_index()
        print(f"[Info] Retrieved {len(pois_df)} features from OSM.")

    except Exception as e:
        print(f"[Warning] OSM query failed: {e}")
        return {feat: 0 for feat in all_features}

    feature_vec = {feat: 0 for feat in all_features}
    for key, value in features:
        col_name = f"{key}:{value}" if value else key
        if key in pois_df.columns:
            if value:
                feature_vec[col_name] = (
                    pois_df[key].astype(str).str.lower().eq(str(value).lower()).sum()
                )
            else:
                feature_vec[col_name] = pois_df[key].notna().sum()

    return feature_vec


def build_feature_dataframe(facility_dicts, features, box_size_km=1):
    results = {}

    for facility, coords in facility_dicts.items():
        vec = get_feature_vector(
            coords["latitude"],
            coords["longitude"],
            box_size_km=box_size_km,
            features=features,
        )

        # Extract County from facility name (assumes "Facility, County")
        if "," in facility:
            county = facility.split(",")[-1].strip()
        else:
            county = "Unknown"

        vec["County"] = county

        # Store results indexed by County
        results[facility] = vec

    df = pd.DataFrame(results).T
    return df
