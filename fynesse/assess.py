from typing import Any, Union, Tuple, List, Optional
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import osmnx as ox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def standardize_county_column(
    df: pd.DataFrame, possible_names: List[str]
) -> pd.DataFrame:
    """
    Renames the first matching county column to 'County' with cleaned formatting
    """
    for col in df.columns:
        if col.strip().lower() in [name.lower() for name in possible_names]:
            df.rename(columns={col: "County"}, inplace=True)
            df["County"] = df["County"].astype(str).str.strip().str.title()
            return df
    print("No county column found in:", df.columns.tolist())
    return df


def harmonize_county_names(
    df_population: pd.DataFrame,
    df_health_facilities: pd.DataFrame,
    df_osm_geo_features: pd.DataFrame,
    gdf_counties: pd.DataFrame,
    mapping: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Harmonize county names across datasets to match GeoJSON counties.

    This falls under: **assess** (data cleaning / standardization)
    """
    print("\nHarmonizing county names...")

    # Get unique counties from each dataset
    population_counties = set(df_population["County"].unique())
    health_facilities_counties = set(df_health_facilities["County"].unique())
    osm_geo_features_counties = set(df_osm_geo_features["County"].unique())
    geo_counties = set(gdf_counties["County"].unique())

    # Find mismatches
    mismatches_population = population_counties - geo_counties
    mismatches_health = health_facilities_counties - geo_counties
    mimatches_geo_feature = osm_geo_features_counties - geo_counties

    print("Counties in population but not in GeoJSON:", mismatches_population, "\n")
    print("Counties in health_facilities but not in GeoJSON:", mismatches_health, "\n")
    print(
        "Counties in osm geo features but not in GeoJSON:", mimatches_geo_feature, "\n"
    )

    # Apply harmonization
    df_health_facilities["County"] = df_health_facilities["County"].replace(mapping)
    df_population["County"] = df_population["County"].replace(mapping)
    df_osm_geo_features["County"] = df_osm_geo_features["County"].replace(mapping)

    print("Harmonization complete.")
    return df_population, df_health_facilities, df_osm_geo_features


def plot_underserved_distribution(
    df: pd.DataFrame, target_col: str = "Underserved"
) -> None:
    """
    Plots class balance in underserved vs adequately served counties
    with percentage labels on the bars.
    """

    # Plot
    ax = sns.countplot(
        data=df, x=target_col, hue=target_col, palette="Set2", legend=False
    )

    # Add percentage labels on bars
    for p in ax.patches:
        height = p.get_height()
        percentage = 100 * height / len(df)
        ax.annotate(
            f"{percentage:.1f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    # Custom x-axis labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Underserved", "Adequately Served"])

    plt.title("Underserved vs Adequately Served Counties")
    plt.ylabel("Count of Counties")
    plt.xlabel("")
    plt.show()


def plot_health_facilities(
    df_facilities: pd.DataFrame,
    gdf_counties: gpd.GeoDataFrame,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    crs_epsg: int = 3857,
    facility_color: str = "blue",
    facility_size: int = 10,
    alpha: float = 0.6,
    title: str = "Health Facility Locations",
) -> None:
    gdf_facilities = gpd.GeoDataFrame(
        df_facilities,
        geometry=gpd.points_from_xy(df_facilities[lon_col], df_facilities[lat_col]),
        crs="EPSG:4326",
    )

    # Project to target CRS
    gdf_counties = gdf_counties.to_crs(epsg=crs_epsg)
    gdf_facilities = gdf_facilities.to_crs(epsg=crs_epsg)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_counties.boundary.plot(ax=ax, color="black", linewidth=0.8)

    gdf_facilities.plot(
        ax=ax,
        color=facility_color,
        markersize=facility_size,
        alpha=alpha,
        label="Health Facilities",
    )

    # Add basemap
    ctx.add_basemap(ax, crs=gdf_counties.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_title(title, fontsize=12)
    ax.axis("off")
    ax.legend()
    plt.show()


def plot_health_facilities_byType(
    df_facilities: pd.DataFrame,
    gdf_counties: gpd.GeoDataFrame,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    crs_epsg: int = 3857,
    facility_size: int = 10,
    alpha: float = 0.6,
    title: str = "Health Facility Locations",
) -> None:
    gdf_facilities = gpd.GeoDataFrame(
        df_facilities,
        geometry=gpd.points_from_xy(df_facilities[lon_col], df_facilities[lat_col]),
        crs="EPSG:4326",
    )

    # Project to target CRS
    gdf_counties = gdf_counties.to_crs(epsg=crs_epsg)
    gdf_facilities = gdf_facilities.to_crs(epsg=crs_epsg)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_counties.boundary.plot(ax=ax, color="black", linewidth=0.8)

    # Plot facilities colored by Type
    gdf_facilities.plot(
        ax=ax,
        column="Type",
        categorical=True,
        legend=False,  # We will add a custom legend
        markersize=facility_size,
        alpha=alpha,
        cmap="tab10",
    )

    # Add basemap
    ctx.add_basemap(ax, crs=gdf_counties.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Create legend outside the map
    types = gdf_facilities["Type"].unique()
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=typ,
            markerfacecolor=plt.get_cmap("tab10")(i / len(types)),
            markersize=10,
        )
        for i, typ in enumerate(types)
    ]
    ax.legend(
        handles=handles,
        title="Facility Type",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


def plot_county_facilities(
    county_name: str,
    df_facilities: pd.DataFrame,
    lon_col: str = "Longitude",
    lat_col: str = "Latitude",
    type_col: str = "Type",
    state_name: Optional[str] = None,
    crs_epsg: int = 3857,
    facility_size: int = 10,
    alpha: float = 0.6,
    figsize: tuple[int, int] = (10, 10),
    title: Optional[str] = None,
) -> None:
    """
    Plots a single county map from OSM and overlays health facilities colored by type,
    with the legend placed outside the map.
    """
    query = (
        f"{county_name}, Kenya"
        if not state_name
        else f"{county_name}, {state_name}, Kenya"
    )
    gdf_county = ox.geocode_to_gdf(query).to_crs(epsg=crs_epsg)

    gdf_facilities = gpd.GeoDataFrame(
        df_facilities,
        geometry=gpd.points_from_xy(df_facilities[lon_col], df_facilities[lat_col]),
        crs="EPSG:4326",
    ).to_crs(epsg=crs_epsg)

    gdf_facilities = gpd.sjoin(
        gdf_facilities, gdf_county, predicate="within", how="inner"
    )

    fig, ax = plt.subplots(figsize=figsize)
    gdf_county.boundary.plot(ax=ax, color="black", linewidth=1)

    # Plot facilities by type
    types = gdf_facilities[type_col].unique()
    cmap = plt.get_cmap("tab10")
    for i, typ in enumerate(types):
        subset = gdf_facilities[gdf_facilities[type_col] == typ]
        subset.plot(
            ax=ax,
            markersize=facility_size,
            alpha=alpha,
            color=cmap(i / len(types)),
            label=typ,
        )

    ctx.add_basemap(ax, crs=gdf_county.crs, source=ctx.providers.OpenStreetMap.Mapnik)

    plot_title = title or f"{county_name} County Health Facilities by Type"
    ax.set_title(plot_title, fontsize=12)
    ax.axis("off")

    ax.legend(title="Facility Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_feature_importance_rf(
    df: pd.DataFrame,
    target_col: str = "Underserved",
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    title: str = "Feature Importance Heatmap",
):
    X = df.drop(
        columns=[target_col] + [col for col in df.columns if col.lower() == "county"]
    )
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    rf_clf.fit(X_train, y_train)

    # Compute feature importances
    feature_importances = pd.DataFrame(
        {"Feature": X.columns, "Importance": rf_clf.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Convert to percentage
    feature_importances["Importance_pct"] = (
        100
        * feature_importances["Importance"]
        / feature_importances["Importance"].sum()
    )
    feature_importances = feature_importances.sort_values(
        by="Importance_pct", ascending=True
    )

    plt.figure(figsize=(8, max(4, 0.5 * len(feature_importances))))
    bars = plt.barh(
        y=feature_importances["Feature"],
        width=feature_importances["Importance_pct"],
        color=sns.color_palette("YlOrRd", n_colors=len(feature_importances)),
        edgecolor="black",
    )

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=8,
        )

    plt.xlabel("Importance (%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return rf_clf, feature_importances


def analyse_correlation(
    df: pd.DataFrame,
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
    annot: bool = True,
    title: str = "Correlation Matrix",
):
    numeric_df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return corr_matrix
