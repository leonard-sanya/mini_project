# Fynesse Template

[![Tests](https://github.com/lawrennd/fynesse_template/workflows/Test/badge.svg)](https://github.com/lawrennd/fynesse_template/actions/workflows/test.yml)
[![Code Quality](https://github.com/lawrennd/fynesse_template/workflows/Code%20Quality/badge.svg)](https://github.com/lawrennd/fynesse_template/actions/workflows/code-quality.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-1.0+-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Predicting Underserved Regions in Kenya: A Machine Learning Approach to Healthcare Facilities Distribution.

## Abstract

This project investigates the spatial distribution of healthcare facilities in Kenya and its relationship to population needs by integrating diverse datasets, including the 2019 Census population data, county boundary GeoJSON files, and healthcare facility records. Additional geographic features are extracted from OpenStreetMap to enrich the analysis. The study follows the *Access–Assess–Address* framework: data are accessed and preprocessed, assessed through exploratory data analysis, visualization, and correlation checks, and then addressed with predictive modeling. Logistic Regression and Naïve Bayes classifiers are applied to identify underserved regions, with model performance evaluated using ROC curves and confusion matrices.

---

##  Project Overview
The study follows the **Access – Assess – Address** framework:

1. **Access**  
   - Load and preprocess datasets:  
     - Kenya 2019 Population Census (per county)  
     - Kenya County Boundaries (GeoJSON)  
     - Healthcare Facility Registry  
     - OpenStreetMap (OSM) features such as schools, hospitals, and places of worship  

2. **Assess**  
   - Perform exploratory data analysis (EDA)  
   - Standardize and merge datasets by county  
   - Correlation analysis to detect redundant or highly related variables  
   - Feature importance evaluation for underserved prediction  

3. **Address**  
   - Train predictive models (Logistic Regression, Naïve Bayes)  
   - Evaluate performance with **ROC curves**, **confusion matrices**, and accuracy metrics  

---

##  Key Results
- Successfully merged heterogeneous datasets into a unified framework.  
- Logistic Regression and Naïve Bayes effectively predicted whether a county was underserved.  
- Visualizations revealed counties with limited health facilities relative to population needs.  

---

##  Data Sources
- **Population Data (2019 Census)** – [HDX Kenya Population Dataset](https://data.humdata.org/dataset/kenya-population-per-county-from-census-report-2019)  
- **Healthcare Facilities Data** – [EnergyData.info](https://energydata.info/dataset/kenya-healthcare-facilities)  
- **County Boundaries** – [Kenya County GeoJSON](https://data.humdata.org/dataset/json-repository/resource/51939d78-35aa-4591-9831-11e61e555130)  
- **OpenStreetMap Features** – Extracted using [OSMnx](https://osmnx.readthedocs.io/)

---

## Libraries
- **Python**: Pandas, Geopandas, NumPy, Scikit-learn  
- **Visualization**: Seaborn, Matplotlib  
- **Geospatial Analysis**: OSMnx, Shapely  
- **Workflow Framework**: Access – Assess – Address methodology  
I see the issue 👀 — it’s with your **code block fencing** in step 1.

You opened the block with **\`\`\`bash** but closed it with \*\*\`\`\`\` (four backticks instead of three). That breaks Markdown formatting, so everything after can look off.

---

## How to Run

1. Clone the repo:
  
   ```bash
   git clone https://github.com/leonard-sanya/mlfc_miniproject.git
   cd mlfc_miniproject
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   jupyter notebook Distribution_of_healthcare_facilities_in_Kenya.ipynb
   ```

---

##  Acknowledgements

* Kenya National Bureau of Statistics (KNBS) for census data.
* Humanitarian Data Exchange (HDX) for open datasets.
* OpenStreetMap contributors for geospatial features.

---

## Framework Structure

The template provides a structured approach to implementing the Fynesse framework:

```
fynesse/
├── access.py      # Data access functionality
├── assess.py      # Data assessment and quality checks
├── address.py     # Question addressing and analysis
├── config.py      # Configuration management
├── defaults.yml   # Default configuration values
└── tests/         # Comprehensive test suite
    ├── test_access.py
    ├── test_assess.py
    └── test_address.py
```

## License

MIT License - see LICENSE file for details.
