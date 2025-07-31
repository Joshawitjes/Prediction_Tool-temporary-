# PredictionAID App

A Streamlit-based application for predictive analysis with multiple tools including Variable Selection, OLS Regression and Random Forest AI.

## Quick Start

# Clone the git repository in some new directory
git clone https://github.com/De-Voogt-Naval-Architects/engineering_spec_predictiontool.git

# Navigate to the correct subfolder with the PredictionAid_App
cd engineering_spec_predictiontool/PredictionAid_App


### To actually run the file: 
```bash

# Create a new environment (Python must be installed)
python -m venv venv

# Activate environment
venv\Scripts\activate # should give (venv)

# Install required packages
pip install -r requirements.txt

# Run streamlit app from within the Tool_App folder
cd Tool_App
python -m streamlit run Main.py
```

## Requirements
Make sure you have Python installed and the required packages:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- streamlit
- pandas
- scikit-learn
- PIL (Pillow)
- Other dependencies as listed in requirements.txt

## Project Structure
```
PredictionAid_App/
├── README.md
├── requirements.txt         # Python dependencies
└── Tool_App/
    ├── __init__.py
    ├── Main.py              # Main Streamlit application
    ├── DesignAID_logo.png
    ├── DeVoogt_logo.jpg
    ├── Feadship_logo.jpg
    ├── pages/               # Streamlit pages
    │   ├── 1_Tool_for_Variable_Selection_(Investigative).py
    │   ├── 2_OLS_Regression_(Linear).py
    │   └── 3_Random_Forest_AI_(NonLinear).py
    └── utils/               # Utility functions
        ├── __pycache__/
        └── snowflake_utils.py
```

## Features

- **OLS Regression (Linear)**: Perform linear regression analysis
- **Random Forest AI (NonLinear)**: Advanced non-linear modeling
- **Variable Selection Tool**: Analyze variable importance
- **Correlation Matrix**: Visualize variable relationships

## Development

The application uses relative paths so it can be run from any location where the repository is cloned. All image and file paths are resolved relative to the script