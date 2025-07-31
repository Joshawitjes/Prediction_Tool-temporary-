# PredictionAID App

A Streamlit-based application for predictive analysis with multiple tools including OLS Regression, Random Forest AI, and Variable Selection.

## Quick Start

### Option 1: Manual startup (Guaranteed to work)
Navigate to the Tool_App folder and run with the specific conda environment:
```bash
cd Tool_App
C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe -m streamlit run Main.py
```

### Option 2: For other users without conda environment
If the conda environment doesn't exist, install dependencies first:
```bash
pip install -r requirements.txt
cd Tool_App
python -m streamlit run Main.py
```

**Note**: The startup scripts prioritize the working conda environment (`tool_app`) but will fall back to system Python if needed.

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