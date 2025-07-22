# PredictionAID App

A Streamlit-based application for predictive analysis with multiple tools including OLS Regression, Random Forest AI, and Variable Selection.

## Quick Start

### Option 1: Using the startup script (Recommended)
Simply run from the repository root:
```bash
python run_app.py
```

### Option 2: Using the batch file (Windows)
Double-click `run_app.bat` or run from command prompt:
```cmd
run_app.bat
```

### Option 3: Manual startup (Guaranteed to work)
Navigate to the Tool_App folder and run with the specific conda environment:
```bash
cd Tool_App
C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe -m streamlit run Main.py
```

### Option 4: For other users without conda environment
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
├── run_app.py          # Startup script
├── run_app.bat         # Windows batch file
├── requirements.txt    # Python dependencies
└── Tool_App/
    ├── Main.py         # Main Streamlit application
    ├── pages/          # Streamlit pages
    ├── utils/          # Utility functions
    └── *.jpg, *.png    # Image assets
```

## Features

- **OLS Regression (Linear)**: Perform linear regression analysis
- **Random Forest AI (NonLinear)**: Advanced non-linear modeling
- **Variable Selection Tool**: Analyze variable importance
- **Correlation Matrix**: Visualize variable relationships

## Development

The application uses relative paths so it can be run from any location where the repository is cloned. All image and file paths are resolved relative to the script locations.
