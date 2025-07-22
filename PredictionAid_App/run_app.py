#!/usr/bin/env python
"""
Startup script for PredictionAID App
This script can be run from the repository root directory
"""
import os
import sys
import subprocess

def find_python_executable():
    """Find the best Python executable to use"""
    # Primary path - the one that works for this project
    primary_path = r"C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe"
    
    if os.path.exists(primary_path):
        return primary_path
    
    # Try other common conda environment paths
    other_conda_paths = [
        os.path.expanduser(r"~\.conda\envs\tool_app\python.exe"),
        os.path.expanduser(r"~\miniconda3\envs\tool_app\python.exe"),
        os.path.expanduser(r"~\anaconda3\envs\tool_app\python.exe"),
    ]
    
    for path in other_conda_paths:
        if os.path.exists(path):
            return path
    
    # Fall back to system Python (may not work without proper environment)
    print("Warning: Could not find conda environment 'tool_app'")
    print("You may need to install dependencies or activate the environment manually")
    return sys.executable

def main():
    # Get the directory where this script is located (repository root)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the Tool_App directory
    tool_app_dir = os.path.join(repo_root, "Tool_App")
    
    if not os.path.exists(tool_app_dir):
        print(f"Error: Tool_App directory not found at {tool_app_dir}")
        sys.exit(1)
    
    # Find the best Python executable
    python_exe = find_python_executable()
    
    # Change to the Tool_App directory
    os.chdir(tool_app_dir)
    
    print(f"Using Python: {python_exe}")
    print(f"Starting Streamlit from: {tool_app_dir}")
    print("Open your browser and go to the URL shown below...")
    print("-" * 50)
    
    # Run streamlit
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", "Main.py"])
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        print("Try running manually with:")
        print(f"cd Tool_App && {python_exe} -m streamlit run Main.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
