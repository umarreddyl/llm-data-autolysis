# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "httpx",
#   "tenacity"
# ]
# ///

#File handling
import sys
import os
from pathlib import Path
#Data work
import pandas as pd
import numpy as np
#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
#LLM calls and Retries
from tenacity import retry,stop_after_attempt,wait_exponential
import httpx

#Config
MAX_SAMPLE_ROWS=1000
MAX_CHARTS=3

FIG_WIDTH=6
FIG_HEIGHT=6

RETRY_ATTEMPTS=3
REQUEST_TIMEOUT=30

#Load CSV
def load_csv(csv_path:Path)->pd.DataFrame:
    try:
        df=pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Loading CSV File Error: {e}")
        sys.exit(1)

#Main Function
def main():
    if len(sys.argv)!=2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    csv_path=Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: File Not Found -> {csv_path}")
        sys.exit(1)

    if csv_path.suffix.lower()!=".csv":
        print("Error: Input is not a CSV")
        sys.exit(1)
    df = load_csv(csv_path)
    print(df.head())
    print(f"Processing Dataset: {csv_path.name}")

if __name__=="__main__":
    main()