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

def profile_data(df:pd.DataFrame) -> dict:
    profile={}

    profile["num_rows"]=df.shape[0]
    profile["num_cols"]=df.shape[1]

    profile["cols"]=list(df.columns)

    profile["dtypes"]={
        col:str(dtype) for col,dtype in df.dtypes.items()
    } #col name and datatypes are stored in this

    profile["missing_vals"]=df.isna().sum().to_dict()

    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    profile["num_numeric_columns"] = numeric_df.shape[1]
    profile["num_categorical_columns"] = categorical_df.shape[1]

    if not numeric_df.empty:
        profile["numeric_summary"]=numeric_df.describe().to_dict()
    else:
        profile["numeric_summary"]={}
    
    return profile


#Analyzing numeric data

def analyze_numeric_data(df:pd.DataFrame)->dict:
    analysis={}

    numeric_df=df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        analysis["has_numeric"]=False
        return analysis
    
    analysis["has_numeric"]=True
    analysis["summary_stats"]=numeric_df.describe().to_dict()
    analysis["correlation"]=numeric_df.corr().to_dict()

    return analysis

#Deciding visualization

def decide_visualization(profile:dict, analysis:dict) -> list:
    charts=[]
    if not analysis.get("has_numeric"):
        return charts
    num_numeric=profile.get("num_numeric_columns",0)

    if num_numeric>=2:
        charts.append("correlation")
    if num_numeric==1:
        charts.append("distribution")
    
    return charts

#Plotting Correlation

def plotting_correlation(df:pd.DataFrame,output_path:Path):
    numeric_df=df.select_dtypes(include=[np.number])
    if numeric_df.shape[1]<2:
        return
    corr=numeric_df.corr()

    plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
    sns.heatmap(corr,annot=True,cmap="coolwarm",fmt=".2f")
    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#Plotting Distribution

def plotting_distribution(df:pd.DataFrame,output_path:Path):
    numeric_df=df.select_dtypes(include=(np.number))

    if numeric_df.shape[1]!=1:
        return

    col=numeric_df.columns[0]
    values=numeric_df[col]

    plt.figure(figsize=(FIG_WIDTH,FIG_HEIGHT))
    sns.histplot(values,kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

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
    profile=profile_data(df)
    analysis = analyze_numeric_data(df)
    charts_decided=decide_visualization(profile,analysis)
    print(f"Processing Dataset: {csv_path.name}")
    print("Dataset profiling completed")
    print("Basic numeric analysis completed")
    print(f"Charts decided: {charts_decided}")
    if "correlation" in charts_decided:
        plotting_correlation(df,Path("correlation.png"))
        print("Correlation chart generated")
    if "distribution" in charts_decided:
        plotting_distribution(df, Path("distribution.png"))
        print("Distribution chart generated")


if __name__=="__main__":
    main()