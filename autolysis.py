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

#Outlier Detection
def outliers_detect(df:pd.DataFrame)->dict:
    outliers={}

    numeric_df=df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        q1=numeric_df[col].quantile(0.25)
        q3=numeric_df[col].quantile(0.75)
        iqr=q3-q1
        
        lower=q1-1.5*iqr
        upper=q3+1.5*iqr
        
        values=numeric_df[(numeric_df[col]<lower) | (numeric_df[col]>upper)][col].tolist()

        outliers[col]={
            "count":len(values),
            "values":values
        }
    return outliers

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

#Summary Preparation
def prep_summary(profile: dict, analysis: dict, charts: list, outliers: dict) ->dict:
    summary={}

    summary["dataset_shape"]={
        "rows":profile.get("num_rows"),
        "columns":profile.get("num_cols")
    }

    summary["columns"]=profile.get("cols")
    summary["missing_values"]=profile.get("missing_vals")
    summary["numeric_columns"]=profile.get("num_numeric_columns")
    summary["categorical_columns"]=profile.get("num_categorical_columns")
    summary["charts_generated"]=charts
    summary["outliers"] = outliers

    return summary

#Creation of ReadME File
def writing_readme(summary:dict,charts:list):
    sentences=[]
    sentences.append("# Automated Data Analysis Report\n")

    sentences.append("## Dataset Overview")
    sentences.append(f"- Rows: {summary['dataset_shape']['rows']}")
    sentences.append(f"- Columns: {summary['dataset_shape']['columns']}")
    sentences.append(f"- Column Names: {', '.join(summary['columns'])}\n")

    
    sentences.append("## Data Quality")
    sentences.append("Missing values per column:")
    for col,count in summary["missing_values"].items():
        sentences.append(f"- {col}: {count}")
    sentences.append("")

    sentences.append("## Analysis Summary")
    sentences.append(f"- Numeric Columns:{summary['numeric_columns']}")
    sentences.append(f"- Categorical Columns: {summary['categorical_columns']}\n")

    sentences.append("## Outlier Analysis")
    for col,info in summary["outliers"].items():
        sentences.append(f"- {col}: {info['count']} outlier(s)")
    sentences.append("")

    if "correlation" in charts:
        sentences.append("## Correlation Analysis")
        sentences.append("A correlation heatmap was generated to understand the relation between numeric columns\n")
        sentences.append("![Correlation Heatmap](correlation.png)\n")

    if "distribution" in charts:
        sentences.append("## Distribution Analysis")
        sentences.append("A distribution plot was generated to understand the spread of numeric columns\n")
        sentences.append("![Distribution Plot](distribution.png)\n")
    
    with open("README.md","w",encoding="utf-8") as writer:
        writer.write("\n".join(sentences))

#Reading Readme
def reading_readme()->str:
    with open("README.md", "r", encoding="utf-8") as reader:
        return reader.read()
    
#LLM Polishing
def polish_with_llm(text: str)->str:
    api_key=os.environ.get("AIPROXY_TOKEN")

    if not api_key:
        print("No API key found. Skipping LLM polishing at this instant.")
        return text
    prompt=f"""
    You are a professional data analyst. 
    Rewrite the following analysis report to be clearer, cleaner, insightful and narrative-driven.
    Do not add new facts. Do not hallucinate. Explain only what is already present.

    Report:
    {text}
    """
    headers={
        "Authorization":f"Bearer {os.environ.get('AIPROXY_TOKEN')}",
        "Content-Type":"application/json"
    }
    payload={
        "model":"gpt-4o-mini",
        "messages":[
            {"role":"user","content":prompt}
        ]
    }

    response=httpx.post(
        "https://api.aiproxy.io/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


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
    outliers = outliers_detect(df)
    charts_decided=decide_visualization(profile,analysis)
    summary=prep_summary(profile,analysis,charts_decided,outliers)


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
    print("Summary Prepared for narration",summary)
    writing_readme(summary,charts_decided)
    print("README.md is generated")
    original_file=reading_readme()
    polished_file=polish_with_llm(original_file)

    with open("README.md","w",encoding="utf-8") as writer:
        writer.write(polished_file)
    
    print("README.md is polished using LLM")

if __name__=="__main__":
    main()