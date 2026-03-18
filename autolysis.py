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
import json
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

def log(message: str):
    print(f"[INFO] {message}")

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

#Skewed Detection
def detect_skewness(df:pd.DataFrame) -> dict:
    skew_details={}

    numeric_df=df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        skew_value=numeric_df[col].skew()

        if skew_value>1:
            skew_type="Highly positively skewed"
        elif skew_value>0.5:
            skew_type="Moderately positively skewed"
        elif skew_value<-1:
            skew_type="Highly negatively skewed"
        elif skew_value<-0.5:
            skew_type="Moderately negatively skewed"
        else:
            skew_type="Approximately symmetric"
        
        skew_details[col]={
            "skew_value":round(skew_value,1),
            "skew_type":skew_type
        }
    return skew_details

#Feature scale check
def detect_scale_variation(df:pd.DataFrame) -> dict:
    scale_info={}

    numeric_df=df.select_dtypes(include=[np.number])

    if numeric_df.shape[1]<2:
        return scale_info
    
    ranges={}
    for col in numeric_df.columns:
        col_range=numeric_df[col].max()-numeric_df[col].min()
        ranges[col]=col_range
    
    max_range=max(ranges.values())
    min_range=min(ranges.values())

    if min_range==0:
        scale_issue=True
    else:
        scale_issue=(max_range/min_range)>10
    
    scale_info["ranges"]=ranges
    scale_info["scale_difference_detected"]=scale_issue

    return scale_info

#Detecting Duplicates
def detect_duplicates(df:pd.DataFrame) -> int:
    return df.duplicated().sum()

#Deciding visualization
def decide_visualization(profile:dict, analysis:dict) -> list:
    charts=[]
    if not analysis.get("has_numeric"):
        return charts
    num_numeric=profile.get("num_numeric_columns",0)
    num_rows=profile.get("num_rows",0)

    if num_numeric >= 2:
        charts.append("correlation")
        charts.append("distribution")
    elif num_numeric == 1:
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

    if numeric_df.empty:
        return

    col=numeric_df.columns[0] #First numeric column [0]
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
def prep_summary(profile: dict, analysis: dict, charts: list, outliers: dict, duplicate_rows: int, skewness: dict, scale_info: dict) ->dict:
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
    summary["outliers"]=outliers
    summary["duplicate_rows"]=duplicate_rows
    summary["skewness"]=skewness
    summary["scale_analysis"]=scale_info

    return summary

#Creation of ReadME File
def writing_readme(summary:dict,charts:list,output_dir:Path):
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

    sentences.append(f"- Duplicate Rows: {summary['duplicate_rows']}")
    sentences.append("")

    sentences.append("## Analysis Summary")
    sentences.append(f"- Numeric Columns: {summary['numeric_columns']}")
    sentences.append(f"- Categorical Columns: {summary['categorical_columns']}\n")

    sentences.append("## Outlier Analysis")
    for col,info in summary["outliers"].items():
        sentences.append(f"- {col}: {info['count']} outlier(s)")
    sentences.append("")

    sentences.append("## Skewness Analysis")
    for col,info in summary["skewness"].items():
        sentences.append(
            f"- {col}: {info['skew_type']} (Skewness = {info['skew_value']})"
        )
    sentences.append("")

    if summary["scale_analysis"]:
        sentences.append("## Feature Scale Analysis")
        if summary["scale_analysis"]["scale_difference_detected"]:
            sentences.append(
                "- Significant scale differences detected across numeric features. "
                "Feature scaling may be beneficial for modeling tasks."
            )
        else:
            sentences.append(
                "- Numeric features appear to be on comparable scales."
            )
        sentences.append("")

    if "correlation" in charts:
        sentences.append("## Correlation Analysis")
        sentences.append("A correlation heatmap was generated to understand the relation between numeric columns\n")
        sentences.append("![Correlation Heatmap](correlation.png)\n")

    if "distribution" in charts:
        sentences.append("## Distribution Analysis")
        sentences.append("A distribution plot was generated to understand the spread of numeric columns\n")
        sentences.append("![Distribution Plot](distribution.png)\n")
    
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as writer:
        writer.write("\n".join(sentences))

#Reading Readme
def reading_readme(output_dir: Path) -> str:
    with open(output_dir / "README.md", "r", encoding="utf-8") as reader:
        return reader.read()
    
#LLM Polishing
@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def polish_with_llm(text: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        log("No Gemini API key found. Skipping LLM polishing.")
        return text

    print("Calling Gemini LLM...")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    prompt = f"""
You are a professional data analyst.

Convert the following dataset analysis into a structured, engaging narrative report.

Include:
1. Dataset Overview
2. Key Observations
3. Insights
4. Implications

Make it clean, readable, and in markdown.

Analysis:
{text}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        log(f"Gemini failed ({response.status_code}), skipping LLM.")
        print(response.text)
        return text

    data = response.json()
    

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        log("Unexpected Gemini response format, skipping...")
        return text



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
    output_dir = Path(csv_path.stem)
    output_dir.mkdir(exist_ok=True)

    print(df.head())

    profile=profile_data(df)
    analysis=analyze_numeric_data(df)
    outliers=outliers_detect(df)
    duplicate_rows=detect_duplicates(df)
    skewness=detect_skewness(df)
    scale_info=detect_scale_variation(df)

    charts_decided=decide_visualization(profile,analysis)
    summary=prep_summary(profile,analysis,charts_decided,outliers,duplicate_rows, skewness, scale_info)

    log(f"Processing dataset: {csv_path.name}")
    log("Profiling completed")
    log("Numeric analysis completed")
    log(f"Charts selected: {charts_decided}")
    if "correlation" in charts_decided:
        plotting_correlation(df,output_dir/"correlation.png")
        log("Correlation chart generated")
    if "distribution" in charts_decided:
        plotting_distribution(df,output_dir/"distribution.png")
        log("Distribution chart generated")
    log("Summary Prepared for narration")
    writing_readme(summary,charts_decided,output_dir)
    log("README.md generated")
    original_file=json.dumps(summary,indent=2,default=str)
    polished_file = polish_with_llm(original_file)

    with open(output_dir / "README.md", "w", encoding="utf-8") as writer:
        writer.write(polished_file)
    
    log("LLM narration completed")

if __name__=="__main__":
    main()