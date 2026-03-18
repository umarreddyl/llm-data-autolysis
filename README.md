# Automated Data Analysis & Reporting - Autolysis

## Overview

This project is an automated data analysis pipeline developed as part of the KaroStartup Internship.
It takes a CSV dataset as input, performs exploratory data analysis, generates visualizations, and produces a structured analytical report.

The objective is to reduce manual effort in understanding datasets by combining statistical analysis with automated report generation, with optional LLM-based refinement.

---

## Features

* Dataset profiling (rows, columns, data types, missing values)
* Detection of:

  * Outliers (using IQR method)
  * Skewness (distribution analysis)
  * Duplicate records
  * Feature scale differences
* Automatic chart generation:

  * Correlation heatmap
  * Distribution plot
* Structured report generation in Markdown format
* Optional LLM-based enhancement of the report (Google Gemini API)

---

## Project Structure

```
autolysis.py

goodreads/
  ├── README.md
  ├── correlation.png
  ├── distribution.png

happiness/
  ├── README.md
  ├── correlation.png
  ├── distribution.png

media/
  ├── README.md
  ├── correlation.png
  ├── distribution.png
```

Each dataset generates its own folder containing:

* Visualizations
* A detailed analysis report

---

## How It Works

1. The script reads a CSV dataset
2. Performs data profiling and statistical analysis
3. Detects patterns such as outliers, skewness, and scale differences
4. Generates relevant visualizations
5. Creates a structured summary
6. Optionally refines the report using an LLM
7. Saves outputs in a dedicated folder

---

## Installation

Make sure Python 3.11+ is installed.

Install dependencies:

```
pip install pandas numpy matplotlib seaborn httpx tenacity
```

---

## Usage

Run the script with a dataset:

```
uv run autolysis.py <dataset.csv>
```

Example:

```
uv run autolysis.py goodreads.csv
```

---

## Output

For each dataset, a folder is created containing:

* `README.md` → Generated analysis report
* `correlation.png` → Correlation heatmap
* `distribution.png` → Distribution plot

---

## LLM Integration (Gemini)

The project supports optional report refinement using Google Gemini.

Set your API key:

```
setx GEMINI_API_KEY "your_api_key_here"
```

If no API key is provided, the script will skip this step and generate a standard report.

---

## Notes

* Designed to work with any CSV dataset
* Output is automatically structured and saved
* Suitable for quick exploratory analysis and reporting
* Feature scaling is recommended before using data for ML tasks
