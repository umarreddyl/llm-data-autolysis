# Automated Data Analysis & Reporting (Project 2)

## Overview

This project is an automated data analysis pipeline built as part of the KaroStartup Internship.
It takes a CSV dataset as input, performs exploratory data analysis, generates visualizations, and produces a structured analytical report.

The goal is to reduce manual effort in understanding datasets by combining statistical analysis with automated narrative generation.

---

## Features

* Dataset profiling (rows, columns, data types, missing values)
* Detection of:

  * Outliers (IQR method)
  * Skewness (distribution analysis)
  * Duplicate records
  * Feature scale differences
* Automatic chart generation:

  * Correlation heatmap
  * Distribution plot
* Structured report generation in Markdown format
* Optional LLM-based refinement of the report (using Gemini API)

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

Each dataset gets its own folder containing:

* Visualizations
* A generated analytical report

---

## How It Works

1. The script reads a CSV file
2. It profiles the dataset and performs statistical analysis
3. Relevant visualizations are generated
4. A summary is created based on the analysis
5. The summary is optionally enhanced using a language model
6. Outputs are saved in a dedicated folder

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

* `README.md` → Detailed analysis report
* `correlation.png` → Correlation heatmap
* `distribution.png` → Distribution plot

---

## LLM Integration (Gemini)

The project supports optional report enhancement using Google Gemini.

Set your API key:

```
setx GEMINI_API_KEY "your_api_key_here"
```

If no API key is provided, the script will skip this step and generate a standard report.

---

## Notes

* The datasets used in this project are small and intended for demonstration.
* The pipeline is designed to scale to larger datasets with minimal changes.
* Feature scaling is recommended before using this data for machine learning tasks.
