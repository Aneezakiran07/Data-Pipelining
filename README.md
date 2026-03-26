# Advanced Data Cleaning Pipeline

A Streamlit app that takes messy data files and cleans them automatically or manually, column by column.

---

## Demo

https://data-pipelining-qnddvbut7ayklccgo5cp8v.streamlit.app/

---

## What It Does

Upload a CSV or Excel file and the app scans it, tells you what is wrong, and lets you fix it. You can let the pipeline handle everything automatically or go through each issue yourself and decide exactly which columns to touch. Every action is tracked, undoable, exportable, and now also reusable across datasets.

---

## Features

### Navbar Tab Layout

The app is organized into tabs at the top of the page like a website navbar Upload Overview Recommendations Clean Validate Profile and History and Export. Each tab focuses on one task so the workflow stays clear.

### Smart Recommendations

When you upload a file the app analyzes every column and surfaces detected issues. Each issue appears as a row with a description and a column selector dropdown. You choose columns and apply fixes or run Auto Fix All to execute the full pipeline.

### Manual Cleaning

All operations are available in the Clean tab. This includes basic cleaning advanced cleaning find and replace and column type override using the same column selection pattern.

### Data Quality Score

A score from 0 to 100 is shown at the top of the Overview tab. It is computed across five dimensions completeness uniqueness type consistency outlier cleanliness and validity. Each dimension contributes equally. The score is color coded with green above 80 amber between 55 and 80 and red below 55. It updates automatically as the dataset changes so improvements are visible in real time. A breakdown section shows each dimension score out of 20 with a short explanation.

### Data Type Guesser

A section in the Clean tab scans all columns and suggests correct data types based on actual values. It detects patterns like email boolean currency percentage units durations datetime numeric strings and low cardinality categories. Each suggestion includes confidence percentage reason and sample values. Users can select multiple suggestions and apply them in one action.

### Validation and Quality Checks

Email phone date outlier and range validation live in the Validate tab. Each follows the same select columns then run pattern.

### Column Profiler

The Profile tab shows per column statistics including min max mean median standard deviation skewness null percentage and sample values. It also includes a before and after comparison view to track exact changes.

### Correlation Heatmap

The Profile tab includes a correlation matrix for numeric columns. It supports Pearson Spearman and Kendall methods. Strong relationships are highlighted so redundant columns can be identified quickly. A short summary surfaces the strongest pairs.

### Cleaning History and Undo

Every operation is recorded in the History and Export tab with dataset shape at that step. You can undo the last action or clear the full history. Up to 20 steps are stored.

### Pipeline Export and Reload

Cleaning steps can be exported as a pipeline.py script with runnable Python code. The workflow can also be saved as a pipeline.json file containing step labels. Uploading this file on a new dataset replays all automatable steps in order while skipping steps that require manual input.

### Cleaning Report PDF

A full PDF report can be generated after cleaning. It includes before and after dataset summary column profiles missing value breakdown applied steps with dataset shape and a sample of cleaned data. The report updates only when the cleaning history changes.

---

## Cleaning Operations

- Strip leading and trailing whitespace from text columns
- Drop duplicate rows and duplicate columns
- Clean unwanted edge characters from string values
- Convert currency columns to numeric supporting multiple formats
- Convert percentage strings to decimal values
- Extract numeric values from unit strings
- Convert duration strings to seconds
- Handle missing values using KNN or MICE
- Find and replace with optional regex
- Force a column to a specific type

---

## Validation Operations

- Email validation flag or remove invalid emails
- Phone standardization to a consistent format
- Date standardization across multiple formats
- Outlier detection using IQR or Z score with cap or remove
- Range validation with user defined limits

---

## Multi Sheet Excel Support

When uploading an Excel file with multiple sheets the app shows a selector to choose the sheet. Switching sheets resets the state and loads the selected sheet.

---

## Getting Started

### Requirements

```
streamlit
pandas
numpy
scikit-learn
openpyxl
reportlab
```

Install with

```bash
pip install streamlit pandas numpy scikit-learn openpyxl reportlab
```

### Run

```bash
streamlit run app.py
```

Upload a CSV or Excel file and start cleaning.

---

## How Column Selection Works

Each operation follows the same pattern. A description is shown with a dropdown selector. Inside the selector there is an apply to all option and individual column checkboxes. The action button activates only after selecting at least one column. The label updates to reflect the number of selected columns.

---

## Data Privacy

Everything runs locally. No data is sent externally.

---

## File Support

- CSV (.csv)
- Excel (.xlsx, .xls) — single sheet or multi-sheet

---

## Test Data

Two test files are included in the repo.

test_data.csv: a single CSV covering every error type the app handles. Mixed date formats, invalid emails, messy phone numbers, currency strings, percentages, unit values, duration strings, missing values, whitespace, and duplicate rows.

test_multisheet.xlsx: an Excel file with three sheets (Employees, Sales, Products), each with a different set of messy columns. Use this to test the sheet selector and verify that switching sheets loads the correct data.