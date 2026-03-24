# Advanced Data Cleaning Pipeline

A Streamlit app that takes messy data files and cleans them — automatically or manually, column by column.

---

## Demo

https://data-pipelining-qnddvbut7ayklccgo5cp8v.streamlit.app/

---

## What It Does

Upload a CSV or Excel file and the app scans it, tells you what is wrong, and lets you fix it. You can let the pipeline handle everything automatically or go through each issue yourself and decide exactly which columns to touch. Every action is tracked, undoable, and exportable as a reusable Python script.

---

## Features

### Navbar Tab Layout

The app is organized into six tabs that sit at the top of the page like a website navbar — Upload, Overview, Recommendations, Clean, Validate, Profile, and History & Export. No scrolling through a long page. Each tab has one job.

### Smart Recommendations

When you upload a file the app analyzes every column and surfaces a list of detected issues. Each issue shows up as a plain row with a description. Next to it is a column selector dropdown — click it, pick which columns you want to apply the fix to, and hit Fix. The button stays disabled until at least one column is selected. You can also hit Auto-Fix All to run the full pipeline in one shot.

### Manual Cleaning

Every operation is also available as a manual button in the Clean tab. Basic cleaning, advanced cleaning, find and replace, and column type override all live here with the same column-selection pattern.

### Validation and Quality Checks

Email, phone, date, outlier, and range validation all live in the Validate tab. Each one uses the same select-columns-then-run pattern.

### Column Profiler

The Profile tab shows per-column stats — min, max, mean, median, std, skewness, null percentage, and sample values — in a single table. It also has a before/after comparison view where you pick a column and see original values side by side with current cleaned values, with changed rows marked.

### Cleaning History and Undo

Every operation is recorded in the History & Export tab with the row and column count at the time it ran. You can undo the last step or clear the whole history. Max 20 steps are kept.

### Pipeline Export

Once you have cleaned a file, the History & Export tab lets you download your cleaning steps as a `pipeline.py` script. The script contains real runnable Python code — not just comments — for every operation you ran. You can reuse it on new files of the same format without opening the app again.

---

## Cleaning Operations

- Strip leading and trailing whitespace from text columns
- Drop duplicate rows and duplicate columns
- Clean unwanted edge characters from string values
- Convert currency columns to numeric — supports USD, EUR, GBP, INR, PKR, and other formats
- Convert percentage strings like "15%" to decimal values like 0.15
- Extract numeric values from unit strings like "70kg" or "154 lbs"
- Convert duration strings like "1h30m" or "90min" to seconds
- Handle missing values using KNN imputation for small datasets and MICE for large ones
- Find and replace across any column with optional regex support
- Force a column to a specific type — string, integer, float, datetime, boolean, or category

---

## Validation Operations

- Email validation — flag or remove rows with invalid email format
- Phone standardization — strips all formatting and outputs +[country code][number]
- Date standardization — parses 17 different date formats and outputs to one consistent format you choose
- Outlier detection — IQR or Z-score, with cap or remove as the action
- Range validation — flags or removes values outside a min/max you define

---

## Multi-Sheet Excel Support

When you upload an Excel file with multiple sheets, the app shows a dropdown to pick which sheet to load. Switching sheets resets the cleaning state and loads the new sheet fresh.

---

## Getting Started

### Requirements

```
streamlit
pandas
numpy
scikit-learn
openpyxl
```

Install with:

```bash
pip install streamlit pandas numpy scikit-learn openpyxl
```

### Run

```bash
streamlit run app.py
```

Then upload any CSV or Excel file and start cleaning.

---

## How Column Selection Works

Every cleaning operation uses the same interaction pattern. You see the operation described in plain text. Next to it is a dropdown that opens a popup. Inside is an "Apply to all" checkbox at the top, followed by individual checkboxes for each affected column. The action button is disabled until at least one column is checked. The label updates to show how many columns are selected.

This means you never accidentally apply a fix to columns you did not intend to touch.

---

## Data Privacy

Everything runs locally. No data is sent anywhere. Your files stay on your machine.

---

## File Support

- CSV (.csv)
- Excel (.xlsx, .xls) — single sheet or multi-sheet

---

## Test Data

Two test files are included in the repo.

`test_data.csv` — a single CSV covering every error type the app handles. Mixed date formats, invalid emails, messy phone numbers, currency strings, percentages, unit values, duration strings, missing values, whitespace, and duplicate rows.

`test_multisheet.xlsx` — an Excel file with three sheets (Employees, Sales, Products), each with a different set of messy columns. Use this to test the sheet selector and verify that switching sheets loads the correct data.


