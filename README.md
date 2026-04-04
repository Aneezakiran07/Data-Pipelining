# Advanced Data Cleaning Pipeline

A Streamlit app that takes messy data files and cleans them automatically or manually, column by column.

---

## Demo

https://data-pipelining-qnddvbut7ayklccgo5cp8v.streamlit.app/

---

## What It Does

Upload a CSV or Excel file and the app scans it, tells you what is wrong, and lets you fix it. You can let the pipeline handle everything automatically or go through each issue yourself and decide exactly which columns to touch. Every action is tracked, undoable, exportable, and reusable across datasets.

---

## Features

### Navbar Tab Layout

The app is organized into tabs at the top of the page like a website navbar. Upload, Overview, Recommendations, Clean, Validate, AI Assistant, Profile, and History and Export. Each tab focuses on one task so the workflow stays clear.

### Smart Recommendations

When you upload a file the app analyzes every column and surfaces detected issues. Each issue appears as a row with a description and a column selector dropdown. You choose which columns to fix or run Auto Fix All to execute the full pipeline in one go.

### Manual Cleaning

All operations are available in the Clean tab. This includes basic cleaning, advanced cleaning, find and replace, and column type override. Every operation follows the same column selection pattern so nothing feels inconsistent.

### AI Assistant Tab

All AI features live in one place.

- **AI Cleaner** sits at the top. Describe what you want in plain English and the app generates and shows you the code before running anything.
- **AI Data Analysis** sits below it. One click scans your dataset and returns prioritised fix cards.
- **General Guide** sits at the bottom as a reference checklist.

Before this update these features were scattered across tabs and easy to miss.

### Data Quality Score

A score from 0 to 100 is shown at the top of the Overview tab. It is computed across five dimensions:

- Completeness
- Uniqueness
- Type consistency
- Outlier cleanliness
- Validity

Each dimension contributes equally. The score is color coded with green above 80, amber between 55 and 80, and red below 55. It updates automatically as the dataset changes so improvements are visible in real time.

### Data Type Guesser

A section in the Clean tab scans all columns and suggests correct data types based on actual values. It detects patterns like email, boolean, currency, percentage, units, durations, datetime, numeric strings, and low cardinality categories. Each suggestion includes a confidence percentage, reason, and sample values. You can select multiple suggestions and apply them in one action.

### Validation and Quality Checks

Email, phone, date, outlier, and range validation all live in the Validate tab. Each follows the same select columns then run pattern.

### Column Profiler

The Profile tab shows per column statistics including:

- Min, max, mean, median
- Standard deviation and skewness
- Null percentage and sample values

It also includes a before and after comparison view to track exact changes made during cleaning.

### Correlation Heatmap

The Profile tab includes a correlation matrix for numeric columns. It supports Pearson, Spearman, and Kendall methods. Strong relationships are highlighted so redundant columns can be identified quickly. A short summary surfaces the strongest pairs.

### Filter and Inspect

The Filter and Inspect tab lets you stack multiple filters and switch between two modes:

- **ALL mode** chains filters on the result of the previous filter (AND logic)
- **ANY mode** builds a union across all filters applied independently to the original data (OR logic)

Filtered results can be exported as CSV or Excel directly from the tab.

### Cleaning History and Undo

- Every operation is recorded with the dataset shape at that step
- Undo the last action or redo it if you change your mind
- Undo and redo are also accessible from the sidebar so you never have to leave your current tab
- Up to 20 steps are stored
- The session persists across page reloads

### Pipeline Export and Reload

Cleaning steps can be exported as a `pipeline.py` script with runnable Python code. The workflow can also be saved as a `pipeline.json` file. Uploading this file on a new dataset replays all automatable steps in order and skips steps that require manual input with a clear note about what was skipped.

### Cleaning Report PDF

A full PDF report can be generated after cleaning. It includes:

- Before and after dataset summary
- Column profiles and missing value breakdown
- Applied steps with dataset shape at each step
- AI summary of the dataset if you generated one
- A sample of the cleaned data

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
- Split one column into multiple columns by a delimiter
- Merge multiple columns into one

---

## Validation Operations

- Email validation to flag or remove invalid addresses
- Phone standardization with support for custom country codes
- Date standardization with support for custom input formats
- Outlier detection using IQR or Z score with cap or remove options
- Range validation with user defined limits

---

## Multi Sheet Excel Support

When uploading an Excel file with multiple sheets the app shows a selector to choose the sheet. Switching sheets resets the state and loads the selected sheet fresh.

---

## Getting Started

### Requirements
streamlit
pandas
numpy
scikit-learn
openpyxl
reportlab

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

## A Note on Performance

Streamlit reruns the entire script on every user interaction which makes performance optimization genuinely difficult. Here is what has been done to keep things fast:

- All expensive operations are button gated so nothing runs until you ask for it
- Analysis functions run on sampled data for large files
- Caching is keyed to the actual dataframe state so results persist across reruns without stale data
- Large CSV files are read in chunks and numeric columns are downcast to reduce memory usage

A framework migration is planned for the next version which should resolve the remaining performance ceiling entirely.

---

## How Column Selection Works

Each operation follows the same pattern. A description is shown with a dropdown selector. Inside the selector there is an apply to all option and individual column checkboxes. The action button activates only after selecting at least one column. The label updates to reflect the number of selected columns.

---

## Data Privacy

Everything runs locally. No data is sent externally. The AI features use the Gemini API which receives only column names, data types, and a small sample of row values.

---

## File Support

- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`) with single sheet or multi sheet support

---

## Test Data

Two test files are included in the repo.

**test_data.csv** covers every error type the app handles including mixed date formats, invalid emails, messy phone numbers, currency strings, percentages, unit values, duration strings, missing values, whitespace, and duplicate rows.

**test_multisheet.xlsx** is an Excel file with three sheets: Employees, Sales, and Products. Each sheet has a different set of messy columns. Use it to test the sheet selector and verify that switching sheets loads the correct data.


