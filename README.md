# Advanced Data Cleaning Pipeline

A Streamlit app that takes messy data files and cleans them — automatically or manually, column by column.

---

## What It Does

Upload a CSV or Excel file and the app will scan it, tell you what is wrong, and let you fix it. You can let the AI handle everything automatically or go through each issue yourself and decide exactly which columns to touch.

---

## Features

### Smart Recommendations

When you upload a file, the app analyzes it and surfaces a list of detected issues. Each issue shows up as a plain row with a description. Next to it is a column selector dropdown — click it, pick which columns you want to apply the fix to, and then hit the Fix button. The Fix button stays disabled until you have selected at least one column. You can also hit Auto-Fix All to run the full pipeline in one shot.

### Manual Cleaning

If you prefer to do things yourself, every operation is also available as a manual button. Basic cleaning, advanced cleaning, and validation all have their own sections with the same column-selection pattern.

### Cleaning Operations

- Strip leading and trailing whitespace from text columns
- Drop duplicate rows and duplicate columns
- Clean unwanted edge characters from string values
- Convert currency columns to numeric — supports USD, EUR, GBP, INR, PKR, and 100+ other formats
- Convert percentage strings like "15%" to decimal values like 0.15
- Extract numeric values from unit strings like "70 kg" or "154 lbs"
- Convert duration strings like "1h30m" or "90 min" to seconds
- Handle missing values using KNN imputation for small datasets and MICE for large ones

### Validation and Quality Checks

- Email validation: flag or remove rows with invalid email format
- Phone standardization: strips all formatting and outputs a clean +[country code][number] format
- Date standardization: parses 17 different date formats including "Jan 5 2021", "28-07-2020", "2022/08/30" and outputs everything to one consistent format you choose
- Outlier detection: uses IQR or Z-score to find outliers, then either caps them at the boundary or removes those rows
- Range validation: flags or removes values outside a min/max you define — useful for columns where you know the valid range

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

## How the Column Selection Works

Every cleaning operation — both in the AI recommendations and in the manual section — uses the same interaction pattern. You see the operation described in plain text. Next to it is a dropdown that opens a popup when clicked. Inside the popup is an "Apply to all" checkbox at the top, followed by individual checkboxes for each affected column. The action button is disabled until at least one column is checked. The dropdown label updates to show how many columns are currently selected.

This means you never have to apply a fix to columns you did not intend to touch.

---

## Data Privacy

Everything runs locally. No data is sent anywhere. Your files stay on your machine.

---

## File Support

- CSV (.csv)
- Excel (.xlsx, .xls)

---

## Test Data

A sample CSV is included that covers every error type the app can handle — mixed date formats, invalid emails, messy phone numbers, currency strings, percentages, unit values, duration strings, outliers, missing values, and duplicate rows.