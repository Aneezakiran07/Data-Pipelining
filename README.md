# 🧹 Clean Data Fast

Your messy spreadsheet is stressing you out. We get it.

This app fixes common data problems automatically. No coding needed. Just upload and click.

## What It Does

Your data probably has issues. Everyone's does.

- Spaces where they shouldn't be
- Duplicate rows eating up memory  
- Currency symbols messing with calculations
- Missing values breaking your analysis
- Percentages stored as text
- Units mixed into numbers

This app finds and fixes all of that. Automatically.

## How It Works

**1. Upload your file**  
Drop in a CSV or Excel file. Done.

**2. See what's wrong**  
Check the stats. Red numbers show problems.

**3. Click to fix**  
Each button fixes one type of issue. Or hit "Full Pipeline" to fix everything at once.

**4. Download cleaned data**  
Get your clean file in CSV or Excel format.

That's it. Five minutes max.

## The Smart Stuff

**Currency Converter**  
Turns "$1,200" and "€950" into actual numbers. Works with 100+ currencies.

**Percentage Handler**  
Changes "75%" into 0.75 so you can do math with it.

**Unit Parser**  
Extracts numbers from "70kg" and "154 lbs". Converts durations like "1h30m" to seconds.

**Missing Value Fixer**  
Uses machine learning to fill in blanks intelligently. Not just averages - actual smart predictions based on your data patterns.

**String Cleaner**  
Removes junk characters but keeps the good stuff. Knows the difference.

## Who Uses This

- Analysts tired of Excel preprocessing
- Students with messy survey data
- Researchers cleaning experiment results
- Anyone who's ever thought "I just need clean data"

## Installation

```bash
pip install -r requirements.txt
streamlit run phase4_app_fixed.py
```

Opens in your browser. Works offline. Your data never leaves your computer.

## Requirements

- Python 3.8 or higher
- 5 minutes of your time
- Messy data (we know you have some)

## Examples

**Before:**
```
Name: "  Alice  "
Price: "$1,200"
Discount: "10%"
Weight: "70kg"
```

**After:**
```
Name: Alice
Price: 1200
Discount: 0.10
Weight: 70
```

Clean. Numeric. Ready for analysis.

## What Makes This Different

Most data cleaning tools are either:
- Too simple (just removes duplicates)
- Too complex (requires coding)

This one is smart but simple. It detects patterns in your data and fixes them correctly. Not just find-and-replace.

## Built For Real Data

We tested this on:
- E-commerce spreadsheets with mixed currencies
- Survey results with weird formatting
- Financial data with parentheses for negatives
- Measurement data with inconsistent units
- Everything with missing values

It handles all of it.

## Free and Open

Use it however you want. Personal projects. Work stuff. Whatever.

No signup. No tracking. No cloud upload required.

## Problems?

Something broken? Data not cleaning right?

Open an issue. We actually read them.

## One More Thing

Your data is your data. This app processes everything locally on your computer. Nothing gets uploaded anywhere. We don't see it, store it, or send it to any server.

Privacy matters.

---

Made with ☕ and frustration from cleaning too many datasets manually.