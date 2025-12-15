# Higgs Discovery Project

This repository contains the complete solution for the Final Project: Discovery of the Higgs boson in the H->ZZ->4l channel.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib scikit-learn
````

## Directory Structure

Ensure your directory looks like this:

```
FinalProject/
├── FinalProject_Complete.py  # The code provided
├── data/
│   ├── clean_data_2011.csv
│   └── clean_data_2012.csv
├── MC/
│   ├── higgs2011.csv
│   ├── zzto4mu2011.csv
│   └── ... (all other MC files)
└── plots/ (Created automatically)
```

## How to Run

Simply execute the script:

```bash
python FinalProject_Complete.py
```

## Output

1. **Terminal**: The script will print the exact numbers needed for your report (Significance, Event Counts, etc.).
2. **Plots**: Check the `plots/` folder for:
      * `step3_mass_no_cuts.png`
      * `step4_Z_masses.png`
      * `step5_mass_final_cuts.png`
      * `step6_mass_ml_cuts.png`

## How to Finish the Report

1. Open `FinalReport.md`.
2. Run the code.
3. Look for lines like `MC Significance: ...` in the terminal output.
4. Replace the `[INSERT NUMBER HERE]` placeholders in the report with these numbers.
5. Insert the generated images into the document (if using Word/LaTeX).

<!-- end list -->