import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime, timedelta
import numpy as np
import os

# Create necessary directories
comparison_folder = Path('Comparison')
backup_folder = Path('Backup')
comparison_folder.mkdir(exist_ok=True)
backup_folder.mkdir(exist_ok=True)

# Generate sample data for last week and current week
def generate_sample_data():
    # Last week's data
    last_week_data = {
        'LoanNumber': ['Loan1', 'Loan2', 'Loan3', 'Loan4'],
        'QCScore': [80, np.nan, 90, 70],
        'Status': ['In Progress', 'Pending', 'Completed', 'Completed'],
        'InspectionForm': ['Form1', np.nan, 'Form3', 'Form4'],
        'InspectionDate': ['2023-10-01', np.nan, '2023-10-02', '2023-10-03'],
        'StartDate': ['2023-10-01', np.nan, '2023-10-02', '2023-10-03'],
        'EndDate': ['2023-10-10', np.nan, '2023-10-12', '2023-10-13'],
        'Bankruptcy': ['Yes', np.nan, 'No', 'Yes'],
        'Multisequence': ['No', 'No', 'Yes', 'No']
    }
    last_week_df = pd.DataFrame(last_week_data)
    
    # Current week's data
    current_week_data = {
        'LoanNumber': ['Loan1', 'Loan2', 'Loan3', 'Loan5', 'Loan6'],
        'QCScore': [80, 85, 90, 95, np.nan],
        'Status': ['Completed', 'In Progress', 'Completed', 'Completed', 'Pending'],
        'InspectionForm': ['Form1', 'Form2', 'Form3', 'Form5', np.nan],
        'InspectionDate': [np.nan, '2023-10-05', '2023-10-02', '2023-10-04', np.nan],
        'StartDate': ['2023-10-01', '2023-10-05', '2023-10-02', '2023-10-04', np.nan],
        'EndDate': ['2023-10-10', '2023-10-15', '2023-10-12', '2023-10-14', np.nan],
        'Bankruptcy': ['Yes', '', 'No', 'No', 'Yes'],
        'Multisequence': ['No', 'No', 'Yes', 'Yes', 'No']
    }
    current_week_df = pd.DataFrame(current_week_data)
    
    return last_week_df, current_week_df

# Function to parse datetime from filename
def parse_filename_datetime(filename):
    try:
        date_part = '_'.join(filename.stem.split('_')[1:3])
        return datetime.strptime(date_part, "%Y%m%d_%H%M%S")
    except:
        return None

# Function to handle file movement and renaming
def process_current_week_file(current_df):
    # Save current week's data to a temporary output file
    output_location = Path('temp_output.csv')
    current_df.to_csv(output_location, index=False)
    
    # Generate current datetime for filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_filename = f"data_{current_time}.csv"
    
    # Check and delete existing files from the same week in Comparison
    current_file_week = datetime.now().isocalendar()[1]
    current_file_year = datetime.now().year
    for file in comparison_folder.glob('*.csv'):
        file_dt = parse_filename_datetime(file)
        if file_dt:
            file_week = file_dt.isocalendar()[1]
            file_year = file_dt.year
            if file_week == current_file_week and file_year == current_file_year:
                os.remove(file)
                print(f"Deleted existing file from same week: {file}")
    
    # Move the current file to Comparison
    new_current_path = comparison_folder / current_filename
    shutil.move(output_location, new_current_path)
    print(f"Moved current week's file to Comparison: {new_current_path}")
    return new_current_path

# Function to find last week's file
def find_last_week_file(current_file_dt):
    all_files = list(comparison_folder.glob('*.csv')) + list(backup_folder.glob('*.csv'))
    last_week_file = None
    latest_date = None
    for file in all_files:
        file_dt = parse_filename_datetime(file)
        if file_dt and file_dt < current_file_dt:
            if (latest_date is None) or (file_dt > latest_date):
                latest_date = file_dt
                last_week_file = file
    return last_week_file

# Function to perform data comparisons
def perform_comparisons(current_df, last_df):
    results = {}
    
    # Check 1: Count of QC not null
    results['current_qc_not_null'] = current_df['QCScore'].notnull().sum()
    results['last_qc_not_null'] = last_df['QCScore'].notnull().sum()
    
    # Check 2: New records
    new_loans = current_df[~current_df['LoanNumber'].isin(last_df['LoanNumber'])]
    results['new_loan_count'] = len(new_loans)
    results['new_loan_numbers'] = new_loans['LoanNumber'].tolist()
    
    # Check 3: QC null count
    results['current_qc_null'] = current_df['QCScore'].isnull().sum()
    
    # Check 4: Newly completed loans
    current_completed = current_df[current_df['Status'] == 'Completed']
    last_completed_loans = last_df[last_df['Status'] == 'Completed']['LoanNumber']
    newly_completed = current_completed[~current_completed['LoanNumber'].isin(last_completed_loans)]
    results['number_newly_completed'] = len(newly_completed)
    results['loans_newly_completed'] = newly_completed['LoanNumber'].tolist()
    results['qc_check_newly_completed'] = newly_completed['QCScore'].notnull().all()
    
    # Check 5: Inspection form for non-null QC
    non_null_qc = current_df[current_df['QCScore'].notnull()]
    missing_inspection = non_null_qc[non_null_qc['InspectionForm'].isnull()]
    results['count_missing_inspection'] = len(missing_inspection)
    
    # Check 6: Dates for completed inspections
    completed_inspections = current_df[current_df['Status'] == 'Completed']
    date_cols = ['InspectionDate', 'StartDate', 'EndDate']
    missing_dates = completed_inspections[completed_inspections[date_cols].isnull().any(axis=1)]
    results['count_missing_dates'] = len(missing_dates)
    
    # Check 7: Bankruptcy yes or blank
    results['bankruptcy_yes_blank'] = current_df['Bankruptcy'].isin(['Yes', '']).sum()
    
    # Check 8: Multisequence yes distinct loans
    results['multisequence_yes_count'] = current_df[current_df['Multisequence'] == 'Yes']['LoanNumber'].nunique()
    
    return results

# Main process
def main():
    # Generate sample data
    last_week_df, current_week_df = generate_sample_data()
    
    # Save last week's file to Backup with a past datetime
    last_week_time = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d_%H%M%S")
    last_week_filename = f"data_{last_week_time}.csv"
    last_week_path = backup_folder / last_week_filename
    last_week_df.to_csv(last_week_path, index=False)
    print(f"Sample last week's file created in Backup: {last_week_path}")
    
    # Process current week's file
    current_file_path = process_current_week_file(current_week_df)
    current_file_dt = parse_filename_datetime(current_file_path)
    
    # Find last week's file
    last_week_file = find_last_week_file(current_file_dt)
    if not last_week_file:
        print("No last week's file found. Exiting.")
        return
    
    print(f"Last week's file found: {last_week_file}")
    last_week_data = pd.read_csv(last_week_file)
    current_week_data = pd.read_csv(current_file_path)
    
    # Perform comparisons
    results = perform_comparisons(current_week_data, last_week_data)
    
    # Print results
    print("\nComparison Results:")
    print(f"1. Current QC Not Null Count: {results['current_qc_not_null']} (Last Week: {results['last_qc_not_null']})")
    print(f"2. New Loans Added: {results['new_loan_count']} - {results['new_loan_numbers']}")
    print(f"3. Current QC Null Count: {results['current_qc_null']}")
    print(f"4. Newly Completed Loans: {results['number_newly_completed']} - {results['loans_newly_completed']}")
    print(f"   All new completed loans have QC Scores: {results['qc_check_newly_completed']}")
    print(f"5. Missing Inspection Forms: {results['count_missing_inspection']}")
    print(f"6. Completed Loans with Missing Dates: {results['count_missing_dates']}")
    print(f"7. Bankruptcy Yes/Blank Count: {results['bankruptcy_yes_blank']}")
    print(f"8. Distinct Loans with Multisequence Yes: {results['multisequence_yes_count']}")
    
    # Move last week's file to Backup if it was in Comparison
    if last_week_file.parent == comparison_folder:
        shutil.move(last_week_file, backup_folder / last_week_file.name)
        print(f"\nMoved last week's file to Backup: {last_week_file}")

if __name__ == "__main__":
    main()