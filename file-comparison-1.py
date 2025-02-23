import pandas as pd
import os
import shutil
from datetime import datetime, timedelta
import numpy as np

def generate_sample_data():
    """Generate sample data for testing"""
    current_data = {
        'loan_number': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
        'qc_score': [95, None, 88, 92, None, 85, 90],
        'status': ['Completed', 'In Progress', 'Completed', 'Completed', 'In Progress', 'Completed', 'In Progress'],
        'inspection_form': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
        'date_b': ['2024-02-01', None, '2024-02-03', '2024-02-04', None, '2024-02-06', '2024-02-07'],
        'date_d': ['2024-02-02', None, '2024-02-04', '2024-02-05', None, '2024-02-07', '2024-02-08'],
        'date_e': ['2024-02-03', None, '2024-02-05', '2024-02-06', None, '2024-02-08', '2024-02-09'],
        'bankruptcy': ['Yes', '', 'No', 'Yes', '', 'No', 'Yes'],
        'multisequence': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']
    }
    
    last_week_data = {
        'loan_number': [1001, 1002, 1008, 1004, 1009],
        'qc_score': [95, None, 87, 90, None],
        'status': ['Completed', 'In Progress', 'Completed', 'In Progress', 'In Progress'],
        'inspection_form': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'date_b': ['2024-02-01', None, '2024-01-25', '2024-01-26', None],
        'date_d': ['2024-02-02', None, '2024-01-26', '2024-01-27', None],
        'date_e': ['2024-02-03', None, '2024-01-27', '2024-01-28', None],
        'bankruptcy': ['Yes', '', 'No', 'No', ''],
        'multisequence': ['Yes', 'No', 'No', 'No', 'Yes']
    }
    
    return pd.DataFrame(current_data), pd.DataFrame(last_week_data)

def setup_folders():
    """Create necessary folders if they don't exist"""
    folders = ['output', 'comparison', 'backup']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_week_info():
    """Generate week information for file naming"""
    current_date = datetime.now()
    week_number = current_date.isocalendar()[1]
    year = current_date.year
    timestamp = current_date.strftime("%Y%m%d_%H%M%S")
    return year, week_number, timestamp

def generate_filename(year, week, timestamp):
    """Generate standardized filename with week information"""
    return f"weekly_report_Y{year}W{week:02d}_{timestamp}.csv"

def get_week_from_filename(filename):
    """Extract year and week number from filename"""
    try:
        # Extract year and week from filename (format: weekly_report_Y2024W06_20240211_120000.csv)
        parts = filename.split('_')
        year_week = parts[2]  # Y2024W06
        year = int(year_week[1:5])
        week = int(year_week[6:8])
        return year, week
    except (IndexError, ValueError):
        return None, None

def move_and_rename_file(source_path, dest_folder):
    """Move file from output to comparison folder with year, week, and timestamp"""
    year, week, timestamp = get_week_info()
    filename = generate_filename(year, week, timestamp)
    dest_path = os.path.join(dest_folder, filename)
    shutil.copy2(source_path, dest_path)
    return dest_path

def get_last_week_file(folder, current_year, current_week):
    """Get the file from the previous week"""
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    last_week_files = []
    
    for file in files:
        year, week = get_week_from_filename(file)
        if year is None or week is None:
            continue
            
        # Calculate previous week number
        prev_week = current_week - 1
        prev_year = current_year
        if prev_week == 0:
            prev_week = 52
            prev_year -= 1
            
        if year == prev_year and week == prev_week:
            last_week_files.append(file)
    
    if not last_week_files:
        return None
        
    # Return the most recent file if multiple files exist for the same week
    return max(last_week_files, key=lambda x: os.path.getctime(os.path.join(folder, x)))

def handle_rerun(comparison_folder, backup_folder, current_year, current_week):
    """Handle rerun scenario by cleaning up and restoring from backup"""
    # Remove existing files from the current week
    for file in os.listdir(comparison_folder):
        year, week = get_week_from_filename(file)
        if year == current_year and week == current_week:
            os.remove(os.path.join(comparison_folder, file))
    
    # Get last week's file from backup
    last_week_file = get_last_week_file(backup_folder, current_year, current_week)
    if last_week_file:
        shutil.copy2(
            os.path.join(backup_folder, last_week_file),
            os.path.join(comparison_folder, last_week_file)
        )
        return os.path.join(comparison_folder, last_week_file)
    return None

def analyze_files(current_df, last_week_df):
    """Perform analysis and comparison between current and last week's data"""
    # [Previous analyze_files function remains the same]
    analysis_results = {}
    
    # 6. Count of records with non-null QC score and new records
    current_non_null_qc = current_df[current_df['qc_score'].notna()]
    last_week_non_null_qc = last_week_df[last_week_df['qc_score'].notna()]
    
    new_loans = set(current_df['loan_number']) - set(last_week_df['loan_number'])
    new_loans_status = current_df[current_df['loan_number'].isin(new_loans)][['loan_number', 'status', 'qc_score']]
    
    analysis_results['non_null_qc_count'] = len(current_non_null_qc)
    analysis_results['new_loans'] = new_loans_status.to_dict('records')
    
    # [Rest of the analysis logic remains the same]
    return analysis_results

def print_analysis_results(results, current_year, current_week):
    """Print analysis results in a formatted way"""
    print(f"\n=== Analysis Results for Year {current_year} Week {current_week} ===")
    # [Rest of the print logic remains the same]
    print(f"\nNon-null QC Score Count: {results['non_null_qc_count']}")
    print("\nNew Loans This Week:")
    for loan in results['new_loans']:
        print(f"Loan: {loan['loan_number']}, Status: {loan['status']}, QC Score: {loan['qc_score']}")
    
    # [Rest of the print logic remains the same]

def main():
    """Main function to orchestrate the entire process"""
    # Setup folders
    setup_folders()
    
    # Generate sample data and save to files
    current_df, last_week_df = generate_sample_data()
    current_file = "output/current_week.csv"
    current_df.to_csv(current_file, index=False)
    
    # Get current week information
    current_year, current_week, _ = get_week_info()
    
    # Move and rename file
    new_file_path = move_and_rename_file(current_file, "comparison")
    
    # Handle rerun if needed
    last_week_file = handle_rerun("comparison", "backup", current_year, current_week)
    
    if last_week_file is None:
        print(f"No file found for Year {current_year} Week {current_week - 1}")
        return
    
    # Load data
    current_week_data = pd.read_csv(new_file_path)
    last_week_data = pd.read_csv(last_week_file)
    
    # Perform analysis
    analysis_results = analyze_files(current_week_data, last_week_data)
    
    # Print results
    print_analysis_results(analysis_results, current_year, current_week)
    
    # Move last week's file to backup
    shutil.move(last_week_file, os.path.join("backup", os.path.basename(last_week_file)))

if __name__ == "__main__":
    main()
