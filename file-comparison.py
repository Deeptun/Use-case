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

def get_timestamp():
    """Generate timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def move_and_rename_file(source_path, dest_folder):
    """Move file from output to comparison folder with timestamp"""
    timestamp = get_timestamp()
    filename = f"weekly_report_{timestamp}.csv"
    dest_path = os.path.join(dest_folder, filename)
    shutil.copy2(source_path, dest_path)
    return dest_path

def get_latest_file(folder):
    """Get the most recent file from a folder"""
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))

def handle_rerun(comparison_folder, backup_folder, current_week_timestamp):
    """Handle rerun scenario by cleaning up and restoring from backup"""
    # Remove existing files from the same week
    for file in os.listdir(comparison_folder):
        if current_week_timestamp in file:
            os.remove(os.path.join(comparison_folder, file))
    
    # Get last week's file from backup
    latest_backup = get_latest_file(backup_folder)
    if latest_backup:
        shutil.copy2(
            os.path.join(backup_folder, latest_backup),
            os.path.join(comparison_folder, latest_backup)
        )
        return os.path.join(comparison_folder, latest_backup)
    return None

def analyze_files(current_df, last_week_df):
    """Perform analysis and comparison between current and last week's data"""
    analysis_results = {}
    
    # 6. Count of records with non-null QC score and new records
    current_non_null_qc = current_df[current_df['qc_score'].notna()]
    last_week_non_null_qc = last_week_df[last_week_df['qc_score'].notna()]
    
    new_loans = set(current_df['loan_number']) - set(last_week_df['loan_number'])
    new_loans_status = current_df[current_df['loan_number'].isin(new_loans)][['loan_number', 'status', 'qc_score']]
    
    analysis_results['non_null_qc_count'] = len(current_non_null_qc)
    analysis_results['new_loans'] = new_loans_status.to_dict('records')
    
    # 7. Count of NULL QC scores
    analysis_results['null_qc_count'] = len(current_df[current_df['qc_score'].isna()])
    
    # 8. New completed loans
    current_completed = set(current_df[current_df['status'] == 'Completed']['loan_number'])
    last_week_completed = set(last_week_df[last_week_df['status'] == 'Completed']['loan_number'])
    new_completed = current_completed - last_week_completed
    
    new_completed_data = current_df[current_df['loan_number'].isin(new_completed)][['loan_number', 'qc_score']]
    analysis_results['new_completed'] = new_completed_data.to_dict('records')
    
    # 9. Check inspection form data for non-null QC scores
    missing_inspection = current_df[
        (current_df['qc_score'].notna()) & 
        (current_df['inspection_form'] != 'Yes')
    ]['loan_number'].tolist()
    analysis_results['missing_inspection'] = missing_inspection
    
    # 10. Check date fields for completed inspections
    missing_dates = current_df[
        (current_df['inspection_form'] == 'Yes') & 
        (current_df[['date_b', 'date_d', 'date_e']].isna().any(axis=1))
    ]['loan_number'].tolist()
    analysis_results['missing_dates'] = missing_dates
    
    # 11. Count of bankruptcy records
    bankruptcy_yes = len(current_df[current_df['bankruptcy'] == 'Yes'])
    bankruptcy_blank = len(current_df[current_df['bankruptcy'] == ''])
    analysis_results['bankruptcy'] = {'yes': bankruptcy_yes, 'blank': bankruptcy_blank}
    
    # 12. Count of distinct loans with multisequence
    multisequence_count = len(current_df[current_df['multisequence'] == 'Yes']['loan_number'].unique())
    analysis_results['multisequence_count'] = multisequence_count
    
    return analysis_results

def print_analysis_results(results):
    """Print analysis results in a formatted way"""
    print("\n=== Analysis Results ===")
    print(f"\nNon-null QC Score Count: {results['non_null_qc_count']}")
    print("\nNew Loans This Week:")
    for loan in results['new_loans']:
        print(f"Loan: {loan['loan_number']}, Status: {loan['status']}, QC Score: {loan['qc_score']}")
    
    print(f"\nNull QC Score Count: {results['null_qc_count']}")
    
    print("\nNewly Completed Loans:")
    for loan in results['new_completed']:
        print(f"Loan: {loan['loan_number']}, QC Score: {loan['qc_score']}")
    
    print("\nLoans Missing Inspection Form:")
    print(results['missing_inspection'])
    
    print("\nCompleted Inspections Missing Dates:")
    print(results['missing_dates'])
    
    print("\nBankruptcy Information:")
    print(f"Yes: {results['bankruptcy']['yes']}")
    print(f"Blank: {results['bankruptcy']['blank']}")
    
    print(f"\nDistinct Loans with Multisequence: {results['multisequence_count']}")

def main():
    """Main function to orchestrate the entire process"""
    # Setup folders
    setup_folders()
    
    # Generate sample data and save to files
    current_df, last_week_df = generate_sample_data()
    current_file = "output/current_week.csv"
    current_df.to_csv(current_file, index=False)
    
    # Move and rename file
    new_file_path = move_and_rename_file(current_file, "comparison")
    
    # Handle rerun if needed
    timestamp = get_timestamp()
    last_week_file = handle_rerun("comparison", "backup", timestamp)
    
    if last_week_file is None:
        print("No previous week's file found.")
        return
    
    # Load data
    current_week_data = pd.read_csv(new_file_path)
    last_week_data = pd.read_csv(last_week_file)
    
    # Perform analysis
    analysis_results = analyze_files(current_week_data, last_week_data)
    
    # Print results
    print_analysis_results(analysis_results)
    
    # Move last week's file to backup
    shutil.move(last_week_file, os.path.join("backup", os.path.basename(last_week_file)))

if __name__ == "__main__":
    main()
