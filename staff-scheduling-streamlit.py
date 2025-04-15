import streamlit as st
import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import csv
import os
import base64
from io import StringIO, BytesIO

# Set page configuration
st.set_page_config(
    page_title="Staff Scheduling System",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0D47A1;
        font-weight: 600;
        margin-top: 20px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'holidays' not in st.session_state:
        st.session_state.holidays = {}
    if 'personal_leaves' not in st.session_state:
        st.session_state.personal_leaves = {}
    if 'date_change_requests' not in st.session_state:
        st.session_state.date_change_requests = {}
    if 'schedule_result' not in st.session_state:
        st.session_state.schedule_result = None
    if 'start_date' not in st.session_state:
        st.session_state.start_date = datetime.now()


def generate_schedule(num_employees=15, num_seats=12, days_per_employee=4,
                     holidays=None, personal_leaves=None, date_change_requests=None):
    """
    Generate an optimal staff schedule with constraints:
    - 15 employees but only 12 seats available in the office
    - Each employee must be scheduled for exactly 4 days per week
    - Holidays and personal leaves count toward the 4 days
    - Support for date change requests from up to 30% of employees
    
    Parameters:
    - num_employees: Total number of employees (default: 15)
    - num_seats: Number of available seats (default: 12)
    - days_per_employee: Number of days each employee must work (default: 4)
    - holidays: Dictionary mapping days to list of employees affected by holidays
    - personal_leaves: Dictionary mapping employee to days they're on personal leave
    - date_change_requests: Dictionary of employee requests to change dates
    
    Returns:
    - Dictionary with schedule information
    """
    # Initialize empty dictionaries if None
    if holidays is None:
        holidays = {}
    if personal_leaves is None:
        personal_leaves = {}
    if date_change_requests is None:
        date_change_requests = {}
    
    # Create the optimization model
    model = pulp.LpProblem("Staff_Scheduling", pulp.LpMinimize)
    
    # Days of the week (0-4 for weekdays: Monday to Friday)
    days = list(range(5))
    
    # Create decision variables
    # x[i,j] = 1 if employee i comes to office on day j, 0 otherwise
    x = {}
    for i in range(num_employees):
        for j in days:
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
    
    # Calculate how many days each employee needs to be physically present
    # (4 minus holidays and personal leaves)
    days_in_office_required = {}
    for i in range(num_employees):
        # Count holidays for this employee
        holiday_days = sum(1 for day, emps in holidays.items() 
                          if i in emps or len(emps) == 0)  # Empty means all employees
        
        # Count personal leaves
        leave_days = len(personal_leaves.get(i, []))
        
        # Calculate required office days (minimum 0)
        days_in_office_required[i] = max(0, days_per_employee - holiday_days - leave_days)
    
    # Constraint: Each employee comes to office exactly the required number of days
    for i in range(num_employees):
        model += pulp.lpSum(x[i, j] for j in days) == days_in_office_required[i]
    
    # Constraint: No more than num_seats employees in the office on any day
    for j in days:
        model += pulp.lpSum(x[i, j] for i in range(num_employees)) <= num_seats
    
    # Constraint: Employees don't come on holidays or personal leaves
    for day, emps in holidays.items():
        if len(emps) == 0:  # If empty, it's a holiday for all employees
            emps = range(num_employees)
        
        for emp in emps:
            if 0 <= emp < num_employees and 0 <= day < 5:
                model += x[emp, day] == 0
    
    for emp, days_list in personal_leaves.items():
        for day in days_list:
            if 0 <= emp < num_employees and 0 <= day < 5:
                model += x[emp, day] == 0
    
    # Handle date change requests
    for emp, request in date_change_requests.items():
        days_to_remove = request.get('remove', [])
        days_to_add = request.get('add', [])
        
        # Ensure employee doesn't come on days requested to remove
        for day in days_to_remove:
            if 0 <= emp < num_employees and 0 <= day < 5:
                model += x[emp, day] == 0
        
        # Try to ensure employee comes on days requested to add
        # This is a soft constraint (might not always be possible)
        for day in days_to_add:
            if 0 <= emp < num_employees and 0 <= day < 5:
                # Check if this day is a holiday or personal leave
                is_holiday = any(day in holidays.keys() and (emp in holidays[day] or len(holidays[day]) == 0))
                is_leave = emp in personal_leaves and day in personal_leaves[emp]
                
                if not is_holiday and not is_leave:
                    # Create a preference for this assignment
                    request_penalty = pulp.LpVariable(f"req_penalty_{emp}_{day}", 0, 1, pulp.LpBinary)
                    model += x[emp, day] + request_penalty >= 1
                    model += request_penalty  # Add to objective to minimize
    
    # Objective: Balance the number of employees per day
    # Calculate employees per day
    employees_per_day = {}
    for j in days:
        employees_per_day[j] = pulp.lpSum(x[i, j] for i in range(num_employees))
    
    # Create variables for max and min employees
    max_employees = pulp.LpVariable("max_employees", lowBound=0)
    min_employees = pulp.LpVariable("min_employees", lowBound=0)
    
    # Set constraints for max and min
    for j in days:
        model += employees_per_day[j] <= max_employees
        model += employees_per_day[j] >= min_employees
    
    # Objective: Minimize the difference between max and min employees
    model += max_employees - min_employees
    
    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Check if a solution was found
    if pulp.LpStatus[model.status] != 'Optimal':
        return {"status": "No optimal solution found", "schedule": None}
    
    # Extract the solution
    schedule = {}
    days_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    for i in range(num_employees):
        employee_schedule = []
        # Days employee comes to office
        for j in days:
            if pulp.value(x[i, j]) == 1:
                employee_schedule.append(days_names[j])
        
        # Add holidays as part of the 4 days
        for day, emps in holidays.items():
            if (i in emps or len(emps) == 0) and 0 <= day < 5:
                employee_schedule.append(f"{days_names[day]} (Holiday)")
        
        # Add personal leaves as part of the 4 days
        if i in personal_leaves:
            for day in personal_leaves[i]:
                if 0 <= day < 5:
                    employee_schedule.append(f"{days_names[day]} (Leave)")
        
        schedule[f"Employee {i+1}"] = sorted(employee_schedule)
    
    # Count employees per day in office
    employees_in_office = {}
    for j in days:
        count = sum(1 for i in range(num_employees) if pulp.value(x[i, j]) == 1)
        employees_in_office[days_names[j]] = count
    
    return {
        "status": "Optimal solution found",
        "schedule": schedule,
        "employees_per_day": employees_in_office,
        "raw_schedule": [[pulp.value(x[i, j]) for j in days] for i in range(num_employees)]
    }


def process_date_change_request(current_requests, employee_id, days_to_remove=None, days_to_add=None,
                              num_employees=15):
    """
    Process a request from an employee to change their scheduled dates.
    Ensures that no more than 30% of employees can make date change requests.
    
    Parameters:
    - current_requests: Current dictionary of date change requests
    - employee_id: ID of the employee making the request (0-indexed)
    - days_to_remove: List of days to remove from schedule (0=Monday, 1=Tuesday, etc.)
    - days_to_add: List of days to add to schedule
    - num_employees: Total number of employees
    
    Returns:
    - Updated date_change_requests dictionary
    - Boolean indicating if the request was added successfully
    """
    if current_requests is None:
        current_requests = {}
    
    # Check if we've reached the limit (30% of employees)
    max_requests = int(num_employees * 0.3)
    if len(current_requests) >= max_requests and employee_id not in current_requests:
        return current_requests, False
    
    # Process the request
    if days_to_remove is None:
        days_to_remove = []
    if days_to_add is None:
        days_to_add = []
    
    if employee_id not in current_requests:
        current_requests[employee_id] = {'remove': [], 'add': []}
    
    # Update the request
    current_requests[employee_id]['remove'] = [day for day in days_to_remove if 0 <= day < 5]
    current_requests[employee_id]['add'] = [day for day in days_to_add if 0 <= day < 5]
    
    # If there are no changes, remove the employee from requests
    if not current_requests[employee_id]['remove'] and not current_requests[employee_id]['add']:
        current_requests.pop(employee_id)
    
    return current_requests, True


def add_holiday(holidays, day, employees=None):
    """
    Add a holiday for specific employees or all employees.
    
    Parameters:
    - holidays: Current dictionary of holidays
    - day: Day of holiday (0=Monday, 1=Tuesday, etc.)
    - employees: List of employee IDs affected, or None for all employees
    
    Returns:
    - Updated holidays dictionary
    """
    if holidays is None:
        holidays = {}
    
    if 0 <= day < 5:  # Ensure valid day
        if day not in holidays:
            holidays[day] = []
        
        if employees is None:
            # Holiday for all employees
            holidays[day] = []  # Empty list means all employees
        else:
            # Holiday for specific employees
            for emp in employees:
                if emp not in holidays[day]:
                    holidays[day].append(emp)
    
    return holidays


def add_personal_leave(personal_leaves, employee_id, day):
    """
    Add a personal leave for an employee.
    
    Parameters:
    - personal_leaves: Current dictionary of personal leaves
    - employee_id: ID of the employee taking leave
    - day: Day of leave (0=Monday, 1=Tuesday, etc.)
    
    Returns:
    - Updated personal_leaves dictionary
    """
    if personal_leaves is None:
        personal_leaves = {}
    
    if 0 <= day < 5:  # Ensure valid day
        if employee_id not in personal_leaves:
            personal_leaves[employee_id] = []
        
        if day not in personal_leaves[employee_id]:
            personal_leaves[employee_id].append(day)
    
    return personal_leaves


def get_weekly_date_range(start_date=None):
    """
    Get a range of weekday dates for a week starting from the given date.
    
    Parameters:
    - start_date: Starting date (default: today)
    
    Returns:
    - List of date strings for weekdays
    """
    if start_date is None:
        start_date = datetime.now()
    
    # Adjust to start on Monday
    weekday = start_date.weekday()
    monday = start_date - timedelta(days=weekday)
    
    # Generate weekdays
    weekdays = []
    for i in range(5):  # Monday to Friday
        date = monday + timedelta(days=i)
        weekdays.append(date.strftime("%Y-%m-%d"))
    
    return weekdays


def generate_schedule_with_dates(num_employees=15, num_seats=12, days_per_employee=4, 
                               start_date=None, holidays=None, personal_leaves=None, 
                               date_change_requests=None):
    """
    Generate a schedule with actual dates instead of weekday names.
    
    Parameters:
    - Same as generate_schedule, but with actual dates
    
    Returns:
    - Schedule with dates
    """
    # Get weekday dates
    weekday_dates = get_weekly_date_range(start_date)
    
    # Generate schedule
    result = generate_schedule(num_employees, num_seats, days_per_employee, 
                             holidays, personal_leaves, date_change_requests)
    
    if result["status"] != "Optimal solution found":
        return result
    
    # Replace weekday names with dates
    days_mapping = {
        "Monday": weekday_dates[0],
        "Tuesday": weekday_dates[1],
        "Wednesday": weekday_dates[2],
        "Thursday": weekday_dates[3],
        "Friday": weekday_dates[4]
    }
    
    # Update schedule with dates
    date_schedule = {}
    for emp, days in result["schedule"].items():
        date_schedule[emp] = []
        for day in days:
            if "(Holiday)" in day:
                weekday = day.split(" (")[0]
                date_schedule[emp].append(f"{days_mapping[weekday]} (Holiday)")
            elif "(Leave)" in day:
                weekday = day.split(" (")[0]
                date_schedule[emp].append(f"{days_mapping[weekday]} (Leave)")
            else:
                date_schedule[emp].append(days_mapping[day])
    
    # Update employees per day
    employees_per_date = {}
    for day, count in result["employees_per_day"].items():
        employees_per_date[days_mapping[day]] = count
    
    # Copy the raw schedule matrix
    raw_schedule = result["raw_schedule"] if "raw_schedule" in result else None
    
    return {
        "status": "Optimal solution found",
        "schedule": date_schedule,
        "employees_per_day": employees_per_date,
        "raw_schedule": raw_schedule
    }


def get_download_link(df, filename, text):
    """Generate a link to download a dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def create_schedule_heatmap(raw_schedule, num_employees):
    """Create a heatmap visualization of the schedule"""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    sns.heatmap(raw_schedule, cmap="YlGnBu", cbar=False, linewidths=.5, 
                xticklabels=days, yticklabels=[f"Emp {i+1}" for i in range(num_employees)],
                annot=True, fmt=".0f", ax=ax)
    
    # Set title and labels
    plt.title("Staff Schedule Heatmap (1 = Scheduled, 0 = Off)", fontsize=14)
    plt.xlabel("Days of Week", fontsize=12)
    plt.ylabel("Employees", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def display_sidebar():
    """Display the sidebar with configuration options"""
    with st.sidebar:
        st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
        
        num_employees = st.number_input("Number of Employees", min_value=1, max_value=50, value=15)
        num_seats = st.number_input("Number of Available Seats", min_value=1, max_value=num_employees, value=min(12, num_employees))
        days_per_employee = st.number_input("Days Per Employee", min_value=1, max_value=5, value=4)
        
        st.session_state.start_date = st.date_input(
            "Schedule Start Date",
            value=datetime.now()
        )
        
        # Calculate the 30% limit
        max_requests = int(num_employees * 0.3)
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Current Settings:</strong><br>
            Employees: {num_employees}<br>
            Seats: {num_seats}<br>
            Days/Employee: {days_per_employee}<br>
            Date Change Request Limit: {max_requests} employees
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Reset All Data", type="primary"):
            st.session_state.holidays = {}
            st.session_state.personal_leaves = {}
            st.session_state.date_change_requests = {}
            st.session_state.schedule_result = None
            st.success("All data has been reset!")
    
    return num_employees, num_seats, days_per_employee, max_requests


def holidays_section(num_employees):
    """Display the holidays section"""
    st.markdown('<p class="section-header">National Holidays</p>', unsafe_allow_html=True)
    
    with st.expander("Add Holidays", expanded=False):
        col1, col2 = st.columns([2, 3])
        
        with col1:
            day_mapping = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4
            }
            
            holiday_day = st.selectbox("Select Day", list(day_mapping.keys()), key="holiday_day")
            day_idx = day_mapping[holiday_day]
            
            all_employees = st.checkbox("Holiday for All Employees", value=True, key="all_emp_holiday")
            
            if not all_employees:
                # Convert to 1-indexed for display, but store as 0-indexed
                holiday_employees = st.multiselect(
                    "Select Employees",
                    [f"Employee {i+1}" for i in range(num_employees)],
                    key="holiday_employees"
                )
                # Convert back to 0-indexed
                holiday_emp_indices = [int(emp.split(" ")[1]) - 1 for emp in holiday_employees]
            else:
                holiday_emp_indices = None
        
        with col2:
            if st.button("Add Holiday", key="add_holiday_btn"):
                st.session_state.holidays = add_holiday(
                    st.session_state.holidays,
                    day_idx,
                    holiday_emp_indices
                )
                st.success(f"Holiday added for {holiday_day}!")
            
            # Display current holidays
            if st.session_state.holidays:
                st.markdown("**Current Holidays:**")
                for day, emps in st.session_state.holidays.items():
                    day_name = list(day_mapping.keys())[day]
                    if len(emps) == 0:
                        st.markdown(f"- {day_name}: All employees")
                    else:
                        emp_list = ", ".join([f"Employee {e+1}" for e in emps])
                        st.markdown(f"- {day_name}: {emp_list}")
                
                if st.button("Clear Holidays", key="clear_holidays"):
                    st.session_state.holidays = {}
                    st.success("All holidays cleared!")
            else:
                st.info("No holidays added yet.")


def personal_leaves_section(num_employees):
    """Display the personal leaves section"""
    st.markdown('<p class="section-header">Personal Leaves</p>', unsafe_allow_html=True)
    
    with st.expander("Add Personal Leaves", expanded=False):
        col1, col2 = st.columns([2, 3])
        
        with col1:
            day_mapping = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4
            }
            
            # Employee selection (1-indexed for display)
            leave_employee = st.selectbox(
                "Select Employee",
                [f"Employee {i+1}" for i in range(num_employees)],
                key="leave_employee"
            )
            # Convert to 0-indexed for processing
            leave_emp_idx = int(leave_employee.split(" ")[1]) - 1
            
            leave_day = st.selectbox("Select Day", list(day_mapping.keys()), key="leave_day")
            leave_day_idx = day_mapping[leave_day]
        
        with col2:
            if st.button("Add Personal Leave", key="add_leave_btn"):
                st.session_state.personal_leaves = add_personal_leave(
                    st.session_state.personal_leaves,
                    leave_emp_idx,
                    leave_day_idx
                )
                st.success(f"Personal leave added for {leave_employee} on {leave_day}!")
            
            # Display current personal leaves
            if st.session_state.personal_leaves:
                st.markdown("**Current Personal Leaves:**")
                for emp, days in st.session_state.personal_leaves.items():
                    emp_name = f"Employee {emp+1}"
                    day_names = [list(day_mapping.keys())[day] for day in days]
                    st.markdown(f"- {emp_name}: {', '.join(day_names)}")
                
                if st.button("Clear Personal Leaves", key="clear_leaves"):
                    st.session_state.personal_leaves = {}
                    st.success("All personal leaves cleared!")
            else:
                st.info("No personal leaves added yet.")


def date_change_requests_section(num_employees, max_requests):
    """Display the date change requests section"""
    st.markdown('<p class="section-header">Schedule Change Requests</p>', unsafe_allow_html=True)
    
    with st.expander("Add Schedule Change Requests", expanded=False):
        num_current_requests = len(st.session_state.date_change_requests)
        
        # Display the current status
        st.markdown(f"""
        <div class="info-box">
            Schedule change requests: {num_current_requests}/{max_requests} (30% limit)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            day_mapping = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4
            }
            
            # Employee selection (1-indexed for display)
            request_employee = st.selectbox(
                "Select Employee",
                [f"Employee {i+1}" for i in range(num_employees)],
                key="request_employee"
            )
            # Convert to 0-indexed for processing
            request_emp_idx = int(request_employee.split(" ")[1]) - 1
            
            days_to_remove = st.multiselect(
                "Days to Remove",
                list(day_mapping.keys()),
                key="days_to_remove"
            )
            remove_indices = [day_mapping[day] for day in days_to_remove]
            
            days_to_add = st.multiselect(
                "Days to Add",
                list(day_mapping.keys()),
                key="days_to_add"
            )
            add_indices = [day_mapping[day] for day in days_to_add]
        
        with col2:
            if st.button("Submit Request", key="add_request_btn"):
                updated_requests, success = process_date_change_request(
                    st.session_state.date_change_requests,
                    request_emp_idx,
                    remove_indices,
                    add_indices,
                    num_employees
                )
                
                if success:
                    st.session_state.date_change_requests = updated_requests
                    st.success(f"Schedule change request added for {request_employee}!")
                else:
                    st.error(f"Cannot add request. Maximum of {max_requests} employees (30%) reached!")
            
            # Display current requests
            if st.session_state.date_change_requests:
                st.markdown("**Current Schedule Change Requests:**")
                for emp, request in st.session_state.date_change_requests.items():
                    emp_name = f"Employee {emp+1}"
                    
                    remove_days = [list(day_mapping.keys())[day] for day in request['remove']]
                    remove_str = ", ".join(remove_days) if remove_days else "None"
                    
                    add_days = [list(day_mapping.keys())[day] for day in request['add']]
                    add_str = ", ".join(add_days) if add_days else "None"
                    
                    st.markdown(f"- {emp_name}: Remove: {remove_str}, Add: {add_str}")
                
                if st.button("Clear All Requests", key="clear_requests"):
                    st.session_state.date_change_requests = {}
                    st.success("All schedule change requests cleared!")
            else:
                st.info("No schedule change requests added yet.")


def generate_schedule_section(num_employees, num_seats, days_per_employee):
    """Display the schedule generation section"""
    st.markdown('<p class="section-header">Generate Schedule</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        generate_btn = st.button("Generate Schedule", type="primary", key="generate_schedule_btn")
        
        if generate_btn:
            with st.spinner("Generating optimal schedule..."):
                result = generate_schedule_with_dates(
                    num_employees=num_employees,
                    num_seats=num_seats,
                    days_per_employee=days_per_employee,
                    start_date=st.session_state.start_date,
                    holidays=st.session_state.holidays,
                    personal_leaves=st.session_state.personal_leaves,
                    date_change_requests=st.session_state.date_change_requests
                )
                
                st.session_state.schedule_result = result
            
            if result["status"] == "Optimal solution found":
                st.success("Schedule generated successfully!")
            else:
                st.error(f"Could not generate schedule: {result['status']}")
    
    with col2:
        # Display summary statistics
        if st.session_state.schedule_result and st.session_state.schedule_result["status"] == "Optimal solution found":
            st.markdown("**Schedule Summary:**")
            
            employees_per_day = st.session_state.schedule_result["employees_per_day"]
            total_employees = sum(employees_per_day.values())
            
            st.markdown(f"""
            <div class="success-box">
                <strong>Total scheduled employee-days:</strong> {total_employees}<br>
                <strong>Average employees per day:</strong> {total_employees/5:.1f}<br>
                <strong>Days with maximum capacity ({num_seats} employees):</strong> {sum(1 for count in employees_per_day.values() if count == num_seats)}
            </div>
            """, unsafe_allow_html=True)


def display_schedule_results():
    """Display the schedule results"""
    if not st.session_state.schedule_result or st.session_state.schedule_result["status"] != "Optimal solution found":
        return
    
    st.markdown('<p class="section-header">Schedule Results</p>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Employee Schedule", "Daily Schedule", "Visualization"])
    
    with tab1:
        # Convert schedule to DataFrame for display
        schedule_data = []
        for employee, days in st.session_state.schedule_result["schedule"].items():
            schedule_data.append({
                "Employee": employee,
                "Scheduled Days": ", ".join(days)
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        st.dataframe(schedule_df, use_container_width=True)
        
        # Download link
        st.markdown(
            get_download_link(schedule_df, "employee_schedule.csv", "Download Employee Schedule CSV"),
            unsafe_allow_html=True
        )
    
    with tab2:
        # Create a cross-tab view (which employees on which days)
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        daily_schedule = []
        
        for day_name in days_of_week:
            # Find the date for this day from the keys in employees_per_day
            day_date = next((date for date in st.session_state.schedule_result["employees_per_day"].keys() 
                            if day_name in date), day_name)
            
            # Get the count
            count = st.session_state.schedule_result["employees_per_day"].get(day_date, 0)
            
            # Find employees working this day
            employees = []
            for emp, days in st.session_state.schedule_result["schedule"].items():
                if any(day_date in day for day in days):
                    employees.append(emp)
            
            daily_schedule.append({
                "Day": day_date,
                "Employee Count": count,
                "Employees": ", ".join(employees)
            })
        
        daily_df = pd.DataFrame(daily_schedule)
        st.dataframe(daily_df, use_container_width=True)
        
        # Download link
        st.markdown(
            get_download_link(daily_df, "daily_schedule.csv", "Download Daily Schedule CSV"),
            unsafe_allow_html=True
        )
    
    with tab3:
        if "raw_schedule" in st.session_state.schedule_result and st.session_state.schedule_result["raw_schedule"]:
            # Create a heatmap
            raw_schedule = st.session_state.schedule_result["raw_schedule"]
            num_employees = len(raw_schedule)
            
            fig = create_schedule_heatmap(raw_schedule, num_employees)
            st.pyplot(fig)
            
            # Create a bar chart of employees per day
            st.markdown("#### Employees Per Day")
            employees_per_day = st.session_state.schedule_result["employees_per_day"]
            
            employees_df = pd.DataFrame({
                "Day": list(employees_per_day.keys()),
                "Employees": list(employees_per_day.values())
            })
            
            # Sort by day of week
            day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4}
            employees_df["SortOrder"] = employees_df["Day"].apply(
                lambda x: next((idx for day, idx in day_order.items() if day in x), 999)
            )
            employees_df = employees_df.sort_values("SortOrder").drop("SortOrder", axis=1)
            
            st.bar_chart(employees_df.set_index("Day"))
        else:
            st.info("Raw schedule data not available for visualization.")


def main():
    """Main function to run the Streamlit app"""
    # Initialize session state
    initialize_session_state()
    
    # Display page title and description
    st.markdown('<p class="main-header">Staff Scheduling System</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This application helps you create an optimal staff schedule with the following constraints:
    * Each employee works exactly 4 days per week
    * Limited office capacity (fewer seats than employees)
    * National holidays and personal leaves count toward the 4-day requirement
    * Support for schedule change requests from employees (limited to 30%)
    """)
    
    # Display sidebar and get configuration
    num_employees, num_seats, days_per_employee, max_requests = display_sidebar()
    
    # Display main sections
    holidays_section(num_employees)
    st.markdown("---")
    
    personal_leaves_section(num_employees)
    st.markdown("---")
    
    date_change_requests_section(num_employees, max_requests)
    st.markdown("---")
    
    generate_schedule_section(num_employees, num_seats, days_per_employee)
    
    # Display results if schedule is generated
    display_schedule_results()


# Run the app
if __name__ == "__main__":
    main()
