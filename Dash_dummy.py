import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data generation function
def generate_sample_data():
    """Generate sample financial data similar to the provided screenshot"""
    np.random.seed(42)
    
    facilities = []
    for i in range(100):
        facilities.append({
            'As of Date': '09/30/2024',
            'Facility ID': f'000000{i:02d}',
            'UCN_EQ': f'000000{i:02d}',
            'Client Name': f'Client_{chr(65 + i % 26)}{i:03d}',
            'Status': np.random.choice(['Pending Commitments', 'Pending Com Approved'], p=[0.3, 0.7]),
            'RBU Level': 'CCBSI',
            'Industry Division': np.random.choice(['Banks & Finance', 'Consumer & Retail', 'Healthcare', 'Real Estate']),
            'Industry Class': np.random.choice(['Co Lessors', 'Arts & Culture', 'Business Services', 'Medical Equipment Mfg']),
            'Default Grade': np.random.choice(['6+', '5+', '6', '5', '4', '7'], p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.15]),
            'Obligor Grade': np.random.choice(['6+', '5+', '6', '5', '4', '7'], p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.15]),
            'RCE': '####',
            'Loans': '$0',
            'Used Exposure': np.random.randint(0, 1000) * 1000,
            'Unused Exposure': np.random.randint(100, 5000) * 10000,
            'C&C': np.random.choice(['Y', 'N'], p=[0.3, 0.7]),
            'NPL': 'N',
            'Revolver Flag': np.random.choice(['R', 'NR'], p=[0.6, 0.4]),
            'Facility Type': np.random.choice(['COMMITMENT-TERM LOAN', 'COMMITMENT-REVOL/LONG'])
        })
    
    return pd.DataFrame(facilities)

def preprocess_query(query):
    """Preprocess the natural language query"""
    query = query.lower().strip()
    # Remove common words and normalize
    query = re.sub(r'\b(what|is|the|and|of|in|for|by|with|total|show|me|give|calculate)\b', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query

def extract_metrics_from_query(query):
    """Extract metrics requested in the query using regex patterns"""
    metrics = []
    
    # Define metric patterns
    metric_patterns = {
        'exposure': r'\b(exposure|exposed|outstanding)\b',
        'used_exposure': r'\b(used\s*exposure|utilized\s*exposure)\b',
        'unused_exposure': r'\b(unused\s*exposure|unutilized\s*exposure|available\s*exposure)\b',
        'line_utilization': r'\b(line\s*utilization|utilization\s*rate|usage\s*rate)\b',
        'total_exposure': r'\b(total\s*exposure|aggregate\s*exposure)\b',
        'loans': r'\b(loans|loan\s*amount)\b',
        'facility_count': r'\b(count|number\s*of\s*facilities|facility\s*count)\b'
    }
    
    for metric, pattern in metric_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            metrics.append(metric)
    
    return metrics if metrics else ['total_exposure', 'line_utilization']  # Default metrics

def extract_filters_from_query(query):
    """Extract filters from the query"""
    filters = {}
    
    # Industry patterns
    industry_pattern = r'\b(banks?|finance|retail|healthcare|real\s*estate|consumer)\b'
    if re.search(industry_pattern, query, re.IGNORECASE):
        match = re.search(industry_pattern, query, re.IGNORECASE)
        industry_map = {
            'bank': 'Banks & Finance',
            'banks': 'Banks & Finance',
            'finance': 'Banks & Finance',
            'retail': 'Consumer & Retail',
            'consumer': 'Consumer & Retail',
            'healthcare': 'Healthcare',
            'real estate': 'Real Estate'
        }
        industry_key = match.group().lower().replace(' ', ' ')
        if industry_key in industry_map:
            filters['Industry Division'] = industry_map[industry_key]
    
    # Grade patterns
    grade_pattern = r'\bgrade\s*([456]\+?|[567])\b'
    if re.search(grade_pattern, query, re.IGNORECASE):
        match = re.search(grade_pattern, query, re.IGNORECASE)
        filters['Default Grade'] = match.group(1)
    
    # Status patterns
    if re.search(r'\bpending\b', query, re.IGNORECASE):
        if re.search(r'\bapproved\b', query, re.IGNORECASE):
            filters['Status'] = 'Pending Com Approved'
        else:
            filters['Status'] = 'Pending Commitments'
    
    # Revolver patterns
    if re.search(r'\brevolver\b', query, re.IGNORECASE):
        filters['Revolver Flag'] = 'R'
    
    return filters

def calculate_metrics(df, metrics):
    """Calculate requested metrics from the dataframe"""
    results = {}
    
    # Convert exposure columns to numeric
    df['Used Exposure'] = pd.to_numeric(df['Used Exposure'], errors='coerce').fillna(0)
    df['Unused Exposure'] = pd.to_numeric(df['Unused Exposure'], errors='coerce').fillna(0)
    
    # Calculate total exposure
    df['Total Exposure'] = df['Used Exposure'] + df['Unused Exposure']
    
    # Calculate line utilization
    df['Line Utilization'] = np.where(df['Total Exposure'] > 0, 
                                     (df['Used Exposure'] / df['Total Exposure']) * 100, 0)
    
    for metric in metrics:
        if metric == 'total_exposure':
            results['Total Exposure ($B)'] = df['Total Exposure'].sum() / 1_000_000_000
        elif metric == 'exposure':
            results['Total Exposure ($B)'] = df['Total Exposure'].sum() / 1_000_000_000
        elif metric == 'used_exposure':
            results['Used Exposure ($B)'] = df['Used Exposure'].sum() / 1_000_000_000
        elif metric == 'unused_exposure':
            results['Unused Exposure ($B)'] = df['Unused Exposure'].sum() / 1_000_000_000
        elif metric == 'line_utilization':
            results['Line Utilization (%)'] = (df['Used Exposure'].sum() / df['Total Exposure'].sum() * 100) if df['Total Exposure'].sum() > 0 else 0
        elif metric == 'loans':
            results['Total Loans'] = len(df[df['Loans'] != '$0'])
        elif metric == 'facility_count':
            results['Facility Count'] = len(df)
    
    return results

def apply_filters(df, filters):
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if column in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[column] == value]
    
    return filtered_df

def create_dashboard_visualization(df, metrics):
    """Create dashboard-style visualization similar to the provided screenshot"""
    # Calculate metrics
    df['Used Exposure'] = pd.to_numeric(df['Used Exposure'], errors='coerce').fillna(0)
    df['Unused Exposure'] = pd.to_numeric(df['Unused Exposure'], errors='coerce').fillna(0)
    df['Total Exposure'] = df['Used Exposure'] + df['Unused Exposure']
    
    # Create metrics cards
    total_exposure = df['Total Exposure'].sum() / 1_000_000_000
    used_exposure = df['Used Exposure'].sum() / 1_000_000_000
    line_utilization = (df['Used Exposure'].sum() / df['Total Exposure'].sum() * 100) if df['Total Exposure'].sum() > 0 else 0
    facility_count = len(df)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=('Exposure ($B)', 'Used Exposure ($B)', 'Line Utilization (%)', 'Facility Count',
                       'Exposure by Industry', 'Utilization by Grade', 'Status Distribution', 'Monthly Trend'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "bar"}, {"type": "bar"}, {"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Add indicator charts (top row)
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=total_exposure,
        title={"text": "Total Exposure ($B)"},
        delta={'reference': total_exposure * 0.95, 'relative': True},
        number={'suffix': "B", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0.7, 1]}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=used_exposure,
        title={"text": "Used Exposure ($B)"},
        delta={'reference': used_exposure * 0.98, 'relative': True},
        number={'suffix': "B", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0.7, 1]}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=line_utilization,
        title={"text": "Line Utilization (%)"},
        delta={'reference': line_utilization * 0.9, 'relative': True},
        number={'suffix': "%", 'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0.7, 1]}
    ), row=1, col=3)
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=facility_count,
        title={"text": "Facility Count"},
        number={'font': {'size': 40}},
        domain={'x': [0, 1], 'y': [0.7, 1]}
    ), row=1, col=4)
    
    # Add charts (bottom row)
    # Exposure by Industry
    industry_exposure = df.groupby('Industry Division')['Total Exposure'].sum() / 1_000_000_000
    fig.add_trace(go.Bar(
        x=industry_exposure.index,
        y=industry_exposure.values,
        name="Exposure by Industry",
        marker_color='lightblue'
    ), row=2, col=1)
    
    # Utilization by Grade
    grade_util = df.groupby('Default Grade').apply(
        lambda x: (x['Used Exposure'].sum() / x['Total Exposure'].sum() * 100) if x['Total Exposure'].sum() > 0 else 0
    )
    fig.add_trace(go.Bar(
        x=grade_util.index,
        y=grade_util.values,
        name="Utilization by Grade",
        marker_color='lightgreen'
    ), row=2, col=2)
    
    # Status Distribution
    status_counts = df['Status'].value_counts()
    fig.add_trace(go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        name="Status Distribution"
    ), row=2, col=3)
    
    # Monthly Trend (simulated)
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    trend_values = [total_exposure * (0.8 + 0.05 * i) for i in range(6)]
    fig.add_trace(go.Scatter(
        x=months,
        y=trend_values,
        mode='lines+markers',
        name="Exposure Trend",
        line=dict(color='orange', width=3)
    ), row=2, col=4)
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Financial Dashboard",
        title_x=0.5
    )
    
    return fig

def process_natural_language_query(query, df):
    """Main function to process natural language query"""
    # Preprocess query
    processed_query = preprocess_query(query)
    
    # Extract metrics and filters
    metrics = extract_metrics_from_query(processed_query)
    filters = extract_filters_from_query(processed_query)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Calculate metrics
    results = calculate_metrics(filtered_df, metrics)
    
    return results, filtered_df, metrics, filters

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="Financial Data Query Assistant", layout="wide")
    
    st.title("🏦 Financial Data Query Assistant")
    st.markdown("Ask questions about your financial exposure and line utilization data in natural language!")
    
    # Load sample data
    if 'df' not in st.session_state:
        st.session_state.df = generate_sample_data()
    
    df = st.session_state.df
    
    # Sidebar for data overview
    with st.sidebar:
        st.header("📊 Data Overview")
        st.metric("Total Records", len(df))
        st.metric("Total Exposure ($B)", f"{df['Used Exposure'].sum() / 1_000_000_000:.2f}")
        st.metric("Industries", df['Industry Division'].nunique())
        
        st.subheader("Sample Queries")
        sample_queries = [
            "What is the total exposure and line utilization?",
            "Show me exposure for banks and finance industry",
            "What's the utilization rate for grade 6+ facilities?",
            "Total exposure for pending approved facilities",
            "Count of revolver facilities",
            "Healthcare industry exposure breakdown"
        ]
        
        for query in sample_queries:
            if st.button(f"📝 {query}", key=f"sample_{hash(query)}"):
                st.session_state.query_input = query
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    # Query input
    query_input = st.text_input(
        "Enter your question:",
        value=st.session_state.get('query_input', ''),
        placeholder="e.g., What is the total exposure and line utilization for banks?",
        key="main_query"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        query_button = st.button("🔍 Query", type="primary")
    
    if query_button and query_input:
        with st.spinner("Processing your query..."):
            try:
                results, filtered_df, metrics, filters = process_natural_language_query(query_input, df)
                
                # Display query interpretation
                st.header("🧠 Query Interpretation")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Detected Metrics:")
                    for metric in metrics:
                        st.write(f"• {metric.replace('_', ' ').title()}")
                
                with col2:
                    st.subheader("Applied Filters:")
                    if filters:
                        for key, value in filters.items():
                            st.write(f"• {key}: {value}")
                    else:
                        st.write("• No filters applied")
                
                # Display results
                st.header("📈 Results")
                
                # Create metrics cards
                cols = st.columns(len(results))
                for i, (metric, value) in enumerate(results.items()):
                    with cols[i]:
                        if isinstance(value, float):
                            if 'Utilization' in metric or '%' in metric:
                                st.metric(metric, f"{value:.2f}%")
                            elif '$' in metric or 'Exposure' in metric:
                                st.metric(metric, f"${value:.2f}B")
                            else:
                                st.metric(metric, f"{value:.2f}")
                        else:
                            st.metric(metric, f"{value:,}")
                
                # Create and display dashboard
                st.header("📊 Dashboard")
                dashboard_fig = create_dashboard_visualization(filtered_df, metrics)
                st.plotly_chart(dashboard_fig, use_container_width=True)
                
                # Display filtered data summary
                st.header("📋 Data Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Filtered Records", len(filtered_df))
                with col2:
                    st.metric("% of Total Data", f"{len(filtered_df)/len(df)*100:.1f}%")
                with col3:
                    avg_utilization = (filtered_df['Used Exposure'].astype(float).sum() / 
                                     filtered_df[['Used Exposure', 'Unused Exposure']].astype(float).sum().sum() * 100) if filtered_df[['Used Exposure', 'Unused Exposure']].astype(float).sum().sum() > 0 else 0
                    st.metric("Avg Utilization", f"{avg_utilization:.1f}%")
                
                # Show sample of filtered data
                if st.checkbox("Show detailed data"):
                    st.dataframe(
                        filtered_df.head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                
            except Exception as e:
                st.error(f"An error occurred while processing your query: {str(e)}")
                st.info("Please try rephrasing your question or use one of the sample queries.")
    
    # Help section
    with st.expander("ℹ️ Help & Examples"):
        st.markdown("""
        ### How to ask questions:
        
        **Metrics you can ask about:**
        - Total exposure, used exposure, unused exposure
        - Line utilization or utilization rate
        - Number of facilities or facility count
        - Loan amounts
        
        **Filters you can apply:**
        - Industry: banks, finance, retail, healthcare, real estate
        - Grade: 4, 5, 6, 7 (with or without +)
        - Status: pending, approved
        - Facility type: revolver
        
        **Example questions:**
        - "What is the total exposure for banks and finance?"
        - "Show me line utilization for grade 6+ facilities"
        - "How many revolver facilities do we have?"
        - "Total exposure for pending approved status"
        """)

if __name__ == "__main__":
    main()
