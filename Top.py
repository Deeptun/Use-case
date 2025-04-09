import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import re
import random
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    return OpenAI(api_key=api_key)

MODEL_ID = "gpt-4o"

# --------------------- Helper Functions ---------------------
def call_gpt4o(prompt: str, system_prompt: str = "", temperature: float = 0.0, max_tokens: int = 2048) -> str:
    """
    Call the GPT-4o model with a given prompt.
    
    Args:
        prompt: The user prompt to send to GPT-4o
        system_prompt: Optional system prompt to set context
        temperature: Controls randomness (0.0 = deterministic)
        max_tokens: Maximum tokens in the response
        
    Returns:
        The response text from GPT-4o
    """
    client = get_openai_client()
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    try:
        with st.spinner("Calling GPT-4o API..."):
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling GPT-4o API: {e}")
        return ""

def extract_json_from_response(response_text: str) -> Dict:
    """
    Extract JSON from GPT-4o response.
    
    Args:
        response_text: The response text from GPT-4o
        
    Returns:
        Parsed JSON as a dictionary
    """
    try:
        # First, try to parse the entire text as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON using regex
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON-like structure
        json_match = re.search(r'{.*}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        st.warning(f"Could not extract JSON from response. Using empty dict.")
        return {}

# --------------------- Data Generation Functions ---------------------
def create_dummy_data(num_records: int = 200, start_date: str = "2024-01-01", end_date: str = "2024-04-01") -> pd.DataFrame:
    """
    Create dummy data to simulate credit officer comments.
    
    Args:
        num_records: Number of records to generate
        start_date: Start date for the records
        end_date: End date for the records
        
    Returns:
        DataFrame with dummy data
    """
    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate random dates within the range
    dates = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_records)]
    dates.sort()  # Sort dates chronologically
    
    # Industry types
    industry_types = [
        "Manufacturing", "Retail", "Healthcare", "Technology", "Finance", 
        "Real Estate", "Construction", "Agriculture", "Energy", "Transportation",
        "Education", "Hospitality", "Entertainment", "Telecommunications", "Utilities"
    ]
    
    # Case ID format: TYPE-YEAR-SEQUENCE
    case_ids = [f"CR-{date.year}-{i+1:04d}" for i, date in enumerate(dates)]
    
    # Credit officer comments templates
    positive_templates = [
        "Client shows strong financial performance with {ratio} debt-to-income ratio. {industry} sector outlook is positive. Recommend approval.",
        "Excellent credit history and stable cash flow. The {industry} business has been operating for {years} years with consistent growth.",
        "Financial statements indicate healthy profit margins of {margin}%. {industry} market indicators are favorable.",
        "Low risk profile with adequate collateral. Business in {industry} sector has diversified revenue streams and strong management.",
        "Solid balance sheet with {ratio} current ratio. {industry} business demonstrates resilience during economic fluctuations."
    ]
    
    neutral_templates = [
        "Client meets minimum requirements for lending. {industry} sector shows moderate stability. Additional monitoring recommended.",
        "Acceptable financial performance but with some fluctuations in quarterly results. {industry} has seen mixed trends.",
        "Credit history shows minor delinquencies but overall remains satisfactory. {industry} business faces typical market challenges.",
        "Debt service coverage ratio at {ratio}, which is adequate but not exceptional for {industry} sector.",
        "Business plan is reasonable but contains some assumptions that may need validation. {industry} market has moderate competition."
    ]
    
    negative_templates = [
        "Client shows concerning debt-to-income ratio of {ratio}. {industry} sector outlook is uncertain with recent downtrends.",
        "Financial statements reveal declining profit margins, down to {margin}% from 15% last year. {industry} facing headwinds.",
        "Multiple late payments in credit history. Business in {industry} sector shows vulnerability to market fluctuations.",
        "Insufficient collateral for the requested amount. {industry} business has concentrated customer base, creating dependency risk.",
        "Cash flow projections appear optimistic given current {industry} market conditions. High fixed costs relative to revenue."
    ]
    
    # Generate comments
    comments = []
    for i in range(num_records):
        industry = random.choice(industry_types)
        template_group = random.choices([positive_templates, neutral_templates, negative_templates], weights=[0.4, 0.3, 0.3])[0]
        template = random.choice(template_group)
        
        ratio = round(random.uniform(0.5, 3.5), 2)
        margin = round(random.uniform(2, 25), 1)
        years = random.randint(1, 20)
        
        comment = template.format(industry=industry, ratio=ratio, margin=margin, years=years)
        comments.append(comment)
    
    # Create DataFrame
    df = pd.DataFrame({
        'case_id': case_ids,
        'date': dates,
        'industry_type': [random.choice(industry_types) for _ in range(num_records)],
        'comment': comments
    })
    
    return df

# --------------------- Topic Modeling Functions ---------------------
def extract_topics_from_comments(df: pd.DataFrame, num_topics: int = 5, with_subtopics: bool = True) -> Dict[str, Any]:
    """
    Extract topics from the corpus of comments.
    
    Args:
        df: DataFrame containing comments
        num_topics: Number of topics to extract
        with_subtopics: Whether to include subtopics
        
    Returns:
        Dictionary containing topics and their metadata
    """
    # Sample comments if there are too many
    sample_size = min(100, len(df))
    sampled_comments = df['comment'].sample(sample_size).tolist()
    
    # Prepare sample text
    sample_corpus = "\n\n---\n\n".join(sampled_comments)
    
    subtopics_instruction = ""
    if with_subtopics:
        subtopics_instruction = "For each main topic, identify 2-3 subtopics that represent more specific themes within that topic."
    
    # Create prompt for GPT-4o to extract topics
    prompt = f"""
    I have a corpus of credit officer comments, and I need to extract {num_topics} main topics.
    Below are {sample_size} representative comments from credit officers:
    
    {sample_corpus}
    
    Please identify exactly {num_topics} distinct topics that best represent the themes in these comments.
    {subtopics_instruction}
    
    For each topic and subtopic, provide:
    1. A concise label (1-5 words)
    2. A brief description (1-2 sentences)
    3. 3-5 keywords that are strongly associated with this topic
    
    Return your answer as JSON with the following format:
    {{
        "topics": [
            {{
                "id": 1,
                "label": "Topic label",
                "description": "Topic description",
                "keywords": ["keyword1", "keyword2", "keyword3"],
                "subtopics": [
                    {{
                        "id": "1.1",
                        "label": "Subtopic label",
                        "description": "Subtopic description",
                        "keywords": ["keyword1", "keyword2", "keyword3"]
                    }}
                ]
            }}
        ]
    }}
    """
    
    system_prompt = "You are an expert in financial text analysis. Your task is to extract coherent topics from credit officer comments."
    
    with st.spinner("Extracting topics from comments..."):
        response = call_gpt4o(prompt, system_prompt, temperature=0.1, max_tokens=4000)
    
    try:
        result = extract_json_from_response(response)
        if "topics" in result:
            result["timestamp"] = datetime.now().isoformat()
            return result
        else:
            st.error("Error: 'topics' key not found in the extracted JSON")
            return {"topics": [], "timestamp": datetime.now().isoformat()}
    except Exception as e:
        st.error(f"Error parsing GPT-4o response: {e}")
        return {"topics": [], "timestamp": datetime.now().isoformat()}

def assign_topics_to_comments(df: pd.DataFrame, topics: Dict[str, Any], progress_bar=None) -> pd.DataFrame:
    """
    Assign topics and subtopics to individual comments.
    
    Args:
        df: DataFrame with comments
        topics: Topics dictionary from extract_topics_from_comments
        progress_bar: Optional Streamlit progress bar
        
    Returns:
        DataFrame with added topic and subtopic columns
    """
    result_df = df.copy()
    
    # Prepare the topics for the prompt
    topics_list = []
    for topic in topics["topics"]:
        topic_info = f"Topic {topic['id']}: {topic['label']} - {topic['description']} - Keywords: {', '.join(topic['keywords'])}"
        topics_list.append(topic_info)
        
        if "subtopics" in topic and topic["subtopics"]:
            for subtopic in topic["subtopics"]:
                subtopic_info = f"  Subtopic {subtopic['id']}: {subtopic['label']} - {subtopic['description']} - Keywords: {', '.join(subtopic['keywords'])}"
                topics_list.append(subtopic_info)
    
    topics_text = "\n".join(topics_list)
    
    # Process in batches to avoid API limits
    batch_size = 10
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Initialize columns
    result_df['assigned_topic_id'] = None
    result_df['assigned_topic_label'] = None
    result_df['assigned_subtopic_id'] = None
    result_df['assigned_subtopic_label'] = None
    result_df['topic_confidence'] = None
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        
        if progress_bar:
            progress_bar.progress((batch_idx + 1) / total_batches, 
                                  text=f"Processing batch {batch_idx+1}/{total_batches} (comments {start_idx+1}-{end_idx})")
        
        batch_comments = df['comment'].iloc[start_idx:end_idx].tolist()
        comments_text = "\n\n---\n\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(batch_comments)])
        
        # Create prompt for GPT-4o to assign topics
        prompt = f"""
        I have a set of predefined topics and a batch of credit officer comments. I need to assign the most relevant topic and subtopic to each comment.
        
        TOPICS:
        {topics_text}
        
        COMMENTS:
        {comments_text}
        
        For each comment, determine which topic and subtopic (if applicable) it most closely relates to.
        
        Return your analysis as a JSON with the following format:
        {{
            "assignments": [
                {{
                    "comment_index": 1,
                    "assigned_topic_id": <topic_id>,
                    "assigned_topic_label": "Topic label",
                    "assigned_subtopic_id": <subtopic_id or null>,
                    "assigned_subtopic_label": "Subtopic label or null",
                    "confidence": <0.0-1.0>
                }},
                ...
            ]
        }}
        """
        
        system_prompt = "You are an expert financial analyst who categorizes credit officer comments according to predefined topics."
        
        response = call_gpt4o(prompt, system_prompt)
        
        try:
            result = extract_json_from_response(response)
            assignments = result.get("assignments", [])
            
            for assignment in assignments:
                comment_idx = assignment.get("comment_index", 0) - 1  # Convert to 0-indexed
                if 0 <= comment_idx < len(batch_comments):
                    df_idx = start_idx + comment_idx
                    result_df.loc[df_idx, 'assigned_topic_id'] = assignment.get("assigned_topic_id")
                    result_df.loc[df_idx, 'assigned_topic_label'] = assignment.get("assigned_topic_label")
                    result_df.loc[df_idx, 'assigned_subtopic_id'] = assignment.get("assigned_subtopic_id")
                    result_df.loc[df_idx, 'assigned_subtopic_label'] = assignment.get("assigned_subtopic_label")
                    result_df.loc[df_idx, 'topic_confidence'] = assignment.get("confidence", 0.0)
        except Exception as e:
            st.error(f"Error processing batch {batch_idx+1}: {e}")
    
    # Fill NA values with appropriate defaults
    result_df['assigned_topic_id'] = result_df['assigned_topic_id'].fillna(0)
    result_df['assigned_topic_label'] = result_df['assigned_topic_label'].fillna("Unclassified")
    result_df['assigned_subtopic_id'] = result_df['assigned_subtopic_id'].fillna("None")
    result_df['assigned_subtopic_label'] = result_df['assigned_subtopic_label'].fillna("None")
    result_df['topic_confidence'] = result_df['topic_confidence'].fillna(0.0)
    
    return result_df

def calculate_topic_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Calculate topic drift between a reference period and the current period.
    
    Args:
        reference_df: DataFrame with topic assignments from the reference period
        current_df: DataFrame with topic assignments from the current period
        
    Returns:
        Tuple of (overall_drift_score, DataFrame with drift scores)
    """
    if len(reference_df) == 0 or len(current_df) == 0:
        st.error("Error: Empty DataFrame provided for drift calculation")
        return 0.0, current_df.copy()
    
    # Add 'period' column for identification
    ref_with_period = reference_df.copy()
    curr_with_period = current_df.copy()
    ref_with_period['period'] = 'reference'
    curr_with_period['period'] = 'current'
    
    # Combine both DataFrames for analysis
    combined_df = pd.concat([ref_with_period, curr_with_period], axis=0)
    
    # Calculate topic distribution for reference period
    ref_topic_counts = reference_df['assigned_topic_id'].value_counts(normalize=True)
    ref_topic_dist = {str(topic_id): count for topic_id, count in ref_topic_counts.items()}
    
    # Calculate topic distribution for current period
    curr_topic_counts = current_df['assigned_topic_id'].value_counts(normalize=True)
    curr_topic_dist = {str(topic_id): count for topic_id, count in curr_topic_counts.items()}
    
    # Prepare topic distributions for GPT-4o
    ref_dist_text = "\n".join([f"Topic {topic_id}: {count*100:.2f}%" for topic_id, count in ref_topic_dist.items()])
    curr_dist_text = "\n".join([f"Topic {topic_id}: {count*100:.2f}%" for topic_id, count in curr_topic_dist.items()])
    
    # Also include the label information
    ref_topics = reference_df[['assigned_topic_id', 'assigned_topic_label']].drop_duplicates()
    topic_labels = {str(row['assigned_topic_id']): row['assigned_topic_label'] for _, row in ref_topics.iterrows()}
    
    labels_text = "\n".join([f"Topic {topic_id}: {label}" for topic_id, label in topic_labels.items()])
    
    # Create prompt for GPT-4o to calculate drift
    prompt = f"""
    I need to calculate the drift between two distributions of topics in credit officer comments.
    
    TOPIC LABELS:
    {labels_text}
    
    REFERENCE PERIOD DISTRIBUTION:
    {ref_dist_text}
    
    CURRENT PERIOD DISTRIBUTION:
    {curr_dist_text}
    
    Please analyze the drift between these two distributions and provide:
    1. An overall drift score from 0 to 100 (where 0 means identical distributions and 100 means completely different)
    2. For each topic, a specific drift score and assessment
    
    Return the analysis as JSON:
    {{
        "overall_drift_score": <0-100>,
        "topic_drift": [
            {{
                "topic_id": "1",
                "topic_label": "Topic label",
                "reference_percentage": <0-100>,
                "current_percentage": <0-100>,
                "percentage_change": <float>,
                "drift_assessment": "Brief explanation"
            }},
            ...
        ]
    }}
    """
    
    system_prompt = "You are an expert at analyzing shifts in topic distributions over time."
    
    with st.spinner("Calculating topic drift..."):
        response = call_gpt4o(prompt, system_prompt)
    
    try:
        drift_result = extract_json_from_response(response)
        overall_drift = drift_result.get("overall_drift_score", 50.0)
        topic_drift = drift_result.get("topic_drift", [])
        
        # Create a DataFrame for detailed drift information
        drift_df = pd.DataFrame(topic_drift)
        
        # Add drift information to the current DataFrame
        result_df = curr_with_period.copy()
        result_df['overall_drift_score'] = overall_drift
        
        # Add topic-specific drift scores
        if not drift_df.empty and 'topic_id' in drift_df.columns:
            drift_map = {str(row['topic_id']): row['percentage_change'] for _, row in drift_df.iterrows()}
            result_df['topic_drift'] = result_df['assigned_topic_id'].astype(str).map(drift_map).fillna(0)
        else:
            result_df['topic_drift'] = 0
        
        return overall_drift, result_df
    except Exception as e:
        st.error(f"Error calculating drift: {e}")
        return 0.0, curr_with_period.copy()

# --------------------- Visualization Functions ---------------------
def create_topic_distribution_chart(df: pd.DataFrame):
    """
    Create a Plotly chart for topic distribution.
    
    Args:
        df: DataFrame with topic assignments
    """
    topic_counts = df['assigned_topic_label'].value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    
    fig = px.bar(topic_counts, x='Count', y='Topic', orientation='h',
                title='Distribution of Topics',
                color='Count',
                color_continuous_scale='Viridis')
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_showscale=False,
        xaxis_title='Number of Comments',
        yaxis_title=None
    )
    
    return fig

def create_confidence_by_topic_chart(df: pd.DataFrame):
    """
    Create a Plotly chart for topic confidence by topic.
    
    Args:
        df: DataFrame with topic assignments and confidence scores
    """
    fig = px.box(df, x='assigned_topic_label', y='topic_confidence',
                title='Topic Confidence by Topic',
                color='assigned_topic_label')
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis_title='Topic',
        yaxis_title='Confidence Score',
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def create_topic_time_chart(df: pd.DataFrame):
    """
    Create a Plotly chart for topic distribution over time.
    
    Args:
        df: DataFrame with topic assignments and dates
    """
    if 'date' in df.columns:
        # Add month column
        df = df.copy()
        df['month'] = df['date'].dt.to_period('M').astype(str)
        
        # Pivot to get topic distribution by month
        pivot_df = pd.crosstab(df['month'], df['assigned_topic_label'], normalize='index').reset_index()
        
        # Convert to long format for Plotly
        long_df = pd.melt(pivot_df, id_vars=['month'], var_name='Topic', value_name='Proportion')
        
        fig = px.area(long_df, x='month', y='Proportion', color='Topic',
                    title='Topic Distribution Over Time',
                    color_discrete_sequence=px.colors.qualitative.Plotly)
        
        fig.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title='Month',
            yaxis_title='Proportion',
            legend_title='Topic'
        )
        
        return fig
    
    return None

def create_drift_visualization(drift_df: pd.DataFrame):
    """
    Create visualization of topic drift.
    
    Args:
        drift_df: DataFrame with topic drift information
    """
    if drift_df.empty or 'topic_id' not in drift_df.columns:
        return None
    
    # Sort by absolute percentage change
    drift_df = drift_df.sort_values(by='percentage_change', key=abs, ascending=False)
    
    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1)
    
    # Add bar chart of percentage changes
    fig.add_trace(
        go.Bar(
            x=drift_df['topic_label'],
            y=drift_df['percentage_change'],
            marker_color=['red' if x < 0 else 'green' for x in drift_df['percentage_change']],
            text=drift_df['percentage_change'].round(2).astype(str) + '%',
            textposition='auto',
            name='Percentage Change'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Topic Drift: Percentage Change by Topic',
        xaxis_title='Topic',
        yaxis_title='Percentage Change (%)',
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# --------------------- Main Processing Functions ---------------------
def process_dataframe(df: pd.DataFrame, reference_file: str = None, save_reference: bool = False) -> pd.DataFrame:
    """
    Process a DataFrame to add topic classifications and drift information.
    
    Args:
        df: Input DataFrame with comments
        reference_file: Optional path to a reference topics file
        save_reference: Whether to save the extracted topics as a reference
        
    Returns:
        DataFrame with added topic and drift columns
    """
    # Make sure date column is datetime
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    st.info(f"Processing {len(df)} comments...")
    
    # Load or extract topics
    topics = None
    reference_df = None
    
    if reference_file and os.path.exists(reference_file):
        try:
            with open(reference_file, 'rb') as f:
                reference_data = pickle.load(f)
                topics = reference_data.get('topics')
                reference_df = reference_data.get('dataframe')
                st.success(f"Loaded reference topics from {reference_file}")
        except Exception as e:
            st.error(f"Error loading reference file: {e}")
    
    if topics is None:
        # Extract topics from the comments
        topics = extract_topics_from_comments(df, num_topics=5, with_subtopics=True)
        
        if save_reference:
            # Save as reference
            reference_df = df.copy()
            reference_data = {
                'topics': topics,
                'dataframe': reference_df,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(reference_file or "references/"), exist_ok=True)
            
            with open(reference_file or "references/topic_reference.pkl", 'wb') as f:
                pickle.dump(reference_data, f)
            st.success(f"Saved reference topics to {reference_file or 'references/topic_reference.pkl'}")
    
    # Display topics
    st.subheader("Identified Topics")
    for topic in topics.get("topics", []):
        with st.expander(f"Topic {topic['id']}: {topic['label']}"):
            st.write(f"**Description:** {topic['description']}")
            st.write(f"**Keywords:** {', '.join(topic['keywords'])}")
            
            if "subtopics" in topic and topic["subtopics"]:
                st.write("**Subtopics:**")
                for subtopic in topic["subtopics"]:
                    st.write(f"- **{subtopic['id']}: {subtopic['label']}** - {subtopic['description']}")
                    st.write(f"  Keywords: {', '.join(subtopic['keywords'])}")
    
    # Assign topics to comments
    progress_bar = st.progress(0, text="Processing comments...")
    result_df = assign_topics_to_comments(df, topics, progress_bar)
    progress_bar.progress(1.0, text="Processing complete!")
    
    # Calculate drift if we have a reference
    drift_df = pd.DataFrame()
    if reference_df is not None:
        overall_drift, result_df = calculate_topic_drift(reference_df, result_df)
        st.success(f"Overall drift score: {overall_drift:.2f}")
        
        # Extract drift details for visualization
        if 'topic_drift' in result_df.columns:
            # Try to extract detailed drift information from the last response
            drift_df = pd.DataFrame(st.session_state.get('last_drift_response', {}).get('topic_drift', []))
    else:
        result_df['overall_drift_score'] = 0
        result_df['topic_drift'] = 0
    
    return result_df, drift_df

# --------------------- Streamlit App ---------------------
def setup_app():
    """
    Set up the Streamlit app.
    """
    st.set_page_config(page_title="Credit Officer Topic Modeling", layout="wide")
    st.title("Credit Officer Comments Topic Modeling")
    
    # Initialize session state
    if 'reference_file' not in st.session_state:
        st.session_state.reference_file = "references/topic_reference.pkl"
    
    if 'last_drift_response' not in st.session_state:
        st.session_state.last_drift_response = {}
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data options
    data_option = st.sidebar.radio("Data Source", ["Use Example Data", "Upload CSV", "Manual Input"])
    
    # Reference model options
    use_reference = st.sidebar.checkbox("Use Reference Model", value=True)
    save_as_reference = st.sidebar.checkbox("Save as Reference", value=False)
    
    if use_reference:
        reference_file = st.session_state.reference_file
    else:
        reference_file = None
    
    # Main content
    if data_option == "Use Example Data":
        with st.sidebar.expander("Example Data Options"):
            num_records = st.slider("Number of Records", 50, 500, 200)
            start_date = st.date_input("Start Date", datetime(2024, 1, 1))
            end_date = st.date_input("End Date", datetime(2024, 4, 1))
            
            if start_date >= end_date:
                st.error("End date must be after start date")
                return
            
            cutoff_date = st.date_input("Reference Cutoff Date", 
                                       datetime.fromordinal((start_date.toordinal() + end_date.toordinal()) // 2))
            
            if cutoff_date <= start_date or cutoff_date >= end_date:
                st.warning("Cutoff date should be between start and end dates")
        
        if st.sidebar.button("Generate Example Data"):
            with st.spinner("Generating example data..."):
                df = create_dummy_data(
                    num_records=num_records, 
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                # Split data for reference and current if needed
                if use_reference and not os.path.exists(reference_file) and save_as_reference:
                    cutoff_datetime = datetime.combine(cutoff_date, datetime.min.time())
                    reference_df = df[df['date'] < cutoff_datetime].copy()
                    current_df = df[df['date'] >= cutoff_datetime].copy()
                    
                    st.session_state.reference_df = reference_df
                    st.session_state.current_df = current_df
                else:
                    st.session_state.current_df = df
            
            st.success("Example data generated successfully!")
    
    elif data_option == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check if required columns exist
                required_columns = ['comment']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {', '.join(missing_columns)}")
                else:
                    # Convert date column if it exists
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    st.session_state.current_df = df
                    st.success("CSV uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
    
    elif data_option == "Manual Input":
        st.sidebar.markdown("### Manual Data Entry")
        st.sidebar.markdown("Enter data in the format below, one record per line:")
        st.sidebar.markdown("```\ncase_id,date,industry_type,comment\n```")
        
        manual_data = st.sidebar.text_area("Enter CSV Data", height=200)
        
        if st.sidebar.button("Process Manual Input"):
            try:
                if manual_data:
                    # Convert string to DataFrame
                    from io import StringIO
                    data_io = StringIO(manual_data)
                    df = pd.read_csv(data_io)
                    
                    # Convert date column if it exists
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                    
                    # Check if comment column exists
                    if 'comment' not in df.columns:
                        st.error("Missing required 'comment' column")
                    else:
                        st.session_state.current_df = df
                        st.success("Manual data processed successfully!")
                else:
                    st.error("No data entered")
            except Exception as e:
                st.error(f"Error processing manual input: {e}")
    
    # Process data if available
    tabs = st.tabs(["Data View", "Topic Analysis", "Drift Analysis"])
    
    with tabs[0]:  # Data View
        if 'current_df' in st.session_state:
            df = st.session_state.current_df
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write(f"Total records: {len(df)}")
            
            if 'date' in df.columns:
                min_date = df['date'].min()
                max_date = df['date'].max()
                st.write(f"Date range: {min_date.date()} to {max_date.date()}")
            
            # Basic statistics
            if 'industry_type' in df.columns:
                st.subheader("Industry Breakdown")
                industry_counts = df['industry_type'].value_counts()
                fig = px.pie(values=industry_counts.values, names=industry_counts.index, title="Industry Distribution")
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Topic Analysis
        if 'current_df' in st.session_state:
            df = st.session_state.current_df
            
            if st.button("Run Topic Analysis"):
                result_df, drift_df = process_dataframe(
                    df, 
                    reference_file=reference_file if use_reference else None,
                    save_reference=save_as_reference
                )
                
                st.session_state.result_df = result_df
                st.session_state.drift_df = drift_df
            
            if 'result_df' in st.session_state:
                result_df = st.session_state.result_df
                
                # Display results
                st.subheader("Results Preview")
                display_cols = ['case_id', 'date', 'industry_type', 'comment', 
                               'assigned_topic_label', 'assigned_subtopic_label', 'topic_confidence']
                display_cols = [col for col in display_cols if col in result_df.columns]
                st.dataframe(result_df[display_cols].head(10), use_container_width=True)
                
                # Visualizations
                st.subheader("Topic Distribution")
                topic_dist_chart = create_topic_distribution_chart(result_df)
                st.plotly_chart(topic_dist_chart, use_container_width=True)
                
                st.subheader("Topic Confidence")
                confidence_chart = create_confidence_by_topic_chart(result_df)
                st.plotly_chart(confidence_chart, use_container_width=True)
                
                if 'date' in result_df.columns:
                    st.subheader("Topic Distribution Over Time")
                    time_chart = create_topic_time_chart(result_df)
                    if time_chart:
                        st.plotly_chart(time_chart, use_container_width=True)
                
                # Download option
                csv_data = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_data,
                    file_name="topic_analysis_results.csv",
                    mime="text/csv"
                )
    
    with tabs[2]:  # Drift Analysis
        if 'result_df' in st.session_state and 'overall_drift_score' in st.session_state.result_df.columns:
            result_df = st.session_state.result_df
            drift_df = st.session_state.get('drift_df', pd.DataFrame())
            
            # Display overall drift metrics
            overall_drift = result_df['overall_drift_score'].iloc[0] if len(result_df) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Overall Drift Score", f"{overall_drift:.2f}")
            
            # Get drift thresholds from sidebar
            with st.sidebar.expander("Drift Thresholds"):
                low_threshold = st.slider("Low Drift Threshold", 0, 100, 20)
                high_threshold = st.slider("High Drift Threshold", 0, 100, 50)
            
            # Display drift status
            with col2:
                if overall_drift < low_threshold:
                    st.success("Low Drift: Topics are stable")
                elif overall_drift < high_threshold:
                    st.warning("Moderate Drift: Some topics shifting")
                else:
                    st.error("High Drift: Significant topic changes")
            
            # Display drift visualization if available
            if not drift_df.empty and 'topic_label' in drift_df.columns:
                st.subheader("Topic-Level Drift")
                drift_chart = create_drift_visualization(drift_df)
                if drift_chart:
                    st.plotly_chart(drift_chart, use_container_width=True)
                    
                # Show drift details in a table
                st.dataframe(drift_df, use_container_width=True)
            else:
                st.info("Detailed drift information is not available. Run topic analysis with a reference model to generate drift metrics.")
                
                if not use_reference:
                    st.warning("Enable 'Use Reference Model' in the sidebar to calculate drift metrics.")

def main():
    """
    Main function for the Streamlit app.
    """
    setup_app()

if __name__ == "__main__":
    main()
