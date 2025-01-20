import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
from typing import Dict, List
import os
import pytz

def load_data() -> pd.DataFrame:
    """Load the most recent data file from the directory."""
    csv_files = [f for f in os.listdir('reddit_scraper') if f.startswith('reddit_business_ideas_') and f.endswith('.csv')]
    if not csv_files:
        return pd.DataFrame()
    
    latest_file = max(csv_files)
    df = pd.read_csv(os.path.join('reddit_scraper', latest_file))
    
    # Convert string timestamps to datetime objects
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'])
    
    return df

def analyze_sentiment_distribution(df: pd.DataFrame) -> Dict[str, List]:
    """Analyze the distribution of positive and negative indicators across ideas."""
    positive_words = []
    negative_words = []
    
    for pos in df['positive_indicators'].dropna():
        positive_words.extend([word.strip() for word in pos.split(',')])
    for neg in df['negative_indicators'].dropna():
        negative_words.extend([word.strip() for word in neg.split(',')])
    
    return {
        'positive': positive_words,
        'negative': negative_words
    }

def main():
    st.set_page_config(page_title="Business Idea Evaluator", layout="wide")
    
    st.title("Business Idea Evaluator 💡")
    st.markdown("""
    This application helps evaluate business ideas extracted from Reddit discussions.
    Use the filters below to explore and analyze different ideas.
    """)
    # st.write("Current working directory:", os.getcwd())
    # st.write("All files in cwd:", os.listdir("."))
    # st.write("Files in reddit_scraper folder:", os.listdir("reddit_scraper"))
    # Load data
    df = load_data()
    if df.empty:
        st.error("No data files found. Please run the scraper first.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        options=df['subreddit'].unique(),
        default=df['subreddit'].unique()
    )
    
    # Add date filter in sidebar
    if 'created_utc' in df.columns:
        min_date = df['created_utc'].min()
        max_date = df['created_utc'].max()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                (df['created_utc'].dt.date >= start_date) & 
                (df['created_utc'].dt.date <= end_date)
            ]
    
    # Filter data
    filtered_df = df[df['subreddit'].isin(selected_subreddits)]
    
    # Business Ideas Overview (Full width)
    st.subheader("Business Ideas Overview")
    ideas_view = filtered_df[['business_idea', 'positive_indicators', 'negative_indicators', 'url']]
    ideas_view = ideas_view[ideas_view['business_idea'].notna() & (ideas_view['business_idea'] != '')]
    st.dataframe(ideas_view, use_container_width=True)
    
    # Sentiment Analysis (Two columns underneath)
    st.subheader("Sentiment Analysis")
    sentiments = analyze_sentiment_distribution(filtered_df)
    
    # Create two columns for positive and negative indicators
    col1, col2 = st.columns(2)
    
    with col1:
        pos_counts = pd.Series(sentiments['positive']).value_counts().head(10)
        fig1 = px.bar(
            pos_counts, 
            title="Top Positive Indicators",
            labels={'value': 'Count', 'index': 'Indicator'},
            color_discrete_sequence=['#2ecc71']  # Green color for positive
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        neg_counts = pd.Series(sentiments['negative']).value_counts().head(10)
        fig2 = px.bar(
            neg_counts, 
            title="Top Negative Indicators",
            labels={'value': 'Count', 'index': 'Indicator'},
            color_discrete_sequence=['#e74c3c']  # Red color for negative
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed idea viewer
    st.subheader("Idea Details")
    if not ideas_view.empty:
        selected_idea = st.selectbox(
            "Select an idea to view details:",
            options=ideas_view['business_idea'].tolist()
        )
        
        if selected_idea:
            idea_details = filtered_df[filtered_df['business_idea'] == selected_idea].iloc[0]
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("### Context")
                st.write(idea_details['idea_context'])
                
                st.markdown("### Explicit Pros")
                st.write(idea_details['explicit_pros'] if idea_details['explicit_pros'] else "No explicit pros mentioned")
            
            with col4:
                st.markdown("### Sentiment Indicators")
                st.write("Positive:", idea_details['positive_indicators'])
                st.write("Negative:", idea_details['negative_indicators'])
                
                st.markdown("### Explicit Cons")
                st.write(idea_details['explicit_cons'] if idea_details['explicit_cons'] else "No explicit cons mentioned")
            
            st.markdown(f"[View Original Post]({idea_details['url']})")

if __name__ == "__main__":
    main() 