import streamlit as st

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="EasyJet DCF Model Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_excel_file
from dcf_analyzer import DCFAnalyzer
import os

# Custom CSS for enhanced appearance
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #1E88E5;
    }
    div.stButton > button:first-child {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 25px;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #1565C0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Header with logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://logos-download.com/wp-content/uploads/2016/03/EasyJet_logo.png", width=100)
with col2:
    st.title("EasyJet Financial DCF Analysis Dashboard")
    st.subheader("Interactive analysis of EasyJet's Discounted Cash Flow model")

# Create tabs for app navigation
main_tab1, main_tab2, main_tab3 = st.tabs([
    "📊 Interactive DCF Dashboard", 
    "📝 Documentation", 
    "⚙️ Settings"
])

with main_tab1:
    # Main analysis dashboard
    if os.path.exists("attached_assets/EasyJet- complete.xlsx"):
        try:
            # Load Excel file directly from attached assets
            file_path = "attached_assets/EasyJet- complete.xlsx"
            df_dict, file_name = load_excel_file(file_path)
            
            if df_dict is None:
                st.error("Error loading the Excel file. Please check the file structure.")
            elif 'DCF' not in df_dict:
                st.error("The Excel file does not contain a 'DCF' tab. Please check the file structure.")
                st.write("Available tabs:", ", ".join(df_dict.keys()))
            else:
                # Create DCF Analyzer instance
                dcf_analyzer = DCFAnalyzer(df_dict['DCF'])
                
                # Display all visualizations
                dcf_analyzer.display_all_visualizations()
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.info("Detailed error information for debugging:", icon="🔍")
            st.exception(e)
    else:
        # File uploader if no file is found
        st.info("Please upload the EasyJet financial model Excel file to begin analysis.")
        uploaded_file = st.file_uploader("Upload EasyJet financial model Excel file", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # Load Excel file
                df_dict, file_name = load_excel_file(uploaded_file)
                
                if 'DCF' not in df_dict:
                    st.error("The Excel file does not contain a 'DCF' tab. Please upload the correct file.")
                else:
                    # Create DCF Analyzer instance
                    dcf_analyzer = DCFAnalyzer(df_dict['DCF'])
                    
                    # Display all visualizations
                    dcf_analyzer.display_all_visualizations()
            
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
        else:
            # Display a placeholder message when no file is uploaded
            st.warning("No file detected. Please upload the EasyJet financial model to continue.")
            
            # Create a slick preview using streamlit cards
            st.markdown("### Dashboard Preview")
            
            # Sample dashboard preview
            preview_col1, preview_col2 = st.columns([3, 2])
            with preview_col1:
                st.image("https://cdn.dribbble.com/users/1577286/screenshots/16457262/media/4a7e2b6139e2ca8e1caee71696b5f71f.png?resize=1600x1200&vertical=center", 
                         caption="Sample Financial Dashboard Visualization", 
                         use_column_width=True)
            
            with preview_col2:
                st.markdown("""
                ### Key Features
                
                The EasyJet DCF Analysis Dashboard provides:
                
                - **Interactive 3D Visualizations** of enterprise value across different WACC and growth assumptions
                - **Advanced Sensitivity Analysis** with heatmaps showing two-factor interactions
                - **Monte Carlo Simulations** to understand probability distributions of potential outcomes
                - **Value Driver Analysis** identifying the most influential factors
                - **Scenario Comparison** across market conditions
                
                Upload your Excel file to unlock these advanced features.
                """)

with main_tab2:
    # Documentation page
    st.header("Documentation and Help")
    
    # Add documentation sections with expanders
    with st.expander("What is a DCF Analysis?", expanded=True):
        st.markdown("""
        Discounted Cash Flow (DCF) is a valuation method used to estimate the value of an investment based on its expected future cash flows.
        
        The DCF analysis attempts to determine the value of a company today, based on projections of how much money it will generate in the future.
        DCF analysis finds the present value of expected future cash flows using a discount rate.
        
        Key components of DCF analysis include:
        - **Projected Free Cash Flows**: Estimates of future cash flows
        - **Terminal Value**: The value of the business beyond the forecast period
        - **Discount Rate (WACC)**: The rate used to discount future cash flows to present value
        - **Present Value**: The sum of all discounted future cash flows
        """)
    
    with st.expander("How to Use This Dashboard"):
        st.markdown("""
        ### Using the Interactive Features
        
        1. **Enterprise Value Analysis**: Explore the 3D visualization to understand how changes in WACC and Terminal Growth rate affect enterprise value.
        
        2. **Share Price Analysis**: Use the waterfall chart and sunburst diagram to understand the components of the share price valuation.
        
        3. **Sensitivity Analysis**: Adjust parameters using the sliders to see real-time changes in valuation scenarios.
        
        4. **Advanced Visualizations**: Explore Monte Carlo simulations, scenario comparisons, and value driver analysis for deeper insights.
        
        ### Tips for Analysis
        
        - Use the tabbed interface to navigate between different analysis views
        - Adjust sensitivity parameters to test different assumptions
        - Hover over chart elements to see detailed data points
        - Use the toggleable scales and views for different perspectives
        """)
    
    with st.expander("Methodology & Calculations"):
        st.markdown("""
        ### DCF Methodology
        
        This dashboard implements a standard DCF methodology with the following components:
        
        1. **Free Cash Flow Forecasts**: Extracted from the model for the explicit forecast period
        
        2. **Terminal Value Calculation**: Uses both:
           - Perpetuity Growth Method: TV = FCF_terminal × (1 + g) / (WACC - g)
           - Exit Multiple Method: TV = EBITDA_terminal × Selected_Multiple
        
        3. **Enterprise Value Calculation**: Sum of discounted free cash flows and terminal value
        
        4. **Equity Value**: Enterprise Value minus Net Debt
        
        5. **Share Price**: Equity Value divided by number of shares outstanding
        
        ### Sensitivity Analysis
        
        The sensitivity analysis uses partial derivatives and interpolation to estimate the impact of changing key variables.
        The two-factor analysis uses a bivariate model to capture interaction effects between variables.
        
        ### Monte Carlo Simulation
        
        The Monte Carlo simulation uses random sampling from probability distributions for key inputs to generate a range of possible outcomes.
        This helps quantify uncertainty and provides probability distributions for share price outcomes.
        """)
    
    with st.expander("About EasyJet"):
        st.markdown("""
        ### Company Overview
        
        EasyJet plc is a British multinational low-cost airline group headquartered at London Luton Airport. 
        It operates domestic and international scheduled services on over 1,000 routes in more than 30 countries.
        
        ### Key Business Facts
        
        - **Founded**: 1995
        - **Fleet Size**: ~300 aircraft
        - **Destinations**: 150+ airports
        - **Business Model**: Low-cost carrier focusing on short to medium-haul European routes
        - **Market Position**: One of Europe's leading low-cost airlines
        
        ### Investment Considerations
        
        - Highly sensitive to fuel prices and currency fluctuations
        - Seasonal business with higher revenues during summer months
        - Competitive industry with other low-cost carriers like Ryanair and Wizz Air
        - Growing focus on sustainability and carbon reduction initiatives
        """)

with main_tab3:
    # Settings page
    st.header("Dashboard Settings")
    
    st.markdown("### Visualization Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Enable 3D Visualizations", value=True)
        st.checkbox("Show Confidence Intervals", value=True)
        st.checkbox("Display Value Annotations", value=True)
    
    with col2:
        st.selectbox("Color Theme", ["Blue/Orange (Default)", "Green/Purple", "Red/Blue", "Monochrome", "Corporate"])
        st.selectbox("Chart Style", ["Modern", "Minimal", "Classic", "Dark", "Print-friendly"])
        
    st.divider()
    
    st.markdown("### Data Settings")
    
    st.slider("Forecast Horizon (Years)", min_value=5, max_value=15, value=10)
    st.slider("Monte Carlo Iterations", min_value=100, max_value=5000, value=1000, step=100)
    
    st.button("Apply Settings", use_container_width=True)
    
    st.divider()
    
    st.markdown("### About This Tool")
    st.markdown("""
    This DCF Analysis Dashboard was created to provide advanced financial analysis capabilities for EasyJet's valuation model.
    
    **Version**: 2.0.0  
    **Last Updated**: March 24, 2025  
    **Built with**: Streamlit, Plotly, Pandas
    """)

# Footer
st.markdown("""
<div style="background-color:#F0F2F6; padding:10px; border-radius:5px; margin-top:20px; text-align:center;">
  <p style="margin:0; font-size:12px; color:#555;">
    This interactive DCF analysis dashboard is for educational and analytical purposes only. 
    It is not financial advice. Data is based on historical information and financial projections.
  </p>
</div>
""", unsafe_allow_html=True)
