import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mysql.connector
from mysql.connector import Error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Marketing Campaign Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BETTER UI
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS AND PREPROCESSORS
# ============================================
@st.cache_resource
def load_models():
    """Load all saved models and preprocessors"""
    try:
        models = {
            'regression': joblib.load('models/best_regression_model.pkl'),
            'classification': joblib.load('models/best_classification_model.pkl'),
            'roi': joblib.load('models/roi_prediction_model.pkl'),
            'scaler_reg': joblib.load('models/scaler_regression.pkl'),
            'scaler_clf': joblib.load('models/scaler_classification.pkl'),
            'scaler_roi': joblib.load('models/scaler_roi.pkl'),
            'label_encoders': joblib.load('models/label_encoders.pkl'),
            'channel_encoder': joblib.load('models/channel_encoder.pkl'),
            'reg_features': joblib.load('models/regression_features.pkl'),
            'clf_features': joblib.load('models/classification_features.pkl'),
            'roi_features': joblib.load('models/roi_features.pkl')
        }
        return models
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("Please run the analysis notebook first to generate models.")
        return None

# ============================================
# DATABASE CONNECTION
# ============================================
@st.cache_resource
def get_db_connection():
    """Create database connection"""
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Shalu0.34',
            database='marketing_campaign_db'
        )
        return conn
    except Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None

@st.cache_data(ttl=600)
def load_data_from_db():
    """Load data from MySQL database"""
    conn = get_db_connection()
    if conn:
        query = "SELECT * FROM marketing_campaigns"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    return None

# ============================================
# PREDICTION FUNCTIONS
# ============================================
def prepare_features(input_data, feature_list, scaler, is_classification=False):
    """Prepare features for prediction"""
    # Create DataFrame with all required features
    features_df = pd.DataFrame(columns=feature_list)
    
    # Fill with input data
    for col in feature_list:
        if col in input_data:
            features_df[col] = [input_data[col]]
        else:
            features_df[col] = [0]
    
    # Scale features
    features_scaled = scaler.transform(features_df)
    return features_scaled

def predict_revenue(input_data, models):
    """Predict revenue"""
    try:
        features = prepare_features(
            input_data, 
            models['reg_features'], 
            models['scaler_reg']
        )
        prediction = models['regression'].predict(features)[0]
        return max(0, prediction)  # Revenue can't be negative
    except Exception as e:
        st.error(f"Revenue prediction error: {e}")
        return None

def predict_roi(input_data, models):
    """Predict ROI"""
    try:
        features = prepare_features(
            input_data,
            models['roi_features'],
            models['scaler_roi']
        )
        prediction = models['roi'].predict(features)[0]
        return prediction
    except Exception as e:
        st.error(f"ROI prediction error: {e}")
        return None

def predict_profitability(input_data, models):
    """Predict if campaign will be profitable"""
    try:
        features = prepare_features(
            input_data,
            models['clf_features'],
            models['scaler_clf'],
            is_classification=True
        )
        prediction = models['classification'].predict(features)[0]
        probability = models['classification'].predict_proba(features)
        # probability is an array like [[prob_loss, prob_profit]]
        return prediction, probability[0]
    except Exception as e:
        st.error(f"Profitability prediction error: {e}")
        return None, None

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_metrics_dashboard(df):
    """Create KPI metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Campaigns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        profitable = df['Profit_flag'].sum()
        profit_rate = (profitable / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{profit_rate:.1f}%</div>
            <div class="metric-label">Profit Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_revenue = df['Revenue'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{avg_revenue:,.0f}</div>
            <div class="metric-label">Avg Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_roi = df['ROI_Calculated'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_roi:,.0f}%</div>
            <div class="metric-label">Avg ROI</div>
        </div>
        """, unsafe_allow_html=True)

def create_revenue_chart(df):
    """Create revenue distribution chart"""
    fig = px.histogram(
        df, x='Revenue', 
        title='Revenue Distribution',
        labels={'Revenue': 'Revenue (₹)', 'count': 'Number of Campaigns'},
        color_discrete_sequence=['#2ecc71']
    )
    fig.add_vline(x=df['Revenue'].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: ₹{df['Revenue'].mean():,.0f}")
    return fig

def create_roi_chart(df):
    """Create ROI distribution chart"""
    fig = px.histogram(
        df, x='ROI_Calculated',
        title='ROI Distribution',
        labels={'ROI_Calculated': 'ROI (%)', 'count': 'Number of Campaigns'},
        color_discrete_sequence=['#3498db']
    )
    fig.add_vline(x=df['ROI_Calculated'].mean(), line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {df['ROI_Calculated'].mean():,.0f}%")
    return fig

def create_channel_performance(df):
    """Create channel performance chart"""
    channel_cols = [col for col in df.columns if col.startswith('Channel_')]
    if channel_cols:
        channel_data = []
        for ch in channel_cols:
            channel_name = ch.replace('Channel_', '')
            usage = df[ch].sum()
            avg_revenue = df[df[ch] == 1]['Revenue'].mean()
            channel_data.append({'Channel': channel_name, 'Usage Count': usage, 'Avg Revenue': avg_revenue})
        
        channel_df = pd.DataFrame(channel_data).sort_values('Usage Count', ascending=False)
        
        fig = px.bar(
            channel_df, x='Channel', y='Usage Count',
            title='Channel Usage',
            color='Usage Count',
            color_continuous_scale='Viridis'
        )
        return fig
    return None

# ============================================
# MAIN APP
# ============================================
def main():
    st.markdown('<h1 class="main-header">📊 Marketing Campaign Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/marketing.png", width=80)
        st.markdown("## 🎯 Navigation")
        
        page = st.radio(
            "Select Page",
            ["📈 Dashboard", "🔮 Predict Campaign", "📊 Analytics", "ℹ️ About"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Info")
        st.markdown("- Built with Streamlit")
        st.markdown("- ML Models: Random Forest, Decision Tree")
        st.markdown("- Database: MySQL")
    
    # Load data and models
    models = load_models()
    df = load_data_from_db()
    
    if df is None or models is None:
        st.warning("⚠️ Unable to load data or models. Please check your setup.")
        return
    
    # Dashboard Page
    if page == "📈 Dashboard":
        st.markdown("## 📊 Campaign Performance Dashboard")
        
        # Metrics
        create_metrics_dashboard(df)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_revenue_chart(df)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_roi_chart(df)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Channel performance
        st.markdown("### 📺 Channel Performance")
        fig3 = create_channel_performance(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        
        # Top campaigns
        st.markdown("### 🏆 Top 10 Campaigns by Revenue")
        top_campaigns = df.nlargest(10, 'Revenue')[['Revenue', 'ROI_Calculated', 'Profit_flag']]
        top_campaigns.columns = ['Revenue (₹)', 'ROI (%)', 'Profitable']
        st.dataframe(top_campaigns, use_container_width=True)
    
    # Predict Campaign Page
    elif page == "🔮 Predict Campaign":
        st.markdown("## 🔮 Predict Campaign Performance")
        st.markdown("Enter campaign details below to get predictions for Revenue, ROI, and Profitability.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Campaign Settings")
            
            # Duration
            duration = st.number_input("Campaign Duration (days)", min_value=1, max_value=365, value=30)
            
            # Impressions
            impressions = st.number_input("Expected Impressions", min_value=1000, max_value=1000000, value=50000, step=5000)
            
            # Clicks
            clicks = st.number_input("Expected Clicks", min_value=100, max_value=50000, value=5000, step=500)
            
            # Leads
            leads = st.number_input("Expected Leads", min_value=10, max_value=20000, value=2000, step=200)
            
            # Conversions
            conversions = st.number_input("Expected Conversions", min_value=1, max_value=10000, value=1000, step=100)
            
            # Acquisition Cost
            acquisition_cost = st.number_input("Acquisition Cost (₹)", min_value=10.0, max_value=10000.0, value=500.0, step=50.0)
        
        with col2:
            st.markdown("### 🎯 Targeting Settings")
            
            # Campaign Type
            campaign_type = st.selectbox(
                "Campaign Type",
                options=['Email', 'Influencer', 'Paid Ads', 'SEO', 'Social Media'],
                index=0
            )
            campaign_type_map = {'Email': 0, 'Influencer': 1, 'Paid Ads': 2, 'SEO': 3, 'Social Media': 4}
            
            # Target Audience
            target_audience = st.selectbox(
                "Target Audience",
                options=['College Students', 'Premium Shoppers', 'Tier 2 City Customers', 'Working Women', 'Youth'],
                index=0
            )
            audience_map = {'College Students': 0, 'Premium Shoppers': 1, 'Tier 2 City Customers': 2, 
                          'Working Women': 3, 'Youth': 4}
            
            # Language
            language = st.selectbox(
                "Language",
                options=['Bengali', 'English', 'Hindi', 'Tamil'],
                index=0
            )
            language_map = {'Bengali': 0, 'English': 1, 'Hindi': 2, 'Tamil': 3}
            
            # Customer Segment
            customer_segment = st.selectbox(
                "Customer Segment",
                options=['College Students', 'Premium Shoppers', 'Tier 2 City Customers', 'Working Women', 'Youth'],
                index=0
            )
            segment_map = {'College Students': 0, 'Premium Shoppers': 1, 'Tier 2 City Customers': 2,
                          'Working Women': 3, 'Youth': 4}
            
            # Brand
            brand = st.selectbox(
                "Brand",
                options=['Nykaa', 'Purplle', 'Tira'],
                index=0
            )
            brand_map = {'Nykaa': 0, 'Purplle': 1, 'Tira': 2}
            
            # Month and Quarter
            month = st.slider("Month", 1, 12, 6)
            quarter = (month - 1) // 3 + 1
        
        # Prepare input data
        input_data = {
            'Duration': duration,
            'Impressions': impressions,
            'Clicks': clicks,
            'Leads': leads,
            'Conversions': conversions,
            'Acquisition_Cost': acquisition_cost,
            'Engagement_Score': (clicks / impressions * 100) if impressions > 0 else 0,
            'Month': month,
            'Quarter': quarter,
            'Year': 2024,
            'Channel_Email': 0, 'Channel_Facebook': 0, 'Channel_Google': 0,
            'Channel_Instagram': 0, 'Channel_WhatsApp': 0, 'Channel_YouTube': 0,
            'Campaign_Type_encoded': campaign_type_map[campaign_type],
            'Target_Audience_encoded': audience_map[target_audience],
            'Language_encoded': language_map[language],
            'Customer_Segment_encoded': segment_map[customer_segment],
            'Brand_encoded': brand_map[brand]
        }
        
        # Predict button
        if st.button("🚀 Predict Campaign Performance", type="primary", use_container_width=True):
            with st.spinner("Calculating predictions..."):
                revenue_pred = predict_revenue(input_data, models)
                roi_pred = predict_roi(input_data, models)
                profit_pred, profit_prob = predict_profitability(input_data, models)
            
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            # Results cards
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                if revenue_pred:
                    st.metric(
                        "💰 Predicted Revenue",
                        f"₹{revenue_pred:,.2f}",
                        delta=None
                    )
            
            with res_col2:
                if roi_pred:
                    st.metric(
                        "📈 Predicted ROI",
                        f"{roi_pred:,.2f}%",
                        delta=None
                    )
            
            with res_col3:
                if profit_pred is not None:
                    status = "✅ PROFITABLE" if profit_pred == 1 else "❌ LOSS"
                    color = "green" if profit_pred == 1 else "red"
                    
                    # Safely get confidence value
                    try:
                        if profit_pred == 1 and len(profit_prob) > 1:
                            confidence = profit_prob[1]
                        elif profit_pred == 0 and len(profit_prob) > 0:
                            confidence = profit_prob[0]
                        else:
                            confidence = profit_prob[0] if len(profit_prob) > 0 else 0.5
                    except (TypeError, IndexError):
                        confidence = 0.5
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {color};">{status}</div>
                        <div class="metric-label">Profitability Prediction</div>
                        <div class="metric-label">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            
            if profit_pred == 1:
                st.success(f"✅ This campaign is predicted to be **PROFITABLE** with {confidence:.1%} confidence. The estimated ROI is {roi_pred:,.0f}% with expected revenue of ₹{revenue_pred:,.2f}.")
            else:
                st.warning(f"⚠️ This campaign is predicted to be a **LOSS** with {confidence:.1%} confidence. Consider adjusting your campaign strategy.")
    
    # Analytics Page
    elif page == "📊 Analytics":
        st.markdown("## 📊 Advanced Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Revenue Analysis", "ROI Analysis", "Channel Analysis"])
        
        with tab1:
            st.markdown("### Revenue by Brand")
            brand_revenue = df.groupby('Brand_encoded')['Revenue'].mean()
            brand_names = {0: 'Nykaa', 1: 'Purplle', 2: 'Tira'}
            brand_revenue.index = brand_revenue.index.map(brand_names)
            
            fig = px.bar(
                x=brand_revenue.index, y=brand_revenue.values,
                title='Average Revenue by Brand',
                labels={'x': 'Brand', 'y': 'Avg Revenue (₹)'},
                color=brand_revenue.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Revenue by Campaign Type")
            campaign_revenue = df.groupby('Campaign_Type_encoded')['Revenue'].mean()
            campaign_names = {0: 'Email', 1: 'Influencer', 2: 'Paid Ads', 3: 'SEO', 4: 'Social Media'}
            campaign_revenue.index = campaign_revenue.index.map(campaign_names)
            
            fig2 = px.bar(
                x=campaign_revenue.index, y=campaign_revenue.values,
                title='Average Revenue by Campaign Type',
                labels={'x': 'Campaign Type', 'y': 'Avg Revenue (₹)'},
                color=campaign_revenue.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.markdown("### ROI by Brand")
            brand_roi = df.groupby('Brand_encoded')['ROI_Calculated'].mean()
            brand_roi.index = brand_roi.index.map(brand_names)
            
            fig = px.bar(
                x=brand_roi.index, y=brand_roi.values,
                title='Average ROI by Brand',
                labels={'x': 'Brand', 'y': 'Avg ROI (%)'},
                color=brand_roi.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### ROI by Target Audience")
            audience_roi = df.groupby('Target_Audience_encoded')['ROI_Calculated'].mean()
            audience_names = {0: 'College Students', 1: 'Premium Shoppers', 2: 'Tier 2 City Customers',
                            3: 'Working Women', 4: 'Youth'}
            audience_roi.index = audience_roi.index.map(audience_names)
            
            fig2 = px.bar(
                x=audience_roi.index, y=audience_roi.values,
                title='Average ROI by Target Audience',
                labels={'x': 'Target Audience', 'y': 'Avg ROI (%)'},
                color=audience_roi.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.markdown("### Channel Usage Analysis")
            fig = create_channel_performance(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Channel Performance Metrics")
            channel_cols = [col for col in df.columns if col.startswith('Channel_')]
            if channel_cols:
                channel_stats = []
                for ch in channel_cols:
                    channel_name = ch.replace('Channel_', '')
                    channel_data = df[df[ch] == 1]
                    stats = {
                        'Channel': channel_name,
                        'Usage Count': len(channel_data),
                        'Avg Revenue': channel_data['Revenue'].mean(),
                        'Avg ROI': channel_data['ROI_Calculated'].mean(),
                        'Profit Rate': (channel_data['Profit_flag'].sum() / len(channel_data)) * 100
                    }
                    channel_stats.append(stats)
                
                channel_df = pd.DataFrame(channel_stats).sort_values('Profit Rate', ascending=False)
                st.dataframe(channel_df.round(2), use_container_width=True)
    
    # About Page
    else:
        st.markdown("## ℹ️ About This Application")
        
        st.markdown("""
        ### 🎯 Purpose
        This application helps marketing teams predict campaign performance before launch, enabling data-driven decision making.
        
        ### 🤖 Machine Learning Models
        - **Revenue Prediction**: Random Forest Regressor (R² = 0.75)
        - **ROI Prediction**: Random Forest Regressor (R² = 0.86)
        - **Profitability Prediction**: Decision Tree Classifier (F1-Score = 1.00)
        
        ### 📊 Data Source
        - Campaign data from 3 beauty brands: Nykaa, Purplle, and Tira
        - Total campaigns analyzed: 52,439
        - Time period: Multi-year data
        
        ### 🛠️ Tech Stack
        - **Frontend**: Streamlit
        - **Backend**: Python, Scikit-learn, Pandas
        - **Database**: MySQL
        - **Visualization**: Plotly, Matplotlib
        
        ### 📞 Support
        For issues or questions, please contact the development team.
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **💡 Top Performing Channels**
            - Google: 100% profit rate
            - Instagram: High engagement
            - Email: Consistent ROI
            """)
        
        with col2:
            st.success("""
            **🎯 Best Practices**
            - Target Working Women segment
            - Use Tamil/Hindi language campaigns
            - Run campaigns mid-year for best results
            """)

# ============================================
# RUN THE APP
# ============================================
if __name__ == "__main__":
    main()