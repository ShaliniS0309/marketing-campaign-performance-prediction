import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Marketing Campaign Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Marketing Campaign Performance Prediction")
st.markdown("---")

# Load all models and preprocessors
@st.cache_resource
def load_models():
    try:
        reg_model = joblib.load('models/best_regression_model.pkl')
        clf_model = joblib.load('models/best_classification_model.pkl')
        scaler_reg = joblib.load('models/scaler_regression.pkl')
        scaler_clf = joblib.load('models/scaler_classification.pkl')
        label_encoders = joblib.load('models/label_encoders.pkl')
        channel_encoder = joblib.load('models/channel_encoder.pkl')
        reg_features = joblib.load('models/regression_features.pkl')
        clf_features = joblib.load('models/classification_features.pkl')
        return reg_model, clf_model, scaler_reg, scaler_clf, label_encoders, channel_encoder, reg_features, clf_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None, None, None

reg_model, clf_model, scaler_reg, scaler_clf, label_encoders, channel_encoder, reg_features, clf_features = load_models()

if reg_model is None:
    st.stop()

# Sidebar for input
st.sidebar.header("📝 Campaign Details")

# Input fields
campaign_type = st.sidebar.selectbox("Campaign Type", ['Email', 'Influencer', 'Paid Ads', 'SEO', 'Social Media'])
target_audience = st.sidebar.selectbox("Target Audience", ['College Students', 'Premium Shoppers', 'Tier 2 City Customers', 'Working Women', 'Youth'])
language = st.sidebar.selectbox("Language", ['English', 'Hindi', 'Tamil', 'Bengali'])
customer_segment = st.sidebar.selectbox("Customer Segment", ['College Students', 'Premium Shoppers', 'Tier 2 City Customers', 'Working Women', 'Youth'])

# Numerical inputs
col1, col2 = st.sidebar.columns(2)
with col1:
    duration = st.number_input("Duration (days)", min_value=1, max_value=365, value=30)
    impressions = st.number_input("Impressions", min_value=0, value=50000)
    clicks = st.number_input("Clicks", min_value=0, value=5000)
with col2:
    leads = st.number_input("Leads", min_value=0, value=500)
    conversions = st.number_input("Conversions", min_value=0, value=100)
    acquisition_cost = st.number_input("Acquisition Cost", min_value=0, value=10000)

# Channel selection
st.sidebar.subheader("📡 Marketing Channels Used")
channels = st.sidebar.multiselect(
    "Select channels (multiple allowed)",
    options=['Email', 'Facebook', 'Google', 'Instagram', 'WhatsApp', 'YouTube'],
    default=['Google', 'Facebook']
)

engagement_score = st.sidebar.slider("Engagement Score", min_value=0.0, max_value=10.0, value=5.0)

# Prediction button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Predict Campaign Performance", type="primary")

# Main content area
st.header("📈 Prediction Results")

if predict_button:
    with st.spinner("Processing prediction..."):
        # Prepare input data
        input_data = {}
        
        # Numerical features
        input_data['Duration'] = duration
        input_data['Impressions'] = impressions
        input_data['Clicks'] = clicks
        input_data['Leads'] = leads
        input_data['Conversions'] = conversions
        input_data['Acquisition_Cost'] = acquisition_cost
        input_data['Engagement_Score'] = engagement_score
        
        # Encode categorical features
        input_data['Campaign_Type_encoded'] = label_encoders['Campaign_Type'].transform([campaign_type])[0]
        input_data['Target_Audience_encoded'] = label_encoders['Target_Audience'].transform([target_audience])[0]
        input_data['Language_encoded'] = label_encoders['Language'].transform([language])[0]
        input_data['Customer_Segment_encoded'] = label_encoders['Customer_Segment'].transform([customer_segment])[0]
        
        # Encode channels (multi-label)
        channel_input = [channels] if channels else [[]]
        channel_encoded = channel_encoder.transform(channel_input)
        
        # Add channel features to input_data
        for i, channel in enumerate(channel_encoder.classes_):
            input_data[f'Channel_{channel}'] = channel_encoded[0][i]
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all expected columns are present
        # For Regression (no ROI, no Revenue, no Profit_Flag)
        for col in reg_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # For Classification (no ROI)
        for col in clf_features:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Select only required columns
        input_reg = input_df[reg_features]
        input_clf = input_df[clf_features]
        
        # Scale features
        input_reg_scaled = scaler_reg.transform(input_reg)
        input_clf_scaled = scaler_clf.transform(input_clf)
        
        # Make predictions
        predicted_revenue = reg_model.predict(input_reg_scaled)[0]
        predicted_profit_flag = clf_model.predict(input_clf_scaled)[0]
        predicted_profit_prob = clf_model.predict_proba(input_clf_scaled)[0]
        
        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Revenue Prediction")
            st.metric(
                label="Predicted Revenue",
                value=f"₹{predicted_revenue:,.2f}",
                delta=f"{predicted_revenue/100000:.1f}L"
            )
            
            # ROI Calculation (estimated)
            if acquisition_cost > 0:
                estimated_roi = ((predicted_revenue - acquisition_cost) / acquisition_cost) * 100
                st.metric(
                    label="Estimated ROI",
                    value=f"{estimated_roi:.1f}%",
                    delta="Projected"
                )
        
        with col2:
            st.subheader("📊 Profitability Prediction")
            if predicted_profit_flag == 1:
                st.success(f"✅ **PROFIT**")
                st.metric(
                    label="Confidence",
                    value=f"{predicted_profit_prob[1]*100:.1f}%",
                    delta="Profit Probability"
                )
            else:
                st.error(f"❌ **LOSS**")
                st.metric(
                    label="Confidence",
                    value=f"{predicted_profit_prob[0]*100:.1f}%",
                    delta="Loss Probability"
                )
        
        # Additional insights
        st.markdown("---")
        st.subheader("📋 Campaign Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric("Campaign Type", campaign_type)
            st.metric("Target Audience", target_audience)
        
        with summary_col2:
            st.metric("Language", language)
            st.metric("Customer Segment", customer_segment)
        
        with summary_col3:
            st.metric("Channels Used", ", ".join(channels) if channels else "None")
            st.metric("Duration", f"{duration} days")
        
        # Conversion metrics
        st.markdown("---")
        st.subheader("📊 Campaign Metrics")
        
        if impressions > 0:
            ctr = (clicks / impressions) * 100
            conversion_rate = (conversions / clicks) * 100 if clicks > 0 else 0
            cost_per_conversion = acquisition_cost / conversions if conversions > 0 else 0
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Click-Through Rate (CTR)", f"{ctr:.2f}%")
            with metric_col2:
                st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
            with metric_col3:
                st.metric("Cost per Conversion", f"₹{cost_per_conversion:.2f}")

else:
    st.info("👈 Enter campaign details in the sidebar and click 'Predict Campaign Performance'")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Marketing Campaign Performance Prediction System | Built with Streamlit & scikit-learn</p>
    </div>
""", unsafe_allow_html=True)