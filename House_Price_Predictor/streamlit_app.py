import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
import time

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS with animations and professional styling
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2c3e50, #3498db);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 2rem 0;
        animation: fadeInDown 1s ease-out;
    }
    
    /* Keyframe Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -200px 0;
        }
        100% {
            background-position: calc(200px + 100%) 0;
        }
    }
    
    /* Enhanced Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        animation: slideInUp 0.8s ease-out;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.4);
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: -200px;
        width: 200px;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        animation: fadeInDown 0.5s ease-out;
    }
    
    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    /* Sidebar Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
        animation: slideInUp 0.7s ease-out;
    }
    
    /* Button Enhancement */
    .predict-button {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.3);
        width: 100%;
        margin: 1rem 0;
    }
    
    .predict-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
        animation: pulse 0.6s ease-in-out;
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-left: 4px solid #17a2b8;
        padding: 1.5rem;
        border-radius: 0 15px 15px 0;
        margin: 1rem 0;
        animation: slideInUp 0.8s ease-out;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    /* Tab Styling */
    .tab-container {
        animation: fadeInDown 0.9s ease-out;
    }
    
    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 4px;
        background-color: #e9ecef;
        border-radius: 2px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        border-radius: 2px;
        animation: progressAnimation 2s ease-out;
    }
    
    @keyframes progressAnimation {
        0% { width: 0%; }
        100% { width: 100%; }
    }
    
    /* Floating Elements */
    .floating-element {
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Data Table Enhancement */
    .dataframe {
        animation: slideInUp 1s ease-out;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }
    
    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideInUp 0.7s ease-out;
    }
    
    /* Error Message */
    .error-message {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideInUp 0.7s ease-out;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_model_artifacts():
    """Load the trained model and associated artifacts"""
    try:
        model = joblib.load("models/house_price_model.pkl")  # Add models/
        label_encoders = joblib.load("models/label_encoders.pkl")  # Add models/
        feature_names = joblib.load("models/feature_names.pkl")  # Add models/
        metrics = joblib.load("models/model_metrics.pkl")  # Add models/
        feature_importance = pd.read_csv("models/feature_importance.csv")  # Add models/
        return model, label_encoders, feature_names, metrics, feature_importance
    except FileNotFoundError as e:
        st.error(
            f"Model files not found. Please run model_training.py first. Error: {e}"
        )
        return None, None, None, None, None


def show_loading_animation():
    """Display a loading animation"""
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            """
        <div class="loading-container">
            <div class="loading-spinner"></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("**🔄 Processing your prediction...**")

    time.sleep(1.5)  # Simulate processing time
    loading_placeholder.empty()


def create_input_form():
    """Create input form for user to enter house features with enhanced styling"""

    st.sidebar.markdown(
        """
    <div class="sidebar-header">
        <h2 style="margin:0; color:#2c3e50;">🏠 House Features Input</h2>
        <p style="margin:0.5rem 0 0 0; color:#7f8c8d; font-size:0.9rem;">Enter property details below</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    inputs = {}

    # Group inputs by category with visual separators
    st.sidebar.markdown("### 📏 **Property Dimensions**")
    inputs["LotArea"] = st.sidebar.number_input(
        "Lot Area (sq ft)", min_value=1000, max_value=50000, value=10000, key="lot_area"
    )
    inputs["1stFlrSF"] = st.sidebar.number_input(
        "1st Floor Area (sq ft)",
        min_value=500,
        max_value=3000,
        value=1200,
        key="first_floor",
    )
    inputs["2ndFlrSF"] = st.sidebar.number_input(
        "2nd Floor Area (sq ft)",
        min_value=0,
        max_value=2000,
        value=800,
        key="second_floor",
    )
    inputs["GrLivArea"] = st.sidebar.number_input(
        "Above Grade Living Area (sq ft)",
        min_value=600,
        max_value=4000,
        value=1500,
        key="gr_liv_area",
    )
    inputs["TotalBsmtSF"] = st.sidebar.number_input(
        "Total Basement Area (sq ft)",
        min_value=0,
        max_value=3000,
        value=1000,
        key="total_bsmt",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⭐ **Quality & Condition**")
    inputs["OverallQual"] = st.sidebar.slider(
        "Overall Quality", min_value=1, max_value=10, value=7, key="overall_qual"
    )
    inputs["OverallCond"] = st.sidebar.slider(
        "Overall Condition", min_value=1, max_value=10, value=7, key="overall_cond"
    )
    inputs["ExterQual"] = st.sidebar.selectbox(
        "Exterior Quality", ["Ex", "Gd", "TA", "Fa"], key="exter_qual"
    )
    inputs["KitchenQual"] = st.sidebar.selectbox(
        "Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], key="kitchen_qual"
    )
    inputs["HeatingQC"] = st.sidebar.selectbox(
        "Heating Quality", ["Ex", "Gd", "TA", "Fa", "Po"], key="heating_qc"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 **Property Age**")
    inputs["YearBuilt"] = st.sidebar.number_input(
        "Year Built", min_value=1900, max_value=2023, value=2000, key="year_built"
    )
    inputs["YearRemodAdd"] = st.sidebar.number_input(
        "Year Remodeled", min_value=1900, max_value=2023, value=2000, key="year_remod"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏠 **Rooms & Features**")
    inputs["FullBath"] = st.sidebar.number_input(
        "Full Bathrooms", min_value=0, max_value=4, value=2, key="full_bath"
    )
    inputs["HalfBath"] = st.sidebar.number_input(
        "Half Bathrooms", min_value=0, max_value=3, value=1, key="half_bath"
    )
    inputs["BedroomAbvGr"] = st.sidebar.number_input(
        "Bedrooms Above Grade", min_value=0, max_value=8, value=3, key="bedroom_abv"
    )
    inputs["KitchenAbvGr"] = st.sidebar.number_input(
        "Kitchens Above Grade", min_value=0, max_value=3, value=1, key="kitchen_abv"
    )
    inputs["TotRmsAbvGrd"] = st.sidebar.number_input(
        "Total Rooms Above Grade", min_value=2, max_value=15, value=7, key="tot_rooms"
    )
    inputs["Fireplaces"] = st.sidebar.number_input(
        "Fireplaces", min_value=0, max_value=4, value=1, key="fireplaces"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚗 **Garage & Outdoor**")
    inputs["GarageCars"] = st.sidebar.number_input(
        "Garage Car Capacity", min_value=0, max_value=4, value=2, key="garage_cars"
    )
    inputs["GarageArea"] = st.sidebar.number_input(
        "Garage Area (sq ft)", min_value=0, max_value=1500, value=500, key="garage_area"
    )
    inputs["GarageType"] = st.sidebar.selectbox(
        "Garage Type", ["Attchd", "Detchd", "BuiltIn", "None"], key="garage_type"
    )
    inputs["WoodDeckSF"] = st.sidebar.number_input(
        "Wood Deck Area (sq ft)",
        min_value=0,
        max_value=1000,
        value=150,
        key="wood_deck",
    )
    inputs["OpenPorchSF"] = st.sidebar.number_input(
        "Open Porch Area (sq ft)",
        min_value=0,
        max_value=500,
        value=50,
        key="open_porch",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏘️ **Location & Type**")
    inputs["MSSubClass"] = st.sidebar.number_input(
        "Building Class", min_value=20, max_value=190, value=60, key="ms_subclass"
    )
    inputs["MSZoning"] = st.sidebar.selectbox(
        "Zoning Classification", ["RL", "RM", "FV", "RH", "C"], key="ms_zoning"
    )
    inputs["Neighborhood"] = st.sidebar.selectbox(
        "Neighborhood",
        [
            "NAmes",
            "CollgCr",
            "OldTown",
            "Edwards",
            "Somerst",
            "Gilbert",
            "NridgHt",
            "Sawyer",
            "NWAmes",
            "SawyerW",
        ],
        key="neighborhood",
    )
    inputs["HouseStyle"] = st.sidebar.selectbox(
        "House Style",
        ["1Story", "2Story", "1.5Fin", "SLvl", "1.5Unf", "SFoyer", "2.5Unf", "2.5Fin"],
        key="house_style",
    )
    inputs["CentralAir"] = st.sidebar.selectbox(
        "Central Air", ["Y", "N"], key="central_air"
    )

    return inputs


def preprocess_inputs(inputs, label_encoders, feature_names):
    """Preprocess user inputs for prediction"""
    processed_inputs = {}

    for feature in feature_names:
        if feature in inputs:
            value = inputs[feature]
            if feature in label_encoders and label_encoders[feature] is not None:
                try:
                    processed_inputs[feature] = label_encoders[feature].transform(
                        [value]
                    )[0]
                except (ValueError, AttributeError):
                    processed_inputs[feature] = 0
            else:
                processed_inputs[feature] = value
        else:
            processed_inputs[feature] = 0

    return pd.DataFrame([processed_inputs])


def create_enhanced_charts(feature_importance):
    """Create enhanced visualizations with professional styling"""

    # Enhanced Feature Importance Chart
    fig_importance = px.bar(
        feature_importance.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="🎯 Top 10 Most Important Features",
        labels={"importance": "Feature Importance", "feature": "Features"},
        color="importance",
        color_continuous_scale=["#e8f4fd", "#1f77b4"],
    )

    fig_importance.update_layout(
        height=500,
        title_font_size=20,
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        coloraxis_showscale=False,
    )

    fig_importance.update_traces(
        marker_line_color="rgba(31, 119, 180, 0.8)", marker_line_width=1
    )

    return fig_importance


def display_enhanced_metrics(metrics):
    """Display model performance metrics with enhanced styling"""

    col1, col2, col3, col4 = st.columns(4)

    metrics_data = [
        ("Test R² Score", f"{metrics['test_r2']:.3f}", "📊", "#28a745"),
        ("Test MAE", f"${metrics['test_mae']:,.0f}", "💰", "#17a2b8"),
        ("Train R² Score", f"{metrics['train_r2']:.3f}", "📈", "#ffc107"),
        ("Train MAE", f"${metrics['train_mae']:,.0f}", "📉", "#dc3545"),
    ]

    columns = [col1, col2, col3, col4]

    for i, (label, value, icon, color) in enumerate(metrics_data):
        with columns[i]:
            st.markdown(
                f"""
            <div class="metric-card">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">{value}</div>
                    <div style="font-size: 0.9rem; color: #6c757d;">{label}</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    # Animated title
    st.markdown(
        '<h1 class="main-header">🏠 House Price Predictor</h1>', unsafe_allow_html=True
    )

    # Subtitle with animation
    st.markdown(
        """
    <div style="text-align: center; animation: fadeInDown 1.2s ease-out; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #7f8c8d; margin: 0;">
            Predict house prices with machine learning powered insights
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load model artifacts
    model, label_encoders, feature_names, metrics, feature_importance = (
        load_model_artifacts()
    )

    if model is None:
        st.stop()

    # Enhanced tabs
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(
        ["🔮 Make Prediction", "📊 Model Insights", "📈 Performance"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(
                """
            <div class="info-card">
                <h3 style="margin-top: 0; color: #2c3e50;">📋 How to Use</h3>
                <p style="margin-bottom: 0; color: #7f8c8d;">
                    1. Fill in property details in the sidebar<br>
                    2. Click the predict button<br>
                    3. Get instant price prediction with confidence range
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            inputs = create_input_form()

            # Enhanced predict button
            if st.button("🔮 Predict Price", type="primary", use_container_width=True):
                try:
                    # Show loading animation
                    show_loading_animation()

                    # Preprocess inputs
                    input_df = preprocess_inputs(inputs, label_encoders, feature_names)

                    # Make prediction
                    prediction = model.predict(input_df)[0]

                    # Display animated prediction
                    st.markdown(
                        f"""
                    <div class="prediction-box">
                        <h2 style="margin: 0 0 1rem 0; font-weight: 600;">🎉 Predicted House Price</h2>
                        <h1 style="margin: 0; font-size: 3rem; font-weight: 700;">${prediction:,.0f}</h1>
                        <p style="margin: 1rem 0 0 0; opacity: 0.9;">Based on the features you provided</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Progress bar animation
                    st.markdown(
                        """
                    <div class="progress-bar">
                        <div class="progress-fill"></div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Confidence interval with enhanced styling
                    std_error = metrics["test_mae"]
                    lower_bound = prediction - std_error
                    upper_bound = prediction + std_error

                    st.markdown(
                        f"""
                    <div class="success-message">
                        <strong>💡 Prediction Range:</strong> ${lower_bound:,.0f} - ${upper_bound:,.0f}
                        <br><small>This range represents the model's confidence interval</small>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Enhanced feature contribution analysis
                    st.markdown("### 📋 **Your Input Summary**")
                    summary_data = []
                    for feature, value in inputs.items():
                        if feature in feature_importance["feature"].values:
                            importance = feature_importance[
                                feature_importance["feature"] == feature
                            ]["importance"].values
                            if len(importance) > 0:
                                summary_data.append(
                                    {
                                        "Feature": str(feature),
                                        "Your Value": str(value),
                                        "Importance": f"{importance[0]:.3f}",
                                    }
                                )

                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        summary_df = summary_df.sort_values(
                            "Importance", ascending=False
                        )

                        st.markdown('<div class="dataframe">', unsafe_allow_html=True)
                        st.dataframe(summary_df, use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.markdown(
                        f"""
                    <div class="error-message">
                        <strong>❌ Error:</strong> {str(e)}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    with tab2:
        st.markdown('<div class="floating-element">', unsafe_allow_html=True)
        st.markdown("## 🎯 **Feature Importance Analysis**")
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            fig_importance = create_enhanced_charts(feature_importance)
            st.plotly_chart(fig_importance, use_container_width=True)

            st.markdown(
                """
            <div class="info-card">
                <h4 style="margin-top: 0; color: #2c3e50;">💡 Understanding Feature Importance</h4>
                <ul style="margin-bottom: 0; color: #7f8c8d;">
                    <li>Higher values indicate stronger influence on price predictions</li>
                    <li>These features are prioritized by the machine learning model</li>
                    <li>Focus on top features when evaluating property improvements</li>
                </ul>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced feature importance table
            st.markdown("### 📊 **Complete Feature Rankings**")
            st.markdown('<div class="dataframe">', unsafe_allow_html=True)
            st.dataframe(feature_importance, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.markdown(
                f"""
            <div class="error-message">
                <strong>❌ Error:</strong> {str(e)}
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown("## 📊 **Model Performance Dashboard**")

    try:
        display_enhanced_metrics(metrics)

        # Fixed Model Information section
        st.markdown(
            """
        <div class="info-card" style="margin-top: 2rem;">
            <h3 style="margin-top: 0; color: #2c3e50;">🤖 Model Information</h3>
            <div style="color: #7f8c8d;">
                <p><strong>Algorithm:</strong> Random Forest Regressor</p>
                <p><strong>Features Used:</strong> {} features from the house dataset</p>
                <p><strong>Training Method:</strong> 80/20 train-test split with cross-validation</p>
            </div>
        </div>
        """.format(len(feature_names) if feature_names else 0),
            unsafe_allow_html=True,
        )

        # Fixed Interpretation Guide section
        st.markdown(
            """
        <div class="info-card" style="margin-top: 1rem;">
            <h4 style="color: #2c3e50; margin-top: 0;">📖 Interpretation Guide</h4>
            <ul style="margin-bottom: 0; color: #7f8c8d;">
                <li><strong>R² Score:</strong> Closer to 1.0 indicates better model performance</li>
                <li><strong>MAE:</strong> Lower values indicate more accurate predictions</li>
                <li><strong>Train vs Test:</strong> Similar scores suggest good generalization</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.markdown(
            f"""
        <div class="error-message">
            <strong>❌ Error:</strong> {str(e)}
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
