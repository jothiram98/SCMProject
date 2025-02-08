import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Custom Styling with Background Image & Logo
st.markdown(
    """
    <style>
        .stApp {
            background: url("background.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        .main-title {
            text-align: center;
            color: #fff;
            font-size: 36px;
            font-weight: bold;
        }
        .sidebar .sidebar-content {
            background-color: #2E3B4E; color: white;
        }
        .stButton>button {
            background-color: #FF5733; color: white; border-radius: 10px;
        }
        .logo-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
        }
        .logo-container img {
            max-width: 150px;
            height: auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)
import base64
@st.cache_data  # ✅ Use st.cache_data instead of st.cache
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with your background image
set_png_as_page_bg('background.jpg')
# Streamlit App Title
st.image("logo2.jpg", width=110)
st.markdown('<p class="main-title">🛒 AI-Powered Inventory Optimization</p>', unsafe_allow_html=True)
st.sidebar.image("logo1.png", width=150)
st.sidebar.header("📂 Upload Inventory Data")
# Upload CSV File
uploaded_file = st.sidebar.file_uploader("Upload your inventory CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    required_columns = ['Product_ID', 'Current_Stock', 'Sales_Last_Week', 'Sales_Last_Month', 
                        'Lead_Time_Days', 'Shelf_Life_Days', 'Seasonality_Index', 
                        'Weather_Index', 'Economic_Index', 'Demand_Next_Week']
    
    if all(col in data.columns for col in required_columns):
        st.success("✅ CSV Loaded Successfully!")
        
        # Handle missing values & ensure numeric columns
        data = data.dropna()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data = data[numeric_cols]
        
        # Train AI Model
        X = data[['Current_Stock', 'Sales_Last_Week', 'Sales_Last_Month', 'Lead_Time_Days', 
                  'Shelf_Life_Days', 'Seasonality_Index', 'Weather_Index', 'Economic_Index']]
        y = data['Demand_Next_Week']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        product_id = st.sidebar.selectbox("📌 Select Product ID", data['Product_ID'])
        product_data = data[data['Product_ID'] == product_id]

        if not product_data.empty:
            st.subheader(f"🛍️ Product ID: {product_id}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📦 Current Stock", int(product_data['Current_Stock']))
                st.metric("📊 Sales Last Week", int(product_data['Sales_Last_Week']))
            with col2:
                st.metric("📅 Sales Last Month", int(product_data['Sales_Last_Month']))
                st.metric("⏳ Lead Time (Days)", int(product_data['Lead_Time_Days']))
            with col3:
                st.metric("⏳ Shelf Life (Days)", int(product_data['Shelf_Life_Days']))
                st.metric("📈 Seasonality Index", round(product_data['Seasonality_Index'].values[0], 2))

            # Predict Demand
            predicted_demand = model.predict(product_data[['Current_Stock', 'Sales_Last_Week', 'Sales_Last_Month', 
                                                           'Lead_Time_Days', 'Shelf_Life_Days', 'Seasonality_Index', 
                                                           'Weather_Index', 'Economic_Index']])[0]
            
            # Adjust demand using normalized multipliers
            seasonality_factor = np.clip(product_data['Seasonality_Index'].values[0], 0.5, 1.5)
            weather_factor = np.clip(product_data['Weather_Index'].values[0], 0.5, 1.5)
            economic_factor = np.clip(product_data['Economic_Index'].values[0], 0.5, 1.5)
            expiry_factor = max(0.5, product_data['Shelf_Life_Days'].values[0] / 180)
            
            adjusted_demand = predicted_demand * seasonality_factor * weather_factor * economic_factor * expiry_factor
            reorder_point = adjusted_demand * int(product_data['Lead_Time_Days'])
            recommended_stock = reorder_point + adjusted_demand * 2
            
            # Display AI Predictions
            st.subheader("📊 AI Recommendations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📉 Predicted Demand", int(predicted_demand))
            with col2:
                st.metric("🔄 Adjusted Demand", int(adjusted_demand))
            with col3:
                st.metric("🛒 Reorder Point", int(reorder_point))
            
            st.metric("📌 Recommended Stock Level", int(recommended_stock))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=['Current Stock', 'Predicted Demand', 'Adjusted Demand', 'Recommended Stock'],
                                 y=[product_data['Current_Stock'].values[0], predicted_demand, adjusted_demand, recommended_stock],
                                 marker_color=['blue', 'orange', 'green', 'red']))
            fig.update_layout(title=f"📊 AI Recommendations for Product {product_id}", xaxis_title="Metrics", yaxis_title="Values")
            st.plotly_chart(fig)
              # 📊 Additional Graphs for Better Insights
            st.markdown("<h2 class='section-title'>📊 Data Insights</h2>", unsafe_allow_html=True)
# 📊 3D Stock vs Demand Analysis (Now Interactive!)
            fig1 = px.scatter_3d(
                data, 
                x='Current_Stock', 
                y='Demand_Next_Week', 
                z='Seasonality_Index', 
                color='Seasonality_Index', 
                title="📊 3D Stock vs Demand Analysis",
                opacity=0.8,
                size_max=10
            )
            fig1.update_traces(marker=dict(size=5, symbol='circle', line=dict(width=1)))
            st.plotly_chart(fig1)
            # 📈 Improved Reorder Point Distribution (Smooth Density)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=data['Demand_Next_Week'], 
                nbinsx=20, 
                marker=dict(color='rgba(255,100,100,0.7)', line=dict(color='red', width=1)),
                opacity=0.75
            ))
            fig2.add_trace(go.Scatter(
                x=np.histogram(data['Demand_Next_Week'], bins=20)[1],
                y=np.histogram(data['Demand_Next_Week'], bins=20)[0],
                mode='lines',
                line=dict(color='black', width=2),
                name="Smooth Curve"
            ))
            fig2.update_layout(title="📌 Reorder Point Distribution (Smoothed)", bargap=0.05)
            st.plotly_chart(fig2)

            # 🔥 Enhanced Correlation Heatmap with Annotations
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, square=True, ax=ax)
            st.pyplot(fig)
            
        
        st.sidebar.write(f"⚡ **Model Accuracy (MAE):** {round(mae, 2)}")
    else:
        st.error("❌ Uploaded CSV does not have the required columns.")
else:
    st.info("📂 Upload a CSV file to get started.")
