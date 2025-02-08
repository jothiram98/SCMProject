import streamlit as st

st.set_page_config(page_title="Factor Explanation", layout="wide")

st.title("📘 Understanding Key Factors in AI-Powered Inventory Optimization")

st.header("📈 Seasonality Factor")
st.write("""
The **Seasonality Factor** represents how demand fluctuates during different times of the year.  
- High values (>1) indicate **peak seasons** (e.g., holiday sales).  
- Low values (<1) indicate **off-seasons** with reduced demand.  
- Formula Used: `seasonality_factor = np.clip(Seasonality_Index, 0.5, 1.5)`
""")

st.header("🌦️ Weather Factor")
st.write("""
The **Weather Factor** accounts for weather-related variations in demand.  
- Some products (e.g., umbrellas, ice creams) have seasonal dependencies.  
- Formula Used: `weather_factor = np.clip(Weather_Index, 0.5, 1.5)`
""")

st.header("💰 Economic Factor")
st.write("""
The **Economic Factor** considers macroeconomic trends affecting demand.  
- A booming economy may increase spending, while downturns reduce demand.  
- Formula Used: `economic_factor = np.clip(Economic_Index, 0.5, 1.5)`
""")

st.header("⏳ Expiry Factor")
st.write("""
The **Expiry Factor** adjusts demand based on product shelf life.  
- Short shelf life -> demand needs to be adjusted to prevent stock loss.  
- Formula Used: `expiry_factor = max(0.5, Shelf_Life_Days / 180)`
""")

st.header("📌 Final Demand Calculation")
st.write("""
The final **Adjusted Demand** is calculated as: 
         adjusted_demand = predicted_demand * seasonality_factor * weather_factor * economic_factor * expiry_factor 
         This helps **optimize stock levels** and reduce excess inventory.
""")

st.success("📌 Return to the main app using the sidebar!")