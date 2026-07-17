import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
from decimal import Decimal

# --- CONFIGURATION ---
st.set_page_config(page_title="Hotel Revenue Forecaster", layout="wide")
st.title("📈 Downstream Revenue & Demand Dashboard")
st.markdown("Translating Attention-LSTM demand predictions into actionable revenue intelligence.")

# --- AWS DYNAMODB CONNECTION ---
@st.cache_data(ttl=600) # Cache data for 10 minutes to save read capacities
def load_forecast_data():
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table('ForecastedDemand')
    
    # Scan table (fine for small Free Tier datasets)
    response = table.scan()
    items = response.get('Items', [])
    
    if not items:
        return pd.DataFrame()
        
    df = pd.DataFrame(items)
    # Convert DynamoDB Decimals back to floats
    df['predicted_demand'] = df['predicted_demand'].apply(float)
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    
    # Ensure data is strictly ordered by hotel_id and date for UI consistency
    df = df.sort_values(by=['hotel_id', 'forecast_date']).reset_index(drop=True)
    return df

# --- BUSINESS LOGIC ---
# Standard Average Daily Rate (ADR) to calculate revenue impact
MOCK_ADR = 150.00 

df = load_forecast_data()

if df.empty:
    st.warning("No forecast data found in DynamoDB. Please run the Step Functions pipeline first.")
else:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    selected_hotel = st.sidebar.selectbox("Select Hotel", df['hotel_id'].unique())
    
    # Filter dataframe
    hotel_df = df[df['hotel_id'] == selected_hotel].copy()
    hotel_df['projected_revenue'] = hotel_df['predicted_demand'] * MOCK_ADR

    # --- TOP LEVEL METRICS ---
    total_demand = hotel_df['predicted_demand'].sum()
    total_revenue = hotel_df['projected_revenue'].sum()
    peak_day = hotel_df.loc[hotel_df['predicted_demand'].idxmax()]['forecast_date'].strftime('%Y-%m-%d')
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Projected Demand (Rooms)", f"{int(total_demand):,}")
    col2.metric("Total Projected Revenue", f"${total_revenue:,.2f}")
    col3.metric("Peak Demand Date", peak_day)

    st.divider()

    # --- VISUALIZATIONS ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Predicted Room Demand (Next 30 Days)")
        fig_demand = px.line(
            hotel_df, 
            x='forecast_date', 
            y='predicted_demand',
            markers=True,
            line_shape='spline',
            color_discrete_sequence=['#ff4b4b']
        )
        fig_demand.update_layout(xaxis_title="Date", yaxis_title="Rooms Needed")
        st.plotly_chart(fig_demand, use_container_width=True)

    with col_chart2:
        st.subheader("Projected Daily Revenue ($)")
        fig_rev = px.bar(
            hotel_df, 
            x='forecast_date', 
            y='projected_revenue',
            color='projected_revenue',
            color_continuous_scale='Viridis'
        )
        fig_rev.update_layout(xaxis_title="Date", yaxis_title="Revenue (USD)")
        st.plotly_chart(fig_rev, use_container_width=True)
        
    # --- RAW DATA TABLE ---
    st.subheader("Forecast Ledger")
    # Displaying in strict order as formatted above
    st.dataframe(
        hotel_df[['hotel_id', 'forecast_date', 'predicted_demand', 'projected_revenue']].style.format({
            'predicted_demand': '{:.0f}',
            'projected_revenue': '${:.2f}'
        }),
        use_container_width=True
    )