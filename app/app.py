# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

# --- Load dataset ---
df = pd.read_csv('../data/destinations.csv')  # adjust path if needed

# --- Clean and preprocess ---
# Strip extra spaces in categorical columns
for col in ['Type', 'Climate', 'Season']:
    df[col] = df[col].astype(str).str.strip()

# Ensure numeric columns are numbers
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
df['Duration_days'] = pd.to_numeric(df['Duration_days'], errors='coerce')

# Drop rows with missing data
df = df.dropna(subset=['Cost', 'Duration_days', 'Type', 'Climate', 'Season'])

# --- Encode categorical features ---
le_type = LabelEncoder()
le_climate = LabelEncoder()
le_season = LabelEncoder()

df['Type_enc'] = le_type.fit_transform(df['Type'])
df['Climate_enc'] = le_climate.fit_transform(df['Climate'])
df['Season_enc'] = le_season.fit_transform(df['Season'])

# --- ML model ---
features = ['Cost', 'Type_enc', 'Climate_enc', 'Duration_days', 'Season_enc']
X = df[features].copy()
X = X.apply(pd.to_numeric, errors='coerce')  # ensure numeric
X = X.dropna()

model = NearestNeighbors(n_neighbors=5)
model.fit(X)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Travel Planner - India", layout="wide")
st.title("🌏 AI Travel Planner for Students - India Edition")

# Sidebar inputs
st.sidebar.header("Enter Your Preferences")
budget = st.sidebar.slider("Budget (INR)", 1000, 5000, 2000, step=100)
duration = st.sidebar.slider("Trip Duration (days)", 1, 14, 5)
travel_type = st.sidebar.selectbox("Travel Type", df['Type'].unique())
climate = st.sidebar.selectbox("Preferred Climate", df['Climate'].unique())
season = st.sidebar.selectbox("Preferred Season", df['Season'].unique())

# Encode user input
input_data = [[
    budget,
    le_type.transform([travel_type])[0],
    le_climate.transform([climate])[0],
    duration,
    le_season.transform([season])[0]
]]

# --- Get recommendations ---
distances, indices = model.kneighbors(input_data)
recommended = df.iloc[indices[0]].copy()

# Calculate cost per day
recommended['Cost_per_day'] = (recommended['Cost'] / recommended['Duration_days']).round(2)

# Highlight cheapest destination
cheapest_index = recommended['Cost_per_day'].idxmin()

# Display recommendations
st.subheader(f"Top {len(recommended)} Recommended Destinations:")

for i, row in recommended.iterrows():
    if i == cheapest_index:
        st.markdown(f"### 🏷️ {row['Destination']} (Cheapest per day!)")
    else:
        st.markdown(f"### {row['Destination']}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Cost (INR)", row['Cost'])
    col2.metric("Duration (days)", row['Duration_days'])
    col3.metric("Cost per Day (INR)", row['Cost_per_day'])
    col4.metric("Travel Type", row['Type'])
    col5.metric("Season", row['Season'])

# Summary table
st.markdown("### Summary Table")
st.dataframe(recommended[['Destination', 'Cost', 'Cost_per_day', 'Type', 'Climate', 'Season', 'Duration_days']])
