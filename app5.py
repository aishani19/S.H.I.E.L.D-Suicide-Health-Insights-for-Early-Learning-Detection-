import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Page Configuration
st.set_page_config(page_title="S.H.I.E.L.D[Suicide Health Insights for Early Learning & Detection]",
             layout="wide", page_icon="🌍")

# CSS Styling
st.markdown("""
    <style>
        .main {background-color: #f0f2f6; font-family: 'Segoe UI', sans-serif;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        h1, h2, h3, h4, h5 {color: #2c3e50;}
        .st-bx {background-color: #808080; border-radius: 5px; margin-bottom: 0.7rem;}
        .sidebar .sidebar-content {background-color: #808080;}
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/aishani billore/OneDrive/Desktop/II_Year_Data_Science_Project_2025-2026/master.csv")
    df = df[(df['year'] != 2016) & (~df['country'].isin(['Dominica', 'Saint Kitts and Nevis']))]
    df['country'] = df['country'].replace({
        'Bahamas': 'The Bahamas',
        'Cabo Verde': 'Cape Verde',
        'Republic of Korea': 'South Korea'
    })
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🌐 Filters")
year = st.sidebar.selectbox("Select Year", sorted(df['year'].unique()), index=0)
gender = st.sidebar.selectbox("Select Gender", ['both'] + df['sex'].unique().tolist())
age_group = st.sidebar.selectbox("Select Age Group", ['all'] + df['age'].unique().tolist())
country = st.sidebar.multiselect("Filter by Country", options=sorted(df['country'].unique()), default=[])

# Filter Data
filtered = df[df['year'] == year]
if gender != 'both':
    filtered = filtered[filtered['sex'] == gender]
if age_group != 'all':
    filtered = filtered[filtered['age'] == age_group]
if country:
    filtered = filtered[filtered['country'].isin(country)]

# Dashboard Title
st.title("🌍S.H.I.E.L.D[Suicide Health Insights for Early Learning & Detection] ")
st.markdown("""<div class="st-bx"><h4>Explore global suicide trends interactively. Filter by demographics such as gender, age, and country to uncover insights across years.</h4></div>""", unsafe_allow_html=True)

# Visualizations
st.subheader("🗺️ Suicide Distribution by Country")
map_data = filtered.groupby('country')['suicides_no'].sum().reset_index()
fig_map = px.choropleth(map_data, locations="country", locationmode="country names",
                        color="suicides_no", color_continuous_scale="Plasma",
                        title=f"Total Suicides by Country in {year}")
st.plotly_chart(fig_map, use_container_width=True)

st.subheader("👨‍🦰👩 Gender-wise Suicides")
gender_data = filtered.groupby('sex')['suicides_no'].sum().reset_index()
fig_gender = px.bar(gender_data, x='sex', y='suicides_no', color='sex',
                    title='Suicides by Gender', text_auto=True, color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig_gender, use_container_width=True)

st.subheader("📊 Age-wise Suicides")
age_data = filtered.groupby('age')['suicides_no'].sum().reset_index().sort_values('age')
fig_age = px.bar(age_data, x='age', y='suicides_no', color='age',
                 title='Suicides by Age Group', text_auto=True, color_discrete_sequence=px.colors.sequential.Aggrnyl)
st.plotly_chart(fig_age, use_container_width=True)

st.subheader("🏳️ Top Countries by Suicide Numbers")
country_data = filtered.groupby('country')['suicides_no'].sum().reset_index().sort_values(by='suicides_no', ascending=False).head(10)
fig_country = px.bar(country_data, x='country', y='suicides_no', color='country',
                     title='Top 10 Countries by Suicide Count', text_auto=True, color_discrete_sequence=px.colors.sequential.Viridis)
st.plotly_chart(fig_country, use_container_width=True)

st.subheader("📈 Global Yearly Suicide Trend")
trend_data = df.groupby('year')['suicides_no'].sum().reset_index()
fig_trend = px.line(trend_data, x='year', y='suicides_no', markers=True,
                    title='Global Suicide Trend Over Years')
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("🔥 Suicide Pattern: Gender vs Age")
heatmap_data = filtered.groupby(['sex', 'age'])['suicides_no'].sum().reset_index()
fig_heat = px.density_heatmap(heatmap_data, x='age', y='sex', z='suicides_no',
                              color_continuous_scale='Reds', title='Heatmap of Suicides by Gender and Age')
st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("📉 Suicide Rate per 100k Population by Country")
rate_data = filtered.groupby('country').agg({'suicides_no': 'sum', 'population': 'sum'}).reset_index()
rate_data['rate_per_100k'] = (rate_data['suicides_no'] / rate_data['population']) * 100000
fig_rate = px.bar(rate_data.sort_values('rate_per_100k', ascending=False).head(10),
                  x='country', y='rate_per_100k', color='country',
                  title='Top 10 Countries by Suicide Rate per 100k Population', text_auto=True,
                  color_discrete_sequence=px.colors.sequential.Blugrn)
st.plotly_chart(fig_rate, use_container_width=True)

st.subheader("🧓 Suicide Distribution by Generation")
generation_map = {
    '5-14 years': 'Gen Z', '15-24 years': 'Gen Z', '25-34 years': 'Millennials',
    '35-54 years': 'Gen X', '55-74 years': 'Boomers', '75+ years': 'Silent'
}
gen_df = filtered.copy()
gen_df['generation'] = gen_df['age'].map(generation_map)
generation_data = gen_df.groupby('generation')['suicides_no'].sum().reset_index()
fig_generation = px.pie(generation_data, values='suicides_no', names='generation', title='Suicide Distribution by Generation')
st.plotly_chart(fig_generation, use_container_width=True)

# ML Model Setup
ml_df = df.copy()
ml_df.dropna(subset=["sex", "age", "country", "suicides/100k pop"], inplace=True)
ml_df = ml_df[ml_df["suicides/100k pop"] > 0]
ml_df = ml_df[["year", "sex", "age", "country", "suicides/100k pop"]]

# Remove top 1% outliers
threshold = ml_df["suicides/100k pop"].quantile(0.99)
ml_df = ml_df[ml_df["suicides/100k pop"] < threshold]

# One-hot encoding
ml_df_encoded = pd.get_dummies(ml_df, columns=["sex", "age", "country"], drop_first=True)
X = ml_df_encoded.drop("suicides/100k pop", axis=1)
y = np.log1p(ml_df_encoded["suicides/100k pop"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
@st.cache_resource
def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Train model
dt_model = train_decision_tree(X_train, y_train)

# Evaluation for Decision Tree
y_test_pred_dt_log = dt_model.predict(X_test)
y_test_pred_dt = np.expm1(y_test_pred_dt_log)
y_test_actual = np.expm1(y_test)

r2_dt = r2_score(y_test_actual, y_test_pred_dt)
mse_dt = mean_squared_error(y_test_actual, y_test_pred_dt)
rmse_dt = np.sqrt(mse_dt)
mae_dt = mean_absolute_error(y_test_actual, y_test_pred_dt)

# Display Evaluation Metrics (Only Decision Tree)
st.subheader("📊 Decision Tree Model Evaluation Metrics")
st.markdown(f"""
- **R² Score:** `{r2_dt:.4f}`
- **Mean Squared Error (MSE):** `{mse_dt:.2f}`
- **Root Mean Squared Error (RMSE):** `{rmse_dt:.2f}`
- **Mean Absolute Error (MAE):** `{mae_dt:.2f}`
""")

# Sidebar Prediction Inputs (using Decision Tree model)
st.sidebar.header("📈 Predict Suicide Rate (Decision Tree)")
pred_year = st.sidebar.number_input("Year", min_value=1985, max_value=2045, value=2015)
pred_sex = st.sidebar.selectbox("Gender", ['male', 'female'])
pred_age = st.sidebar.selectbox("Age Group", ml_df['age'].unique())
pred_country = st.sidebar.selectbox("Prediction Country", ml_df['country'].unique())

if st.sidebar.button("Predict"):
    input_dict = {"year": [pred_year]}
    for col in X.columns:
        if col.startswith("sex_"):
            input_dict[col] = [1 if col == f"sex_{pred_sex}" else 0]
        elif col.startswith("age_"):
            input_dict[col] = [1 if col == f"age_{pred_age}" else 0]
        elif col.startswith("country_"):
            input_dict[col] = [1 if col == f"country_{pred_country}" else 0]
        elif col != "year":
            input_dict[col] = [0]
    input_data = pd.DataFrame(input_dict)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    log_pred = dt_model.predict(input_data)[0]
    final_pred = np.expm1(log_pred)
    st.sidebar.success(f"Predicted Suicide Rate: {final_pred:.2f} per 100k population")

# Raw Data Viewer
st.subheader("📄 View Raw Data")
if st.checkbox("Show raw data"):
    st.dataframe(filtered)
#Run this code using streamlit run app5.py

