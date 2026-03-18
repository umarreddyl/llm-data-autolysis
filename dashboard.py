import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Startup Dashboard", layout="wide")

st.title("Startup Ecosystem Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -------- CLEAN --------
    df.columns = df.columns.str.strip()

    # Fix funding column
    funding_col = None
    for col in df.columns:
        if "amount" in col.lower():
            funding_col = col
            break

    if funding_col:
        df[funding_col] = df[funding_col].astype(str).str.replace(',', '')
        df[funding_col] = pd.to_numeric(df[funding_col], errors='coerce')

    # Detect useful columns
    city_col = next((c for c in df.columns if "city" in c.lower()), None)
    industry_col = next((c for c in df.columns if "industry" in c.lower()), None)

    # -------- METRICS --------
    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    total = df[funding_col].sum() if funding_col else 0
    avg = df[funding_col].mean() if funding_col else 0

    col1.metric("Total Funding", f"{total:,.0f}")
    col2.metric("Average Funding", f"{avg:,.0f}")
    col3.metric("Total Records", len(df))

    # -------- CHARTS --------
    st.subheader("Insights")

    col1, col2 = st.columns(2)

    # Funding distribution
    if funding_col:
        with col1:
            st.write("Funding Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df[funding_col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    # Industry
    if industry_col:
        with col2:
            st.write("Top Industries")
            top = df[industry_col].value_counts().head(8)
            fig, ax = plt.subplots()
            top.plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # City
    if city_col:
        st.write("Top Cities")
        top = df[city_col].value_counts().head(8)
        fig, ax = plt.subplots()
        top.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # -------- SHORT ANALYSIS --------
    st.subheader("Quick Insights")

    insights = []

    if funding_col:
        insights.append(f"- Most funding values range around {int(avg):,} USD on average.")

    if city_col:
        top_city = df[city_col].value_counts().idxmax()
        insights.append(f"- Highest number of startups are from {top_city}.")

    if industry_col:
        top_ind = df[industry_col].value_counts().idxmax()
        insights.append(f"- Most common industry is {top_ind}.")

    for i in insights:
        st.write(i)