import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os

from dotenv import load_dotenv
from groq import Groq


# Load .env file
load_dotenv()

# Groq AI setup
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


def ask_ai(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


# Page setup
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="n",
    layout="wide"
)


st.title("AI Data Analysis Tool")


# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


if uploaded_file:

    # Read CSV
    df = pd.read_csv(uploaded_file)


    # Top metrics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )


    # Preview
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    # Automatic charts

    st.subheader("Data Visualization")


    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()


    chart_type = st.selectbox(
        "Chart type:",
        [
            "Histogram",
            "Box Plot",
            "Heatmap",
            "Scatter"
        ]
    )


    if chart_type == "Histogram" and numeric_cols:

        col = st.selectbox(
            "Column:",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    elif chart_type == "Box Plot" and numeric_cols:

        col = st.selectbox(
            "Column:",
            numeric_cols
        )

        fig = px.box(
            df,
            y=col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    elif chart_type == "Heatmap" and len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=True
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    elif chart_type == "Scatter" and len(numeric_cols) >= 2:

        x_col = st.selectbox(
            "X axis:",
            numeric_cols,
            index=0
        )

        y_col = st.selectbox(
            "Y axis:",
            numeric_cols,
            index=1
        )

        fig = px.scatter(
            df,
            x=x_col,
            y=y_col
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # AI analysis section

    st.divider()

    st.subheader("AI Auto-Analysis")


    if st.button("Generate AI Insights"):

        with st.spinner("Analyzing..."):

            # Capture df.info()

            buffer = io.StringIO()

            df.info(
                buf=buffer
            )

            info_text = buffer.getvalue()


            prompt = (
                "You are an expert data analyst.\n\n"
                "Dataset information:\n"
                + info_text
                +
                "\n\nStatistical summary:\n"
                + df.describe().to_string()
                +
                "\n\nGive:\n"
                "1. What the dataset is about\n"
                "2. Key patterns and trends\n"
                "3. Anomalies or outliers\n"
                "4. Three business recommendations\n"
                "5. Most important columns"
            )


            result = ask_ai(prompt)

            st.success(result)



    # Custom questions

    st.divider()

    st.subheader("Ask AI About Your Data")


    custom_q = st.text_area(
        "Ask a custom question:",
        placeholder="What is the average salary by department?"
    )


    if st.button("Ask AI") and custom_q:

        with st.spinner("Processing..."):

            prompt = (
                "Dataset sample:\n"
                + df.head(20).to_string()
                +
                "\n\nColumns:\n"
                + str(list(df.columns))
                +
                "\n\nQuestion:\n"
                + custom_q
            )


            answer = ask_ai(prompt)

            st.info(answer)
            