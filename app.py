import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Netflix Stock Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
    background:#0E1117;
}

h1,h2,h3,h4{
    color:white;
}

.metric{
    background:#1E293B;
    padding:15px;
    border-radius:12px;
}

.css-1d391kg{
    background:#161B22;
}

.block-container{
    padding-top:1rem;
}

</style>
""",unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/netflix_rnn.keras"
    )

model = load_model()

# ==========================================================
# LOAD SCALER
# ==========================================================

@st.cache_resource
def load_scaler():

    with open(
        "scaler/scaler.pkl",
        "rb"
    ) as f:

        scaler = pickle.load(f)

    return scaler

scaler = load_scaler()

# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv("NFLX.csv")

df = load_data()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🎬 Netflix AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📈 Prediction",

        "📄 Dataset",

        "📊 Statistics",

        "🤖 Model Info"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("SimpleRNN Model")

# ==========================================================
# TITLE
# ==========================================================

st.title("🎬 Netflix Stock Prediction using RNN")

st.caption("Deep Learning based Stock Price Forecasting")

st.markdown("---")


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    st.header("📊 Dashboard")

    current = df["Close"].iloc[-1]
    highest = df["High"].max()
    lowest = df["Low"].min()
    average = df["Close"].mean()

    # ----------------------------
    # KPI CARDS
    # ----------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Current Price",
            f"${current:.2f}"
        )

    with col2:
        st.metric(
            "📈 Highest Price",
            f"${highest:.2f}"
        )

    with col3:
        st.metric(
            "📉 Lowest Price",
            f"${lowest:.2f}"
        )

    with col4:
        st.metric(
            "📊 Average Price",
            f"${average:.2f}"
        )

    st.markdown("---")

    # ----------------------------
    # STOCK CHART
    # ----------------------------

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=df["Date"],

            y=df["Close"],

            mode="lines",

            line=dict(
                color="red",
                width=3
            ),

            name="Close Price"

        )

    )

    fig.update_layout(

        template="plotly_dark",

        title="Netflix Closing Price",

        xaxis_title="Date",

        yaxis_title="Price ($)",

        hovermode="x unified",

        height=550

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ----------------------------
    # LAST 15 RECORDS
    # ----------------------------

    st.subheader("📋 Latest Stock Records")

    st.dataframe(

        df.tail(15),

        use_container_width=True

    )

    st.markdown("---")

    st.info(
        "📌 This dashboard displays historical Netflix stock prices and key statistics."
    )


    # ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "📈 Prediction":

    st.header("📈 Predict Next Day Netflix Stock Price")

    st.markdown(
        """
        This prediction uses the **last 30 closing prices**
        as input to the trained **SimpleRNN model**.
        """
    )

    st.markdown("---")

    # Current Close Price

    current_price = df["Close"].iloc[-1]

    st.metric(
        "Today's Closing Price",
        f"${current_price:.2f}"
    )

    st.markdown("---")

    # Last 30 Days Chart

    st.subheader("Last 30 Closing Prices")

    last30 = df.tail(30)

    fig = px.line(

        last30,

        x="Date",

        y="Close",

        markers=True,

        title="Previous 30 Days"

    )

    fig.update_layout(

        template="plotly_dark",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # Prediction Button

    if st.button("🚀 Predict Next Day Price", use_container_width=True):

        with st.spinner("Predicting... Please Wait"):

            close = df["Close"].values.reshape(-1,1)

            scaled = scaler.transform(close)

            last30 = scaled[-30:]

            X = np.array(last30)

            X = X.reshape(1,30,1)

            prediction = model.predict(X)

            prediction = scaler.inverse_transform(prediction)

            predicted_price = prediction[0][0]

            difference = predicted_price - current_price

            percentage = (difference/current_price)*100

        st.success("Prediction Completed Successfully")

        st.markdown("---")

        c1,c2,c3 = st.columns(3)

        with c1:

            st.metric(

                "Current Price",

                f"${current_price:.2f}"

            )

        with c2:

            st.metric(

                "Predicted Price",

                f"${predicted_price:.2f}",

                f"{difference:.2f}"

            )

        with c3:

            st.metric(

                "Expected Change",

                f"{percentage:.2f}%"

            )

        st.markdown("---")

        # Recommendation

        if difference > 0:

            st.success(

                "📈 Model predicts the stock price may increase."

            )

        else:

            st.error(

                "📉 Model predicts the stock price may decrease."

            )

        st.markdown("---")

        # Gauge Style Progress

        progress = min(max((percentage + 10)/20,0),1)

        st.progress(progress)

        st.caption(
            f"Expected Movement : {percentage:.2f}%"
        )


        # ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📄 Dataset":

    st.header("📄 Netflix Dataset")

    st.write("Explore the dataset used to train the RNN model.")

    st.markdown("---")

    rows = st.slider(
        "Select Number of Rows",
        5,
        100,
        20
    )

    st.dataframe(
        df.tail(rows),
        use_container_width=True
    )

    st.markdown("---")

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name="NFLX.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

# ==========================================================
# STATISTICS PAGE
# ==========================================================

elif page == "📊 Statistics":

    st.header("📊 Statistical Analysis")

    st.markdown("---")

    st.subheader("Summary Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Correlation Matrix")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Closing Price Distribution")

    fig2 = px.histogram(
        df,
        x="Close",
        nbins=40,
        title="Close Price Distribution"
    )

    fig2.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Trading Volume")

    fig3 = px.line(
        df,
        x="Date",
        y="Volume",
        title="Trading Volume"
    )

    fig3.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # ==========================================================
# MODEL INFORMATION PAGE
# ==========================================================

elif page == "🤖 Model Info":

    st.header("🤖 Model Information")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
### 🧠 Model Details

- Model : SimpleRNN
- Framework : TensorFlow / Keras
- Optimizer : Adam
- Loss Function : Mean Squared Error
- Epochs : 50
- Batch Size : 32
- Time Step : 30 Days
""")

    with col2:

        st.success("""
### 📂 Dataset Information

- Dataset : Netflix Stock Price
- File : NFLX.csv
- Target : Close Price
- Feature Used : Close
- Prediction : Next Day Close Price
""")

    st.markdown("---")

    st.subheader("📌 Project Workflow")

    st.write("""
1️⃣ Load Netflix Dataset

⬇

2️⃣ Normalize Close Price using MinMaxScaler

⬇

3️⃣ Create 30-Day Sequences

⬇

4️⃣ Train SimpleRNN Model

⬇

5️⃣ Predict Next Day Stock Price

⬇

6️⃣ Display Prediction on Dashboard
""")

    st.markdown("---")

    st.subheader("🏗 RNN Architecture")

    architecture = """
Input (30 Days)
        │
        ▼
SimpleRNN (50 Units)
        │
        ▼
Dense (25 Units)
        │
        ▼
Dense (1 Unit)
        │
        ▼
Predicted Close Price
"""

    st.code(architecture)

    st.markdown("---")

    st.subheader("📚 Libraries Used")

    st.write("""
- TensorFlow
- NumPy
- Pandas
- Streamlit
- Plotly
- Scikit-learn
""")

    st.markdown("---")

    st.subheader("🎯 Project Objective")

    st.write("""
This project predicts the **next day's Netflix stock closing price**
using a **Simple Recurrent Neural Network (SimpleRNN)** trained on
historical stock market data.
""")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.success("""
Name : K.Shiva Shenker

Project : Netflix Stock Prediction using RNN

Technology :
Python • TensorFlow • Streamlit • Plotly
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center; color:gray;'>

### 📈 Netflix Stock Prediction using SimpleRNN

Built with ❤️ using Streamlit & TensorFlow

© 2026 K.Shiva Shenker

</div>
""",

unsafe_allow_html=True
)