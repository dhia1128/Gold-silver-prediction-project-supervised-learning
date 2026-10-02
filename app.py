import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fa;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: #192332;
    }
    .eyebrow {
        color: #a66b12;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero-title {
        color: #192332;
        font-size: clamp(2.3rem, 5vw, 3.6rem);
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.05;
        margin: 0 0 1rem;
    }
    .hero-copy {
        color: #586273;
        font-size: 1.08rem;
        line-height: 1.7;
        max-width: 38rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e8ebef;
        border-radius: 14px;
        padding: 1rem 1.15rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #667085;
    }
    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #b7791f;
        border: 1px solid #b7791f;
        border-radius: 10px;
        color: #ffffff;
        font-weight: 650;
        min-height: 2.8rem;
        transition: background 150ms ease, border-color 150ms ease;
    }
    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #965f14;
        border-color: #965f14;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv("clean_data.csv")


@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load("gold_scaler.pkl")
    model = joblib.load("gold_random_forest_model.pkl")
    return scaler, model


INPUT_RANGES = {
    "PrixArgent": {
        "label": "Silver price",
        "min": 786.38,
        "max": 4279.79,
        "default": 2039.01,
        "help": "Historical silver price used as a model input.",
    },
    "Réserve extérieur": {
        "label": "Foreign reserves",
        "min": 65063.0,
        "max": 153075.0,
        "default": 109130.21,
        "help": "Historical external reserves indicator.",
    },
    "Prix Gaz naturel": {
        "label": "Natural gas price",
        "min": 1.08,
        "max": 2.64,
        "default": 1.78,
        "help": "Historical natural gas price indicator.",
    },
    "Indice des prix à la consommation": {
        "label": "Consumer price index",
        "min": 198.1,
        "max": 234.85,
        "default": 217.63,
        "help": "Historical consumer price index.",
    },
}

df = load_data()

hero_copy, hero_image = st.columns([1.45, 1], gap="large")
with hero_copy:
    st.markdown(
        """
        <div class="eyebrow">Market insights · Gold</div>
        <div class="hero-title">A clearer view of gold.</div>
        <div class="hero-copy">
            Explore how key economic indicators shape a gold price estimate.
            Set the assumptions below to generate a prediction from the
            project's trained Random Forest model.
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_image:
    st.image("pepite-d-or.jpg", use_container_width=True)

st.write("")
metric_columns = st.columns(3)
date_range = f"{int(df['year'].min())}–{int(df['year'].max())}"
with metric_columns[0]:
    st.metric("Historical observations", f"{len(df):,}")
with metric_columns[1]:
    st.metric("Economic indicators", len(INPUT_RANGES))
with metric_columns[2]:
    st.metric("Data period", date_range)

st.divider()
st.subheader("Build your prediction")
st.caption("Adjust each indicator within its historical range, then run the estimate.")

with st.form("prediction_form"):
    inputs = {}
    input_columns = st.columns(2, gap="large")
    for index, (feature, values) in enumerate(INPUT_RANGES.items()):
        with input_columns[index % 2]:
            inputs[feature] = st.slider(
                label=values["label"],
                min_value=float(values["min"]),
                max_value=float(values["max"]),
                value=float(values["default"]),
                step=0.1,
                help=values["help"],
            )
    submitted = st.form_submit_button("Generate gold price estimate")

if submitted:
    try:
        scaler, model = load_model_and_scaler()
        input_data = pd.DataFrame([inputs], columns=INPUT_RANGES.keys())
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]

        st.divider()
        result_column, details_column = st.columns([1, 1.4], gap="large")
        with result_column:
            st.markdown('<div class="eyebrow">Your estimate</div>', unsafe_allow_html=True)
            st.metric("Predicted gold price", f"{prediction:,.2f}")
            st.caption("Estimate generated by the project's pre-trained model.")
        with details_column:
            st.subheader("Inputs used")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Indicator": INPUT_RANGES[feature]["label"],
                            "Value": value,
                        }
                        for feature, value in inputs.items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
    except Exception as error:
        st.error(f"Unable to generate a prediction: {error}")

st.divider()
with st.expander("Explore the historical data"):
    st.caption("Preview of the cleaned dataset used by this application.")
    st.dataframe(df.head(10), hide_index=True, use_container_width=True)

st.sidebar.title("About this model")
st.sidebar.write(
    "A Random Forest model estimates gold prices from four economic indicators. "
    "Predictions are exploratory and should not be treated as financial advice."
)
st.sidebar.markdown(
    "[Learn about factors that influence gold prices](https://www.goldinfo.fr/"
    "quels-sont-les-facteurs-qui-influencent-le-cours-de-l-or/)"
)
