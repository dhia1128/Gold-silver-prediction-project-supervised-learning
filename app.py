import streamlit as st
import pandas as pd
import joblib


# Création de deux colonnes
col1, col2 = st.columns([3, 2])  # Ajustez les proportions selon vos besoins

# Affichage du titre et de la description dans la première colonne
with col1:
    st.title("Gold Price Predictor")
    st.write("""
        This app predicts the price of gold based on key economic indicators.
        Adjust the sliders to input values and click **Predict** to see the result.
    """)

# Affichage de l'image dans la deuxième colonne avec centrage
with col2:
    st.image("pepite-d-or.jpg", width=250)  # Vous pouvez augmenter cette valeur pour agrandir l'image

# Conteneur avec un lien hypertexte simple et un style amélioré
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); margin-top: 20px;'>
        <h3>Un récap historique vers les facteurs qui influencent le cours de l’or ?</h3>
        <a href='https://www.goldinfo.fr/quels-sont-les-facteurs-qui-influencent-le-cours-de-l-or/' 
           target='_blank' style='color: #007bff; text-decoration: none; font-size: 18px;'>
            Cliquez ici pour en savoir plus
        </a>
    </div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")  # Ensure the CSV is in the same directory
    return df


df = load_data()

# Define input ranges for features (based on historical data)
INPUT_RANGES = {
    "PrixArgent": {"min": 786.38, "max": 4279.79, "default": 2039.01},
    "Réserve extérieur": {"min": 65063.0, "max": 153075.0, "default": 109130.21},
    "Prix Gaz naturel": {"min": 1.08, "max": 2.64, "default": 1.78},
    "Indice des prix à la consommation": {"min": 198.1, "max": 234.85, "default": 217.63},
}

# Sidebar for project info and user inputs
st.sidebar.title("Gold Price Predictor")
st.sidebar.write("""
This app predicts the price of gold using a pre-trained machine learning model.
Adjust the sliders below to input feature values and click **Predict**.
""")

inputs = {}
for feature, values in INPUT_RANGES.items():
    inputs[feature] = st.sidebar.slider(
        label=feature,
        min_value=float(values["min"]),
        max_value=float(values["max"]),
        value=float(values["default"]),
        step=0.1
    )

# Load model and scaler
MODELS_DIR = "app.py"  # Current directory (where app.py is located)


def load_model_and_scaler():
    scaler_path = "gold_scaler.pkl"
    model_path = "gold_random_forest_model.pkl"
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


scaler, model = load_model_and_scaler()

# Prediction logic
if st.sidebar.button("Predict Gold Price"):
    try:
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame([inputs], columns=INPUT_RANGES.keys())

        # Scale the input data
        scaled_input = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Display prediction in a column
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Prediction Result")
            st.success(f"Predicted Gold Price: **{prediction:,.2f}**")

        with col2:
            st.subheader("Input Values")
            st.write(inputs)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display sample data in a column
st.subheader("Sample Data")
st.write(df.sample(5))

# Footer
st.markdown("---")
