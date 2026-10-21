# Gold Price Predictor

A Streamlit application that predicts the gold price (`Prix Or`) from four economic indicators using a pre-trained Random Forest regression model.

## Model performance

The saved model and scaler were evaluated against all 98 rows in `clean_data.csv` (monthly data spanning 2016–2023):

| Metric | Result |
| --- | ---: |
| Mean absolute error (MAE) | 19.41 |
| Root mean squared error (RMSE) | 29.31 |
| R-squared ($R^2$) | 0.9944 (99.44%) |

These are dataset-wide comparison scores, not verified held-out test results. The project does not include a documented test split, so the dataset may overlap with the model's training data. In particular, the $R^2$ value should not be interpreted as a guarantee of future prediction accuracy. The price units are not specified in the project data.

## Inputs

The app accepts these four features, scaled using the saved `StandardScaler` before prediction:

- `PrixArgent` (silver price)
- `Réserve extérieur` (external reserves)
- `Prix Gaz naturel` (natural gas price)
- `Indice des prix à la consommation` (consumer price index)

The saved estimator is a `RandomForestRegressor` with 100 trees. Its target is `Prix Or` (gold price).

## Run locally

Use Python 3.10 or newer, then install the dependencies and start Streamlit from the project directory:

```bash
python -m pip install streamlit pandas joblib scikit-learn==1.6.1
streamlit run app.py
```

The serialized model and scaler were created with scikit-learn 1.6.1. Using a different scikit-learn version may produce compatibility warnings or behavior changes when loading the pickle files.

## Project files

- `app.py`: Streamlit user interface and prediction code.
- `clean_data.csv`: cleaned historical data used by the app and for the metrics above.
- `gold_random_forest_model.pkl`: pre-trained Random Forest model.
- `gold_scaler.pkl`: fitted feature scaler.
- `pepite-d-or.jpg`: image displayed in the app.
