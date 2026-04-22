# Predict Airline Passenger using LSTM

Monthly airline passenger demand forecasting with a **stacked LSTM** pipeline: scaling, sequence windows, training with callbacks, test-set evaluation (RMSE / MAE), and multi-step future forecasts.

## Notebook

| File | Description |
|------|-------------|
| [`lstm-time-series-forecasting-predicting-passenger.ipynb`](lstm-time-series-forecasting-predicting-passenger.ipynb) | Full workflow: EDA → preprocessing → LSTM → metrics → future forecast |

## Requirements

- Python 3.9+ (3.10+ recommended)
- `numpy`, `pandas`, `matplotlib`
- `scikit-learn`
- `tensorflow` (includes Keras)

Install example:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Data (Kaggle)

If you run on [Kaggle](https://www.kaggle.com/), add the **Air Passengers** dataset and point the notebook to:

```
/kaggle/input/air-passengers/AirPassengers.csv
```

Adjust the path in the notebook if your dataset folder name differs.

## What the pipeline does

1. Load and parse monthly passenger series  
2. Exploratory plots (trend / seasonality)  
3. Time-ordered train / test split and scaling  
4. Build sliding windows (e.g. 12 months) for supervised learning  
5. Train an LSTM with dropout and training callbacks  
6. Report **RMSE** and **MAE** on the test period  
7. Recursive forecast for the next 12 months  

## License

Add a `LICENSE` file if you want to specify terms; otherwise visitors should assume all rights reserved unless you state otherwise.
