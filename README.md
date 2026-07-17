# Hotel Booking Demand Forecasting: End-to-End MLOps Pipeline

An end-to-end machine learning platform designed to predict daily hotel room demand. This project features a custom-built PyTorch Attention-LSTM neural network to capture complex temporal seasonality, orchestrated through a fault-tolerant, serverless AWS pipeline.

## 🎯 Business Value
Accurate demand forecasting is the backbone of dynamic pricing and revenue management in the hospitality sector. By accurately predicting occupancy spikes and dips 7 to 30 days in advance, hotels can automate yield management, optimize OTA distribution, and maximize RevPAR (Revenue Per Available Room).

## 🧠 Deep Learning Architecture

Traditional time-series models (like ARIMA) often struggle with the extreme volatility of hospitality data, while standard LSTMs suffer from **mean collapse** (predicting the historical average to minimize loss). This model solves both:

* **Multivariate Temporal Inputs:** Engineered features (scaled Day of Week and Month) are concatenated with historical booking data, allowing the network to explicitly learn weekend vs. weekday seasonality.
* **Attention Mechanism:** A custom Multi-Head Attention layer applied over the LSTM outputs allows the network to dynamically assign weights to specific past days (e.g., heavily weighting *last* Tuesday to predict *next* Tuesday).
* **Huber Loss Optimization:** Swapped standard MSE for Huber Loss to prevent massive gradient penalties on outlier days (e.g., local events causing 300+ room spikes), giving the model the capacity to predict amplitude peaks accurately.
* **Experiment Tracking:** Hyperparameters and loss curves are versioned and logged via **Weights & Biases (W&B)**.

## ☁️ Cloud Infrastructure (AWS)

The model is deployed via a serverless architecture designed to bypass standard AWS Lambda deployment limits for large PyTorch binaries.

1. **AWS Step Functions (Orchestrator):** A state machine manages the workflow, allowing individual steps to fail and retry independently without re-running expensive ML inferences.
2. **Data Prep Lambda:** Queries a 30-day historical window from **DynamoDB**, normalizes the data, and extracts temporal features.
3. **PyTorch Inference Container (ECR/Lambda):** A custom, CPU-optimized Docker container pulls the `.pt` model weights from **S3**, executes the tensor forward-pass, and outputs the normalized prediction.
4. **Storage Lambda:** Applies the inverse transformation and writes the final predicted room demand back to **DynamoDB** for downstream dashboard consumption.

## 🚀 Local Execution & Training

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt