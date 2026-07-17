import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import your custom modules
from src.model import AttentionLSTM
from src.data_prep import get_dataloaders

def evaluate_model():
    # 1. Load the test data and the normalization stats
    csv_path = 'data/raw/hotel_bookings.csv'  # Update if your data is elsewhere
    seq_length = 14
    batch_size = 64
    
    print("Loading data and generating DataLoaders...")
    _, test_loader, stats, test_data = get_dataloaders(
        csv_path=csv_path, 
        seq_length=seq_length, 
        batch_size=batch_size
    )
    
    mean_rooms = stats['mean']
    std_rooms = stats['std']
    
    # 2. Load the trained model
    print("Loading trained Attention-LSTM...")
    device = torch.device('cpu') 
    model = AttentionLSTM(input_size=6, hidden_size=64).to(device)
    model.load_state_dict(torch.load('models/best_attention_lstm.pt', map_location=device))
    model.eval()
    
    actuals = []
    predictions = []
    
    # 3. Run Inference on the Test Set
    print("Running inference on test set...")
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            
            actuals.extend(y_batch.numpy().flatten())
            predictions.extend(y_pred.numpy().flatten())
            
    # 4. Inverse Transform to get actual Room Counts
    # Using your data_prep.py logic: original = (normalized * std) + mean
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    actuals_rooms = (actuals * std_rooms) + mean_rooms
    pred_rooms = (predictions * std_rooms) + mean_rooms
    
    # 5. Create the Baseline (7-Day Moving Average)
    # Reconstruct the raw room counts for the entire test split
    test_raw_rooms = (test_data[:, 0] * std_rooms) + mean_rooms
    
    df = pd.DataFrame({'actual': test_raw_rooms})
    # Predict today by averaging the previous 7 days
    df['baseline_pred'] = df['actual'].shift(1).rolling(window=7).mean()
    
    # Slice the dataframe to perfectly align with the DataLoader outputs 
    # (which skip the first seq_length days)
    df_aligned = df.iloc[seq_length:].copy().reset_index(drop=True)
    df_aligned['model_pred'] = pred_rooms
    
    df_clean = df_aligned.dropna().copy()
    
    # 6. Calculate Metrics
    baseline_mae = mean_absolute_error(df_clean['actual'], df_clean['baseline_pred'])
    model_mae = mean_absolute_error(df_clean['actual'], df_clean['model_pred'])
    
    # Using np.sqrt to avoid scikit-learn deprecation warnings for squared=False
    baseline_rmse = np.sqrt(mean_squared_error(df_clean['actual'], df_clean['baseline_pred']))
    model_rmse = np.sqrt(mean_squared_error(df_clean['actual'], df_clean['model_pred']))
    
    # 7. Calculate Percentage Improvements
    mae_improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
    rmse_improvement = ((baseline_rmse - model_rmse) / baseline_rmse) * 100
    
    # 8. Output Results for Resume
    print("\n--- EVALUATION METRICS (Test Set) ---")
    print(f"Baseline (7-Day MA) MAE:  {baseline_mae:.1f} rooms")
    print(f"Attention-LSTM MAE:       {model_mae:.1f} rooms")
    print(f"-> MAE Error Reduction:   {mae_improvement:.1f}%\n")
    
    print(f"Baseline (7-Day MA) RMSE: {baseline_rmse:.1f} rooms")
    print(f"Attention-LSTM RMSE:      {model_rmse:.1f} rooms")
    print(f"-> RMSE Error Reduction:  {rmse_improvement:.1f}%\n")
    
    print("--- RESUME BULLET DATA ---")
    print(f"Reduced forecast error by {max(mae_improvement, rmse_improvement):.0f}% vs. a moving-average baseline...")

if __name__ == '__main__':
    evaluate_model()