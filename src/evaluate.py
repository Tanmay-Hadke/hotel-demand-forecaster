# src/evaluate.py
import yaml
import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np

from src.data_prep import get_dataloaders
from src.model import AttentionLSTM

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    # Reconnect to W&B to log the evaluation artifact
    wandb.init(project=config['project_name'], job_type="evaluation")
    
    # Load test data and normalization stats
    _, test_loader, stats, test_data_raw = get_dataloaders(
        config['data_path'], 
        config['seq_length'], 
        config['batch_size'], 
        config['train_split']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model
    model = AttentionLSTM(
        input_size=config['input_size'], 
        hidden_size=config['hidden_size'], 
        num_layers=config['num_layers']
    ).to(device)
    
    model.load_state_dict(torch.load(config['model_save_path']))
    model.eval()
    
    predictions = []
    actuals = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).squeeze()
            
            if pred.dim() == 0:
                predictions.append(pred.item())
            else:
                predictions.extend(pred.tolist())
            
            actuals.extend(batch_y.tolist())
            
    # Inverse transform to get actual room numbers
    predictions = np.array(predictions) * stats['std'] + stats['mean']
    
    # ONLY pull the first feature (normalized_rooms) for the actuals calculation
    actuals_normalized = np.array(actuals)[:, 0] if np.array(actuals).ndim > 1 else np.array(actuals)
    actuals = actuals_normalized * stats['std'] + stats['mean']
    
    # Plotting
    plt.figure(figsize=(14, 6))
    plt.plot(actuals, label="Actual Booked Rooms", color='black', alpha=0.7)
    plt.plot(predictions, label="Model Forecast", color='blue', linestyle='--')
    plt.title("Hotel Demand Forecast: Actual vs Predicted")
    plt.xlabel("Days into Future")
    plt.ylabel("Rooms Booked")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save to disk and log to W&B
    plt.savefig("forecast_results.png")
    wandb.log({"forecast_chart": wandb.Image("forecast_results.png")})
    print("Evaluation complete. Chart logged to Weights & Biases.")
    
    wandb.finish()
    plt.show()

if __name__ == "__main__":
    main()