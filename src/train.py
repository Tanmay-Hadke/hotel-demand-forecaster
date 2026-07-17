# src/train.py
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os

from src.data_prep import get_dataloaders
from src.model import AttentionLSTM

def load_config(config_path="configs/train_config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    # Initialize W&B
    wandb.init(project=config['project_name'], config=config)
    
    # Load Data
    print("Loading data...")
    train_loader, _, _, _ = get_dataloaders(
        config['data_path'], 
        config['seq_length'], 
        config['batch_size'], 
        config['train_split']
    )
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionLSTM(
        input_size=config['input_size'], 
        hidden_size=config['hidden_size'], 
        num_layers=config['num_layers']
    ).to(device)
    
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training Loop
    print(f"Starting training on {device}...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}/{config['epochs']} | Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"--> Saved new best model with loss {best_loss:.4f}")
            
    wandb.finish()

if __name__ == "__main__":
    main()