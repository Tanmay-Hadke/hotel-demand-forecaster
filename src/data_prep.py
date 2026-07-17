import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HotelDemandDataset(Dataset):
    def __init__(self, data, seq_length=30):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # x is now shape (seq_length, 3) -> [rooms, day_of_week, month]
        x = self.data[index : index + self.seq_length]
        # y is just the target variable (normalized_rooms) for the next day, which is at index 0
        y = self.data[index + self.seq_length][0] 
        
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)

def prep_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # Stitch and sort dates
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_date_month'] + '-' + 
        df['arrival_date_day_of_month'].astype(str)
    )
    df = df.sort_values('arrival_date')
    
    daily_demand = df.groupby('arrival_date').size().reset_index(name='total_rooms')
    
    # --- FEATURE ENGINEERING ---
    # Extract temporal features
    daily_demand['day_of_week'] = daily_demand['arrival_date'].dt.dayofweek # 0-6
    daily_demand['month'] = daily_demand['arrival_date'].dt.month # 1-12
    
    # Normalize target variable
    mean_rooms = daily_demand['total_rooms'].mean()
    std_rooms = daily_demand['total_rooms'].std()
    daily_demand['normalized_rooms'] = (daily_demand['total_rooms'] - mean_rooms) / std_rooms
    
    # Min-Max scale temporal features to roughly [-1, 1] for neural network stability
    daily_demand['day_of_week'] = (daily_demand['day_of_week'] - 3.0) / 3.0
    daily_demand['month'] = (daily_demand['month'] - 6.5) / 5.5
    
    normalization_stats = {'mean': mean_rooms, 'std': std_rooms}
    
    # Stack into a multivariate array: shape (N, 3)
    features = daily_demand[['normalized_rooms', 'day_of_week', 'month']].values
    
    return features, normalization_stats

def get_dataloaders(csv_path, seq_length, batch_size, train_split=0.8):
    data, stats = prep_data(csv_path)
    
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_dataset = HotelDemandDataset(train_data, seq_length)
    test_dataset = HotelDemandDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, stats, test_data