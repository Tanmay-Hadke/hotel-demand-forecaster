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
    
    # --- FEATURE ENGINEERING ---
    # Aggregate daily statistics instead of just counting rooms
    daily_stats = df.groupby('arrival_date').agg(
        total_rooms=('arrival_date', 'size'),
        avg_lead_time=('lead_time', 'mean'),
        avg_adr=('adr', 'mean'),
        cancellation_rate=('is_canceled', 'mean')
    ).reset_index()
    
    # Extract temporal features
    daily_stats['day_of_week'] = daily_stats['arrival_date'].dt.dayofweek # 0-6
    daily_stats['month'] = daily_stats['arrival_date'].dt.month # 1-12
    
    # Normalize the target variable (Rooms)
    mean_rooms = daily_stats['total_rooms'].mean()
    std_rooms = daily_stats['total_rooms'].std()
    daily_stats['normalized_rooms'] = (daily_stats['total_rooms'] - mean_rooms) / std_rooms
    
    # Scale temporal features to roughly [-1, 1]
    daily_stats['day_of_week'] = (daily_stats['day_of_week'] - 3.0) / 3.0
    daily_stats['month'] = (daily_stats['month'] - 6.5) / 5.5
    
    # Z-score normalize the new continuous features
    for col in ['avg_lead_time', 'avg_adr']:
        mean_val = daily_stats[col].mean()
        std_val = daily_stats[col].std()
        daily_stats[col] = (daily_stats[col] - mean_val) / std_val
        
    normalization_stats = {'mean': mean_rooms, 'std': std_rooms}
    
    # Stack into a multivariate array: shape (N, 6)
    # Order matters: target variable MUST remain at index 0
    features = daily_stats[[
        'normalized_rooms', 
        'day_of_week', 
        'month', 
        'avg_lead_time', 
        'avg_adr', 
        'cancellation_rate'
    ]].values
    
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