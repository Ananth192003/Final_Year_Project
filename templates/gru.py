import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch.nn.functional as F
 

class AssistDataset(Dataset):
    def __init__(self, df):
        grouped = df.groupby('user_id')[['problem_id', 'correct']]
        self.data = [torch.tensor(group.values, dtype=torch.float32) for _, group in grouped]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence[:-1], sequence[1:, 1]  

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return inputs, targets


class KnowledgeTracingGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KnowledgeTracingGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)  
        out = self.fc(out) 
        out = torch.sigmoid(out)  
        return out.squeeze(-1)  


if __name__ == "__main__":
    df = pd.read_csv("assistment_2009_cleaned.csv")  
    train_loader = DataLoader(AssistDataset(df), batch_size=32, shuffle=True, collate_fn=collate_fn)

    gru_model = KnowledgeTracingGRU(2, 64, 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

    # Training
    for epoch in range(10):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = gru_model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(gru_model.state_dict(), "knowledge_tracing_gru.pth")
