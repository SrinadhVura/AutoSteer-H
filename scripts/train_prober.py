import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class HallucinationProber(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512):
        super(HallucinationProber, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()     # Output between 0 and 1
        )

    def forward(self, x):
        return self.network(x).squeeze() # Squeeze removes extra dimensions

def main():
    print("Loading extracted features...")
    # Load the feature maps saved
    features = torch.load("layer_19_features.pt").to(torch.float32) # Standardize to float32 for training
    labels = torch.load("layer_19_labels.pt").to(torch.float32)
    dataset = TensorDataset(features, labels)
    # 80/20 Train-Test Split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HallucinationProber().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate Accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            predictions = model(batch_features)
            predicted_classes = (predictions > 0.5).float()
            correct += (predicted_classes == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total
    print(f"\nProber Test Accuracy: {accuracy * 100:.2f}%")

    # Save the trained weights!
    torch.save(model.state_dict(), "hallucination_prober_layer_19.pth")
    print("Model weights saved to 'hallucination_prober_layer_19.pth'")

if __name__ == "__main__":
    main()  