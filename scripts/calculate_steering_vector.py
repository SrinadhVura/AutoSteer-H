import torch
"""
Calculates a simple global steering vector as difference between
the mean of feature maps at layer 19 of all the fatual samples and 
mean of feature maps at layer 19 of all the hallucinated samples
"""
def main():
    print("Loading Layer 19 features and labels...")
    features = torch.load("layer_19_features.pt")
    labels = torch.load("layer_19_labels.pt")
    # Separate features based on labels
    # Label 0.0 is Factual, Label 1.0 is Hallucinated
    factual_features = features[labels == 0.0]
    hallucinated_features = features[labels == 1.0]
    print(f"Found {len(factual_features)} factual samples and {len(hallucinated_features)} hallucinated samples.")
    # Calculate the mean vector for both classes
    mean_factual = torch.mean(factual_features, dim=0)
    mean_hallucinated = torch.mean(hallucinated_features, dim=0)
    # Calculate the steering vector
    steering_vector = mean_factual - mean_hallucinated
    # Save the vector for inference
    torch.save(steering_vector, "steering_vector_layer_19.pt")
    print("Success! Steering vector saved to 'steering_vector_layer_19.pt'.")

if __name__ == "__main__":
    main()