import os
import json
import csv
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
"""
Peforms a grid search across the hyperparameters alpha and treshold to find
the best set of arguments 
"""
# 1. Prober Architecture
class HallucinationProber(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512):
        super(HallucinationProber, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x).squeeze()

def evaluate_pope(predictions, labels):
    """Calculates all classification metrics dynamically."""
    pred_list = []
    for text in predictions:
        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '').lower()
        words = text.split(' ')
        if 'no' in words or 'not' in words:
            pred_list.append(0)
        else:
            pred_list.append(1)

    label_list = [1 if lbl == 'yes' else 0 for lbl in labels]

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1: TP += 1
        elif pred == 1 and label == 0: FP += 1
        elif pred == 0 and label == 0: TN += 1
        elif pred == 0 and label == 1: FN += 1

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0

    return TP, FP, TN, FN, acc, precision, recall, f1, yes_ratio

def main():
    print("Loading LLaVA-1.5 for Grid Search...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    device = model.device
    print("Loading Prober and Steering Vector...")
    prober = HallucinationProber().to(device)
    prober.load_state_dict(torch.load("hallucination_prober_layer_19.pth", weights_only=True))
    prober.to(torch.float16)
    prober.eval()
    steering_vector = torch.load("steering_vector_layer_19.pt", weights_only=True).to(device, torch.float16)
    # Find Layer 19
    target_layer_module = None
    for name, module in model.named_modules():
        if "vision" not in name and name.endswith("layers.19"):
            target_layer_module = module
            break
    if target_layer_module is None:
        raise ValueError("Could not find Layer 19!")
    pope_question_file = "POPE/output/coco/coco_pope_random.json"
    coco_image_dir = "coco_data/val2014"
    with open(pope_question_file, 'r') as f:
        questions = [json.loads(line) for line in f]
    # Grid Search Parameters
    alphas = [2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    csv_file = "grid_search_results.csv"
    # Initialize CSV with Headers
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Alpha", "Threshold", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1_Score", "Yes_Ratio"])
    print(f"Starting Grid Search: {len(alphas)} Alphas x {len(thresholds)} Thresholds = {len(alphas)*len(thresholds)} combinations.")

    for threshold in thresholds:
        for alpha in alphas:
            print(f"\n--- Testing ALPHA: {alpha} | THRESHOLD: {threshold} ---")
            # Define Hook dynamically with the loop parameters
            def dynamic_steering_hook(module, input, output):
                is_tuple = isinstance(output, tuple)
                hidden_states = output[0] if is_tuple else output
                current_device = hidden_states.device
                prober_device = next(prober.parameters()).device
                if hidden_states.dim() == 3:
                    latest_token_state = hidden_states[:, -1, :]
                    with torch.no_grad():
                        prob = prober(latest_token_state.to(prober_device))
                    if prob.item() > threshold:
                        hidden_states[:, -1, :] = latest_token_state + (alpha * steering_vector.to(current_device))       
                elif hidden_states.dim() == 2:
                    latest_token_state = hidden_states[-1, :]
                    with torch.no_grad():
                        prob = prober(latest_token_state.to(prober_device))
                    if prob.item() > threshold:
                        hidden_states[-1, :] = latest_token_state + (alpha * steering_vector.to(current_device))
                return (hidden_states,) + output[1:] if is_tuple else hidden_states
            # Attach Hook
            hook_handle = target_layer_module.register_forward_hook(dynamic_steering_hook)
            predictions = []
            labels = []
            # Inference Loop
            for item in tqdm(questions, desc="Generating"):
                image_name = item['image']
                question_text = item['text']
                labels.append(item['label'])
                image_path = os.path.join(coco_image_dir, image_name)
                try:
                    image = Image.open(image_path).convert('RGB')
                except Exception:
                    predictions.append("Error")
                    continue
                prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
                with torch.no_grad():
                    output_ids = model.generate(**inputs, max_new_tokens=10, use_cache=True)
                generated_text = processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                predictions.append(generated_text)

            hook_handle.remove()
            # Calculating all the metrics required
            TP, FP, TN, FN, acc, precision, recall, f1, yes_ratio = evaluate_pope(predictions, labels)
            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([alpha, threshold, TP, FP, TN, FN, acc, precision, recall, f1, yes_ratio])

    print("\nGrid Search Complete! Check 'grid_search_results.csv' for the full matrix.")

if __name__ == "__main__":
    main()