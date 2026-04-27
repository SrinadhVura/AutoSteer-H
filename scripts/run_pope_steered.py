import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. Define the Prober
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

def main():
    # Hyperparams
    THRESHOLD = 0.5
    ALPHA = -1.0 
    output_file = f"llava_1_5_pope_steered_alpha_{ALPHA}.jsonl"
    print("Loading LLaVA-1.5...")
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
    def dynamic_steering_hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        current_device = hidden_states.device
        prober_device = next(prober.parameters()).device
        
        if hidden_states.dim() == 3:
            latest_token_state = hidden_states[:, -1, :]
            with torch.no_grad():
                prob = prober(latest_token_state.to(prober_device))
            if prob.item() > THRESHOLD:
                hidden_states[:, -1, :] = latest_token_state + (ALPHA * steering_vector.to(current_device))
                
        elif hidden_states.dim() == 2:
            latest_token_state = hidden_states[-1, :]
            with torch.no_grad():
                prob = prober(latest_token_state.to(prober_device))
            if prob.item() > THRESHOLD:
                hidden_states[-1, :] = latest_token_state + (ALPHA * steering_vector.to(current_device))

        return (hidden_states,) + output[1:] if is_tuple else hidden_states
    target_layer_module = None
    for name, module in model.named_modules():
        if "vision" not in name and name.endswith("layers.19"):
            target_layer_module = module
            break
    if target_layer_module is None:
        raise ValueError("Could not find Layer 19!")
    hook_handle = target_layer_module.register_forward_hook(dynamic_steering_hook)

    pope_question_file = "POPE/output/coco/coco_pope_random.json"
    coco_image_dir = "coco_data/val2014"
    print("Loading POPE questions...")
    with open(pope_question_file, 'r') as f:
        questions = [json.loads(line) for line in f]

    results = []

    print(f"Starting steered inference (ALPHA={ALPHA})...")
    for item in tqdm(questions):
        image_name = item['image']
        question_text = item['text']
        question_id = item['question_id']

        image_path = os.path.join(coco_image_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            continue

        prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=10, use_cache=True)
        
        generated_text = processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

        results.append({
            "question_id": question_id,
            "image": image_name,
            "text": question_text,
            "answer": generated_text
        })

    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    hook_handle.remove()
    print(f"Evaluation complete! Saved to {output_file}")

if __name__ == "__main__":
    main()