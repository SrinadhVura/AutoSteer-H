import os
import json
import random
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
"""
Performs the generation of captions through chosen decoding strategy 
Runs on a randomly sampled subset of 500 samples from MS-COCO dataset
"""
DECODING_STRATEGY = "nucleus"  # Greedy/ Nucleus/ Beam search
ALPHA = -1.0 
THRESHOLD = 0.6

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
    output_file = f"chair_captions_alpha_{ALPHA}_{THRESHOLD}_{DECODING_STRATEGY}.jsonl"
    print(f"Starting CHAIR Generation: {DECODING_STRATEGY.upper()} run with ALPHA={ALPHA}")

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
                prob = prober(latest_token_state.to(prober_device)).view(-1) 
            mask = (prob > THRESHOLD).to(current_device).to(torch.float16).view(-1, 1)
            hidden_states[:, -1, :] = latest_token_state + (ALPHA * mask * steering_vector.to(current_device))
                
        elif hidden_states.dim() == 2:
            latest_token_state = hidden_states[-1, :]
            with torch.no_grad():
                prob = prober(latest_token_state.to(prober_device))
            if prob.item() > THRESHOLD:
                hidden_states[-1, :] = latest_token_state + (ALPHA * steering_vector.to(current_device))

        return (hidden_states,) + output[1:] if is_tuple else hidden_states

    target_layer_module = None
    if ALPHA != 0.0:
        for name, module in model.named_modules():
            if "vision" not in name and name.endswith("layers.19"):
                target_layer_module = module
                break
        if target_layer_module is None:
            raise ValueError("Could not find Layer 19!")
        hook_handle = target_layer_module.register_forward_hook(dynamic_steering_hook)
    # 1. Select 500 Random Images from COCO val2014 for CHAIR evaluation
    coco_image_dir = "coco_data/val2014"
    all_images = [f for f in os.listdir(coco_image_dir) if f.endswith('.jpg')] 
    # Set seed so we evaluate the exact same 500 images across different runs
    random.seed(42) 
    sample_images = random.sample(all_images, 500)
    results = []
    for image_name in tqdm(sample_images, desc="Generating Captions"):
        # Extract the integer ID from the COCO filename (e.g., COCO_val2014_000000391895.jpg -> 391895)
        image_id = int(image_name.split('_')[-1].split('.')[0])
        image_path = os.path.join(coco_image_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception:
            continue
        # Generative prompt
        prompt = "USER: <image>\nPlease describe this image in detail.\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            if DECODING_STRATEGY == "greedy":
                output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True)
            elif DECODING_STRATEGY == "beam":
                output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_beams=5, use_cache=True)
            elif DECODING_STRATEGY == "nucleus":
                output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.9, use_cache=True)
        generated_text = processor.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        # Save exactly how CHAIR expects it
        results.append({
            "image_id": image_id,
            "caption": generated_text
        })
    with open(output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    if ALPHA != 0.0:
        hook_handle.remove()
    print(f"CHAIR Generation complete! Saved to {output_file}")

if __name__ == "__main__":
    main()