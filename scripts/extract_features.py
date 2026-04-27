import os
import json
import random
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
def main():
    print("Loading LLaVA-1.5 for Feature Extraction...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    coco_dir = "coco_data/val2014"
    pope_file = "POPE/output/coco/coco_pope_random.json"
    TARGET_LAYER = 19  # Identified layer
    image_to_questions = {}
    with open(pope_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            img_name = item['image']
            if img_name not in image_to_questions:
                image_to_questions[img_name] = {'yes': [], 'no': []}
            image_to_questions[img_name][item['label']].append(item['text'])
    valid_images = [img for img, qs in image_to_questions.items() if qs['yes'] and qs['no']]
    sample_images = random.sample(valid_images, min(500, len(valid_images))) # 500 images = 1000 samples
    def get_layer_state(image, question_text):
        prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[TARGET_LAYER][0, -1, :].cpu()
    all_features = []
    all_labels = []
    print("Extracting features from Layer 19...")
    for img_name in tqdm(sample_images):
        image_path = os.path.join(coco_dir, img_name)
        try:
            image = Image.open(image_path).convert('RGB')
            # Factual (Label 0)
            fac_q = random.choice(image_to_questions[img_name]['yes'])
            all_features.append(get_layer_state(image, fac_q))
            all_labels.append(0.0)
            # Hallucinated (Label 1)
            hal_q = random.choice(image_to_questions[img_name]['no'])
            all_features.append(get_layer_state(image, hal_q))
            all_labels.append(1.0)
            
        except Exception as e:
            continue
    # Save the dataset for downstream model -prober
    torch.save(torch.stack(all_features), "layer_19_features.pt")
    torch.save(torch.tensor(all_labels), "layer_19_labels.pt")
    print("Saved layer_19_features.pt and layer_19_labels.pt!")

if __name__ == "__main__":
    main()