import os
import json
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration
"""
Calculates the hallucination aware score (HAS) across layers and 
outputs the best layer
"""
def main():
    print("Loading LLaVA-1.5...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    coco_dir = "coco_data/val2014"
    pope_file = "POPE/output/coco/coco_pope_random.json"
    #Parse POPE to Group the entire dataset by Image
    print("Parsing POPE dataset to dynamically pair factual/hallucinated prompts...")
    image_to_questions = {}
    with open(pope_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            img_name = item['image']
            if img_name not in image_to_questions:
                image_to_questions[img_name] = {'yes': [], 'no': []}
            # Group into factual ('yes') and hallucinated ('no') pools
            label = item['label']
            image_to_questions[img_name][label].append(item['text'])
    valid_images = [img for img, qs in image_to_questions.items() if qs['yes'] and qs['no']]
    num_samples = 10000 
    sample_images = random.sample(valid_images, min(num_samples, len(valid_images)))
    def get_hidden_states(image, question_text):
        # Format the POPE question exactly as the model expects
        prompt = f"USER: <image>\n{question_text}\nASSISTANT:"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = [layer[0, -1, :] for layer in outputs.hidden_states] # Store the required feature map from layer 19
        return torch.stack(hidden_states)
    accumulated_similarities = torch.zeros(33, device=model.device)
    successful_samples = 0
    print(f"Running dynamic HAS extraction over {len(sample_images)} paired samples...")
    # Dynamic Inference Loop
    for img_name in tqdm(sample_images):
        image_path = os.path.join(coco_dir, img_name)
        try:
            image = Image.open(image_path).convert('RGB')
            factual_question = random.choice(image_to_questions[img_name]['yes'])
            hallucinated_question = random.choice(image_to_questions[img_name]['no'])
            factual_states = get_hidden_states(image, factual_question)
            hallucinated_states = get_hidden_states(image, hallucinated_question)
            similarity = F.cosine_similarity(factual_states, hallucinated_states, dim=1)
            accumulated_similarities += similarity
            successful_samples += 1
            
        except Exception as e:
            print(f"Skipping {img_name} due to error: {e}")
            continue
    # Compute the HAS defined by formula in report and Plot
    average_similarities = (accumulated_similarities / successful_samples).cpu().numpy()
    target_layer = average_similarities.argmin()
    layers = range(len(average_similarities))
    plt.figure(figsize=(10, 6))
    plt.plot(layers, average_similarities, marker='o', linestyle='-', color='g')
    plt.title(f"Dynamic HAS Across {successful_samples} Paired Prompts (POPE)")
    plt.xlabel("Model Layer")
    plt.ylabel("Mean Cosine Similarity")
    plt.grid(True)
    plt.savefig("dynamic_has_layer_plot.png")
    print(f"\nPlot saved to 'dynamic_has_layer_plot.png'.")
    print(f"Generalized Target Layer Identified: {target_layer} (Average Lowest Similarity: {average_similarities[target_layer]:.4f})")

if __name__ == "__main__":
    main()