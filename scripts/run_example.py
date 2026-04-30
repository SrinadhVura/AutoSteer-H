import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor,LlavaForConditionalGeneration

class HallucinationProber(nn.Module):
    def __init__(self,input_dim=4096,hidden_dim=512):
        super(HallucinationProber,self).__init__()
        self.network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.network(x).squeeze()

def main():
    print("Loading LLaVA-1.5...")
    model_id="llava-hf/llava-1.5-7b-hf"
    processor=AutoProcessor.from_pretrained(model_id)
    model=LlavaForConditionalGeneration.from_pretrained(model_id,torch_dtype=torch.float16,device_map="auto")
    model.eval()
    device=model.device

    print("Loading Prober and Steering Vector...")
    prober=HallucinationProber().to(device)
    prober.load_state_dict(torch.load("hallucination_prober_layer_19.pth",weights_only=True))
    prober.to(torch.float16)
    prober.eval()
    steering_vector=torch.load("steering_vector_layer_19.pt",weights_only=True).to(device,torch.float16)

    THRESHOLD=0.6
    ALPHA=-1.0

    def dynamic_steering_hook(module,input,output):
        is_tuple=isinstance(output,tuple)
        hidden_states=output[0] if is_tuple else output
        current_device=hidden_states.device
        prober_device=next(prober.parameters()).device
        if hidden_states.dim()==3:
            latest_token_state=hidden_states[:,-1,:]
            with torch.no_grad():
                prob=prober(latest_token_state.to(prober_device))
            if prob.item()>THRESHOLD:
                print(f" [!] Steering triggered! (Prob: {prob.item():.2f})")
                hidden_states[:,-1,:]=latest_token_state+(ALPHA*steering_vector.to(current_device))
        elif hidden_states.dim()==2:
            latest_token_state=hidden_states[-1,:]
            with torch.no_grad():
                prob=prober(latest_token_state.to(prober_device))
            if prob.item()>THRESHOLD:
                print(f" [!] Steering triggered! (Prob: {prob.item():.2f})")
                hidden_states[-1,:]=latest_token_state+(ALPHA*steering_vector.to(current_device))
        if is_tuple:
            return (hidden_states,)+output[1:]
        else:
            return hidden_states

    target_layer_module=None
    for name,module in model.named_modules():
        if "vision" not in name and name.endswith("layers.19"):
            target_layer_module=module
            break
    if target_layer_module is None:
        raise ValueError("Could not find Layer 19 in the text model architecture!")

    image_path="coco_data/val2014/COCO_val2014_000000544456.jpg"
    try:
        image=Image.open(image_path).convert('RGB')
        print("Displaying image...")
        image.show()
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    question_text="Is there a cell phone in the image?"
    conversation=[{"role":"user","content":[{"type":"text","text":question_text},{"type":"image"}]}]
    prompt=processor.apply_chat_template(conversation,add_generation_prompt=True)
    inputs=processor(images=image,text=prompt,return_tensors="pt").to(device,torch.float16)

    print("\n========================================")
    print("RUN 1: UNSTEERED (BASELINE HALLUCINATION)")
    print("========================================")
    with torch.no_grad():
        output_ids_unsteered=model.generate(**inputs,max_new_tokens=30,use_cache=True)
    text_unsteered=processor.decode(output_ids_unsteered[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    print(f"Model Output: {text_unsteered}\n")

    print("\n========================================")
    print("RUN 2: STEERED (AUTOSTEER-H INTERVENTION)")
    print("========================================")
    hook_handle=target_layer_module.register_forward_hook(dynamic_steering_hook)
    with torch.no_grad():
        output_ids_steered=model.generate(**inputs,max_new_tokens=30,use_cache=True)
    text_steered=processor.decode(output_ids_steered[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    print(f"\nModel Output: {text_steered}")
    print("========================================")

    hook_handle.remove()

if __name__=="__main__":
    main()
