## Scripts Functionality

* **`calculate_has.py`**: Calculates the Hallucination Awareness Score (HAS) across model layers by comparing cosine similarities of factual vs. hallucinated activations. Used to identify the optimal intervention layer.
* **`extract_features.py`**: Runs the VLM on POPE dataset prompts to extract and save hidden state tensors from the targeted layer (Layer 19).
* **`train_prober.py`**: Trains a lightweight 2-layer Multi-Layer Perceptron (MLP) binary classifier on the extracted features to detect hallucinations.
* **`calculate_steering_vector.py`**: Computes the textual/visual direction by finding the vector difference between stable factual embeddings and unstable hallucinated embeddings.
* **`dynamic_inference.py`**: The core integration script. Utilizes PyTorch Forward Hooks to pause the model at Layer 19 during generation, pass hidden states to the prober, and dynamically inject the steering vector if a hallucination is predicted.
* **`run_pope_steered.py`**: Executes the steered LLaVA model on the POPE benchmark to measure intervention efficacy.
* **`run_pope_decoding.py`**: Evaluates the model on the POPE benchmark using different decoding strategies (greedy, beam, nucleus) alongside the auto-correction mechanism.
* **`run_chair_generation.py`**: Generates image descriptions for the CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation using MS-COCO images. 
* **`run_eval.py`**: Calculates classification metrics (TP, FP, TN, FN, Accuracy, Precision, Recall, F1) for the POPE benchmark results.
* **`grid_search.py`**: Conducts a hyperparameter search to find the optimal steering intensity ($\alpha$) and prober threshold ($\tau$).
* **`run_example.py`**: Runs the unsteered model and steered model on one of the example to show how AutoSteer-H steered the model away from hallucination

---

## Steps to reproduce the optimal results obtained

The optimal pipeline  was achieved as (Accuracy: 0.901, F1: 0.893 on POPE Greedy). 

#### Phase 1: Feature Extraction & Prober Training
To determine the target layer and train the inference-time intervention components.

```bash
# 1. Identify the target layer (in our setup it turned out to be to be Layer 19)
python calculate_has.py

# 2. Extracts Layer 19 hidden states for factual/hallucinated data as stores them in .pt files
python extract_features.py

# 3. Train the Neural Network prober (achieved ~93.5% test accuracy)
python train_prober.py

# 4. Calculate the global steering vector
python calculate_steering_vector.py
```

#### Phase 2: Intervened Model Inference & POPE Evaluation
Run the auto-corrected model on the POPE dataset. The optimal parameters discovered were a steering intensity of `alpha=-1.0` and a prober threshold of `0.6` to get a strictly safe model. And a steering intensity of `alpha=-2.0` and prober threshold of `0.8` for an overall better model performance.

```bash
# 1. Run dynamic inference with Greedy decoding
python run_pope_decoding.py

# 2. Evaluate the generated outputs
python run_eval.py
```
***Note**: The paths in the begining of scripts have to be changed to point to the correct files and directories - The main paths to be checked are the factual and hallucinated feature maps stored, steering vector*

***Note**: To test alternative decoding strategies, modify the decoding parameters in `run_pope_decoding.py` and execute the script for beam or nucleus generation.*

#### Phase 3: CHAIR Evaluation
Generate captions for the CHAIR metric using MS-COCO images.
```bash
# 1. Generate captions using the steered model (e.g., Greedy)
python run_chair_generation.py
```

Run the command `python grid_search.py` to perform grid search on the hyperparameters `alpha` and prober threshold

