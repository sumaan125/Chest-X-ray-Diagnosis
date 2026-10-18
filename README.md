# 🩻 Chest X-ray Diagnosis (AI Radiologist)

An AI-powered diagnostic application that analyzes chest X-ray images to classify them as COVID-19, Pneumonia (non-COVID), or Normal, providing clinical justifications based on visual features. 

[![Hugging Face Model](https://img.shields.io/badge/HuggingFace%20Model-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/Suman2004/covid_chestXray_radiologist)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace%20Dataset-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Suman2004/covid_chestXray)

## 🛠️ Tech Stack & Tools
* **Base Model:** `unsloth/Llama-3.2-11B-Vision-Instruct`.
* **Fine-Tuning Framework:** `unsloth` for highly optimized, memory-efficient vision-language model training.
* **Quantization:** `bitsandbytes` for 4-bit model loading (QLoRA).
* **Deep Learning Backend:** PyTorch (`torch`, `torchvision`).
* **Hugging Face Ecosystem:** `transformers`, `peft`, `datasets`, and `huggingface_hub`.
* **User Interface:** `gradio` web application.
* **Hardware:** NVIDIA Tesla T4 GPU (Google Colab / Kaggle environments).

## 🧠 Fine-Tuning Methodology 

### 1. 4-bit Quantization (QLoRA)
Training an 11-billion parameter vision-language model conventionally requires massive VRAM. To resolve this, the model was loaded in 4-bit precision using Unsloth's `FastVisionModel` (`load_in_4bit=True`). This drastically reduced the GPU memory footprint, enabling the entire fine-tuning process to execute on a single 16GB Tesla T4 GPU. Gradient checkpointing (`use_gradient_checkpointing="unsloth"`) was also enabled to further optimize memory for long-context vision tasks.

### 2. Parameter-Efficient Fine-Tuning (PEFT / LoRA)
Instead of updating all 11 billion parameters, Low-Rank Adaptation (LoRA) was applied to train only ~1% of the network. 
* **Target Modules:** Adapters were injected into both the vision encoder (`finetune_vision_layers=True`) and the language backbone (`finetune_language_layers=True`). Specifically, both the attention modules and MLP layers were fine-tuned. This dual-tuning approach is critical for VLMs to learn the mapping between pixel-level anomalies (radiography features) and text-level medical terminology.
* **Hyperparameters:** LoRA Rank (r) was set to 16, and LoRA Alpha to 16, ensuring an optimal balance between accuracy and overfitting. Dropout was disabled (`lora_dropout=0`).

### 3. Dataset & Instruction Formatting
The model was fine-tuned on the `Suman2004/covid_chestXray` dataset containing 1,288 structured examples of X-rays paired with expert radiologist descriptions. The instruction pipeline was strictly formatted into a chat-style template using system prompts. The model was trained to follow this specific schema:
1. **Classification:** [Condition Name]
2. **Justification:** [Clinical Findings]

## ⚡ Inference & Application Architecture

### 1. Model Loading & Generation
For inference, the custom fine-tuned weights (`Suman2004/covid_chestXray_radiologist`) are pulled from the Hugging Face Hub using `AutoModelForImageTextToText` and cast to `torch.float16` to maximize inference speed on CUDA devices. The `AutoProcessor` handles the simultaneous tokenization of the text prompt and preprocessing of the PIL image.

To improve the user experience during long clinical generations, a `TextStreamer` is implemented. This streams the diagnostic tokens sequentially as they are generated (`max_new_tokens=300`, `temperature=1.2`, `top_p=0.9`), eliminating long blocking times.

### 2. Web Interface Deployment
The prediction logic is wrapped in a `gradio` interface (`gr.Interface`). The application dynamically applies the model's `chat_template` behind the scenes, ensuring the end-user only needs to upload an image via the UI to trigger the inference pipeline. To mitigate high hosting costs for an 11B VLM (requires 15GB+ VRAM), the Gradio app is configured with `share=True` to serve the application directly from the Colab GPU environment for testing.

## 📊 Training Details
Below are the training graphs demonstrating the loss reduction during the finetuning process:
  
  ![Training Graphs](https://github.com/user-attachments/assets/0986755c-4672-4f36-820e-e2e26638234a)

Run summary logged during training:
  
  ![Run Summary](https://github.com/user-attachments/assets/7a3d62a6-9ce6-449b-a4a3-c51710029d98)

## 🖥️ User Interface
This system provides a clean Gradio user interface for immediate testing.

![Gradio UI](https://github.com/user-attachments/assets/03ef7086-1ef9-462d-bd42-99a9b461b312)


Testing the application with an input:

![Testing Input](https://github.com/user-attachments/assets/0ff10875-8aee-4f89-b69a-f9c6d30deee6)

