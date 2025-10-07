# PDS: Patten-aware Diffusion Synthesis for Multimodal Medical Imaging

PDS is a two-stage diffusion-based framework designed for high-fidelity synthesis and refinement of multimodal medical images. The pipeline first learns a joint representation via a dual-modal diffusion model, then refines outputs with task-specific guidance.

---

## 🛠️ Environment Setup

Install all required dependencies using:

```bash
pip install -r requirements.txt
💡 We recommend using a virtual environment (e.g., venv or conda) to avoid dependency conflicts.

▶️ Running the Model
PDS follows a two-stage training protocol:

1. Train the Dual-Modal Diffusion Model
Open opt.py and set:
python
opt.refine = False
Launch training:
bash
python main.py
This stage learns the core generative prior from paired multimodal data.

2. Run Refinement with Pretrained Weights
After training, locate your best checkpoint (e.g., ./checkpoints/best_model.pth).
In opt.py, update:
python
opt.resume_path = "./checkpoints/best_model.pth"  # path to your trained weights
opt.refine = True                                 # enable refinement mode
Re-run:
bash
python main.py
The model will now load the pretrained diffusion backbone and perform refinement.

3. Saving Intermediate Outputs
To visualize or debug intermediate results during training:

Modify the logging frequency in train.py (Stage 1) or train_refine.py (Stage 2):
python
save_every = 100  # save samples every N iterations
Generated images will be saved to the results/ directory by default.

4. Hyperparameter Configuration
All configurable options are centralized in opt.py, including:

Learning rate, batch size, number of diffusion timesteps
Model architecture settings (e.g., UNet depth, channel dimensions)
Input image size: Adjust opt.image_size to match your data resolution
📌 Note: Ensure opt.image_size matches your dataset. Mismatched dimensions will cause runtime errors.

📁 Project Structure
PDS/
├── main.py                 # Main entry point
├── opt.py                  # Global configuration and hyperparameters
├── train.py                # Stage 1: diffusion model training
├── train_refine.py         # Stage 2: refinement training
├── models/                 # Model definitions (diffusion + refinement)
├── checkpoints/            # Saved model weights
└── results/                # Generated images and training logs
PDS provides a flexible, modular, and reproducible pipeline for diffusion-based medical image synthesis—ideal for research in cross-modality translation, data augmentation, and generative modeling.

深色版本

你可以将以上内容复制并保存为项目根目录下的 `README.md` 文件，GitHub 会自动渲染为美观的文档页面。如需进一步定制（如添加 badges、citation、license 等），也可以在此基础上扩展。
