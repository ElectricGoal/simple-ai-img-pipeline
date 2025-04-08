## SIMPLE AI PIPELINE

### Requirements

- Nvidia GPU, VRAM at least 12 gb

### Demo
```
# Download checkpoint
wget https://civitai.com/api/download/models/351306

# Rename
mv 351306 DreamShaperXL.safetensors

# Install env
conda env create -f environment.yml

# Select env
conda activate sd

# Run
python main.py
```