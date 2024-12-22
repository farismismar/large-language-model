# Large Language Model
I am doing this from scratch for the experience.

Hugging Face uses PyTorch (while TensorFlow is also supported).

The only supported (and available) generative pretrained transformer is GPT-2.  If more advanced GPTs exist, I can try to update the code.

At present, the context window cannot exceed 1024 tokens due to the architectural limit of GPT-2.

To ensure that the code runs on your NVIDIA GPU, install CUDA-compiled PyTorch using ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```.
