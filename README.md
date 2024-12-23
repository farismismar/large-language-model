# Large Language Model
I am doing this from scratch for the experience.

Hugging Face uses PyTorch (while TensorFlow is also supported).

The only supported (and available) generative pretrained transformer is GPT-2.  If more advanced GPTs exist, I can try to update the code.  There are two modes: scratch and pre-trained using the Hugging Face pre-trained GPT-2 for less hallucinations.

At present, the context window cannot exceed 1024 tokens due to the architectural limit of GPT-2.  Further, I define a few stop tokens to prevent the output from always having to match the context window size.

To ensure that the code runs on your NVIDIA GPU, install CUDA-compiled PyTorch using ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```.
