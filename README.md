# Deep-Unsupervised-Pixelization with some modifications.
This model is based on [Deep-Unsupervised-Pixelization](https://github.com/csqiangwen/Deep-Unsupervised-Pixelization).
## modifications
Switching between filters, algorithms, etc. must be done by editing the code directly.
### model/networks.py
In 'class ImageGradient'  
1. Changed input to grayscale
2. Added noise reduction filter
3. Implemented smoothing filter

### model/pixelization_model.py
In 'def change_size'  
Added several proprietary algorithms

In 'def backward_all_net'  
Implemented function to save image to ./test_images

### train.py
Implemented function to save loss of generator to ./logs/loss and loss of discriminator to ./logs/D
