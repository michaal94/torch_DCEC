import numpy as np

def tensor2img(tensor):
    img = tensor.cpu().data[0].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')