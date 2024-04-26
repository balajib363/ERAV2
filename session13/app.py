import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from resnet import LitResnet, BasicBlock
import gradio as gr
import matplotlib.pyplot as plt

model = LitResnet(BasicBlock, [2, 2, 2, 2], lr=0.05)
model = LitResnet.load_from_checkpoint("weights/model_60.ckpt")

inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def resize_image_pil(image, new_width, new_height):

    # Convert to PIL image
    img = Image.fromarray(np.array(image))
    
    # Get original size
    width, height = img.size

    # Calculate scale
    width_scale = new_width / width
    height_scale = new_height / height 
    scale = min(width_scale, height_scale)

    # Resize
    resized = img.resize((int(width*scale), int(height*scale)), Image.NEAREST)
    
    # Crop to exact size
    resized = resized.crop((0, 0, new_width, new_height))

    return resized

def plot_misclassified_imgs(model_S10, test_loader, device, classes):
    model_S10.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model_S10(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(classes[target[i]])
                    predicted_labels.append(classes[pred[i]])

    # Plot the misclassified images
    fig = plt.figure(figsize=(12, 5))
    for i in range(10):
        sub = fig.add_subplot(2, 5, i+1)
        img = misclassified_images[i].cpu()
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        img = np.transpose(img, (1,2,0))
        plt.imshow(img, cmap='gray', interpolation='none')
        sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
    plt.tight_layout()
    plt.show()

def inference(input_img, gradcam_res, top_res, transparency = 0.5, target_layer_number = -1):
    input_img = resize_image_pil(input_img, 32, 32)
    
    input_img = np.array(input_img)
    org_img = input_img
    input_img = input_img.reshape((32, 32, 3))
    transform = transforms.ToTensor()
    input_img = transform(input_img)
    input_img = input_img
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    confidences = {classes[i]: float(o[i]) for i in range(10)}
    _, prediction = torch.max(outputs, 1)
    target_layers = [model.layer2[target_layer_number]]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    
    if gradcam_res:
        visualization = show_cam_on_image(org_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)
    else:
        visualization = org_img
        
    confidences_sorted = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    conf = dict(confidences_sorted[:])
    confidences_filter = {k: conf[k] for k in list(conf)[:top_res]}
    return classes[prediction[0].item()], visualization, confidences_filter

title = "CIFAR10 trained on ResNet18 Model with GradCAM"
description = "A simple Gradio interface to infer on ResNet model, and get GradCAM results"
examples = [["images/image_01.jpg", True, 3, 0.5, -1], ["images/image_06.jpg", False, 4, 0.5, -1],
["images/image_02.jpg", True, 5, 0.5, -1], ["images/image_07.jpg", False,6, 0.65, -1],
["images/image_03.jpg", True, 7, 0.5, -1], ["images/image_08.jpg", False,8, 0.5, -1],
["images/image_04.jpg", True,3, 0.5, -1], ["images/image_09.jpg", False,5, 0.5, -1],
["images/image_05.jpg", True,6, 0.77, -1], ["images/image_10.jpg", False,7, 0.5, -1]]
demo = gr.Interface(
    inference, 
    inputs = [
        gr.Image(width=256, height=256, label="Input Image"), 
        gr.Checkbox(label="GradCam", info="Check gradcam result?"),
        gr.Slider(1, 10, value = 3, step=1, label="select number of top predcitions"),
        gr.Slider(0, 1, value = 0.5, label="Overall Opacity of Image"), 
        gr.Slider(-2, -1, value = -2, step=1, label="Which Layer?")
        ], 
    outputs = [
        "text", 
        gr.Image(width=256, height=256, label="Output"),
        gr.Label()
        ],
    title = title,
    description = description,
    examples = examples,
)
demo.launch()