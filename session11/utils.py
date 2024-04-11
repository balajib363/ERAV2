import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, PadIfNeeded, Cutout
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import matplotlib.pyplot as plt


class album_train():
    def __init__(self):
        self.transform = Compose([
            PadIfNeeded(36,36),
            RandomCrop(32,32),
            HorizontalFlip(),
            Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4471], always_apply=True, p=0.50),
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])
    def __call__(self,img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class album_test():
    def __init__(self):
        self.transform = Compose([
            Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self,img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
    
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

def plot_training_metrics(train, test):
  t = [t_items.item() for t_items in train.train_losses]
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(t)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train.train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test.test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test.test_acc)
  axs[1, 1].set_title("Test Accuracy")

def unnormalize(img):
    channel_means = (0.4914, 0.4822, 0.4471)
    channel_stdevs = (0.2469, 0.2433, 0.2615)
    img = img.numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
         img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))


def plot_grad_cam_images(model, test_loader, classes, device):
  # set model to evaluation mode
  model.eval()
  target_layers = [model.layer4[-2]]

  # Construct the CAM object once, and then re-use it on many images:
  cam = GradCAM(model=model, target_layers=target_layers)

  misclassified_images = []
  actual_labels = []
  actual_targets = []
  predicted_labels = []

  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          _, pred = torch.max(output, 1)
          for i in range(len(pred)):
              if pred[i] != target[i]:
                  actual_targets.append(target[i])
                  misclassified_images.append(data[i])
                  actual_labels.append(classes[target[i]])
                  predicted_labels.append(classes[pred[i]])

  # Plot the misclassified images
  fig = plt.figure(figsize=(12, 5))
  for i in range(10):
      sub = fig.add_subplot(2, 5, i+1)
      input_tensor = misclassified_images[i].unsqueeze(dim=0)
      targets = [ClassifierOutputTarget(actual_targets[i])]
      grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
      grayscale_cam = grayscale_cam[0, :]
      visualization = show_cam_on_image(unnormalize(misclassified_images[i].cpu()), grayscale_cam, use_rgb=True)#), image_weight=0.7)
      plt.imshow(visualization)
      sub.set_title("Actual: {}, Pred: {}".format(actual_labels[i], predicted_labels[i]),color='red')
  plt.tight_layout()
  plt.show()

