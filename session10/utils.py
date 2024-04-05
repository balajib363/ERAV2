from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

class train:
    # Training model
    def __init__(self):
        self.train_losses = []
        self.train_acc    = []

    def execute(self,net, device, train_loader, optimizer, scheduler, criterion,epoch):
        net.train()
        train_loss = 0
        correct = 0
        processed = 0
        pbar = tqdm(train_loader)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            scheduler.step()

            train_loss += loss.item()

            _, predicted = outputs.max(1)
            processed += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(desc= f'Epoch: {epoch},Loss={loss.item():3.2f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)


class test:
    # testing the model
    def __init__(self):
        self.test_losses = []
        self.test_acc    = []

    def execute(self, net, device, test_loader, criterion):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))


def plot_misclassified_imgs(model_S10, test_loader, device):
  classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

