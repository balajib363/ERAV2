# Session5
## MNIST Classification

**The project includes 3 scripts:**  
a. model.py  
b. utils.py  
c. MNIST_NN.ipynb  

**Brief Notes:**  
a. model script is used to import the neural network to solve the MNIST classification problem.  
b. utils script contains the helping modules such as check_cuda_availability, create_train_transforms, create_test_transforms, GetCorrectPredCount, train, test and plot_metrics.  
c. MNIST_NN notebook is the main files and it imports the model and utils files for all the operations to train and inference the NN trained for the classifcation problem.  

**Notes:**  
NN model size: 2.94 MB   
Number of conv layers:  4  
Loss function:  Negative Log-Likelihood (NLL) Loss  
Final Accuracy: 99.34%