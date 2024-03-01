# Session6  

### PART 2  
###### Write an NN considering the below points:    
99.4% validation accuracy  
Less than 20k Parameters  
You can use anything from above you want.   
Less than 20 Epochs  
Have used BN, Dropout,  
a Fully connected layer {or} GAP

### Major steps followed to achieve the validation accuracy of 99.4%
1. Added 1x1 after the convolution layer to reduce the num of channels
2. Added Batchnormalization for the initial layers after the convolution
3. Added dropout with a small percentage in this 20% in the initial layers.

### Model details:
Params: 16k  
Epoch num: 16  
Max validation accuracy: 99.47%  
Estimated Total Size (MB): 0.97
