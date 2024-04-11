# Session 11 Assignment

## Problem Statement
1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar  
2. (Optional) You are going to follow the same structure for your Code (as a reference). So Create:  
    1. models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. **Delete Bottleneck Class**  
    2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):  
        1. training and test loops  
        2. data split between test and train  
        3. epochs  
        4. batch size  
        5. which optimizer to run  
        6. do we run a scheduler?  
    3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:  
        1. image transforms,  
        2. gradcam,  
        3. misclassification code,  
        4. tensorboard related stuff  
        5. advanced training policies, etc  
        6. etc  
3. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:  
    1. pull your Github code to google colab (don't copy-paste code)  
    2. prove that you are following the above structure  
    3. that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files  
    4. your colab file must:  
        1. train resnet18 for 20 epochs on the CIFAR10 dataset  
        2. show loss curves for test and train datasets  
        3. show a gallery of 10 misclassified images  
        4. show gradcamLinks to an external site. output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬  
    5. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure.  
    6. Train for 20 epochs  
    7. Get 10 misclassified images  
    8. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)  
    9. Apply these transforms while training:  
        1. RandomCrop(32, padding=4)  
        2. CutOut(16x16)  
4. Assignment Submission Questions:

    1. Share the COMPLETE code of your model.py or the link for it  
    2. Share the COMPLETE code of your utils.py or the link for it  
    3. Share the COMPLETE code of your main.py or the link for it  
    4. Copy-paste the training log (cannot be ugly)  
    5. Copy-paste the 10/20 Misclassified Images Gallery  
    6. Copy-paste the 10/20 GradCam outputs Gallery  
    7. Share the link to your MAIN repo  
    8. Share the link to your README of Assignment  (cannot be in the MAIN Repo, but Assignment 11 repo)  


## Results
Training accuracy : 97.03%  
Validation accuracy : 91.96%  

The **training and test metrics** :    
![Training_metrics](./images/metric.png)

The **10 misclassified images** :   
![miss_classified](./images/misclassified.png)

The **Gradcam on 10 misclassified images** :    

![gradcam](./images/gradcam.png)  