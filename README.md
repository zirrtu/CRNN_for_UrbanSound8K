# CRNN_for_UrbanSound8K
A code that uses crnn to process urbansound 8K data sets

First, you have to install the dependent environment. Use the command
`pip install -r requirements to install`


Secondly, you need to download the urbansound8k dataset and put it under the project root directory. You can download it according to this link
https://www.kaggle.com/chrisfilo/urbansound8k


Then you need to run the feature extraction script
Use the command 

`python featureget.py`


After extraction, run the training script through the command 

`python CRNNtrain.py`

After the training, the weight file will be saved as mm.pt

This is the structure of the network
![image](https://github.com/zirrtu/CRNN_for_UrbanSound8K/blob/main/img/net.png)

This is the image of the accuracy of the training set and the verification set during the training process
![image](https://github.com/zirrtu/CRNN_for_UrbanSound8K/blob/main/img/acc_and_val.png)
