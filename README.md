Multi-Head MetroNet.
Crowd Counting Neural Network based on different backbones and a multi-head mechanism. 


![Alt text](Architecture/Architecture.png?raw=true "Multi-head MetroNet Architecture")


For more details please visit https://www.mdpi.com/2313-433X/6/7/62


---Usage---

1. Install the requirements in requirements.txt file (pip install -r requirements.txt)
2. Prepare the dataset, use make_dataset.py
3. In config.py you can specify the backbone and the dataset you want to use and other parameter such as learning rate, number of epochs, number of gpu and so on.
4. Run python train.py  --> this will be generate a folder in exp where you can find the best model of each run (.pth)
5. Open test.py and modify 'EXP_NAME' (Folder where you want to save results), 'DATASET_PATH' (Path of test images), 'WEIGHT_PATH.pth' (path of the model you want to test)
6. Run python test.py 


