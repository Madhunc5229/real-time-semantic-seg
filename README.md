# real-time-semantic-segmentation


## This project deals with the development of a real-time semantic segmentation neural network.


### Dependencies

- Ubuntu 20.04 LTS or latest
- PyTorch (torchvision, tensorboard)
- PIL
- tqdm
- Sklearn (metrics)
- NumPy
- cityscapesScripts
- OpenCV

### Dataset

For training and evaluating the model, cityScapes dataset is used. It has total of 30 classes but in this project, only 19 classes will be used and the filtering operation is done using CityscapesScripts libray. 

### Model

The model chosen is an encoder decoder based architecture called U-Net. The U-Net architecture has a 4 stage encoder followed by a 4 stage upsampling decoder. The encoder used is pretrained DenseNet169 Model. There are total of 4 upsampling blocks connecting from encoder sections.

### Training the model

- To train the model, you would have to have the dataset downloaded. Please download the cityscapes dataset ('leftImg8bit_trainvaltest.zip ') from [here](https://www.cityscapes-dataset.com/downloads/). After downloading the dataset, please follow the following folder structure in order to train the model

├── data  
│   ├── gtFine  
│   └── leftImg8bit  
├── dataset  
│   └── [CityscapesDataset.py](dataset/CityscapesDataset.py)  

- Download the cityscapesscripts folder from [here](https://github.com/mcordts/cityscapesScripts) and place the `cityscapesscripts` folder in the root folder of the repository. Please note that you will get erros if you do not do this.
- Make sure you keep only the first 5 cities data and delete the rest. And use the cityscapesScript to reduce the label size from 30 to 19.
- To train the model, please uncomment the `train.train_model` line (58) in the `main.py` 
- Run the `main.py` 

### Testing the model
- To test the model, download the trained model [here](https://drive.google.com/file/d/1nND3Z6z3b2cxlS8Lcu5PlDifiPDQR1C_/view?usp=sharing)
- Place the model in the root folder with the file name `densenet169_40.pt`.
- To test the model, comment the line 58 and uncomment the `evaluate` line (61). Make sure to have reduced the label size to 19 to avoid errors

### Running the real time segmentation 
- To run real time segmentation from the model, first please download the demo video 'stuttgart_01' which is present in the folder 'leftImg8bit_demoVideo.zip' in the website, which can be downloaded from [here](https://www.cityscapes-dataset.com/downloads/)
- After download the folder, copy the folder 'stuttgart_01' and place in the root folder.
- Comment both train and evlauate lines and uncomment the `real_time_segmentation` line (60)
- Run the `main.py` file

### Results
- Please follow this link to view the output of the model: [output](https://www.youtube.com/watch?v=7ZoiyrH4lMw)
- https://www.youtube.com/watch?v=7ZoiyrH4lMw
