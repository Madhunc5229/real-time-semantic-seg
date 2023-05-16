import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
from dataset.CityscapesDataset import CityscapesDataset
from utils.utils import *
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import time
from cityscapesscripts.helpers.labels import trainId2label as t2l

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

colormap = np.zeros((19, 3), dtype=np.uint8)

colormap[0] = [128, 64, 128]
colormap[1] = [244, 35, 232]
colormap[2] = [70, 70, 70]
colormap[3] = [102, 102, 156]
colormap[4] = [190, 153, 153]
colormap[5] = [153, 153, 153]
colormap[6] = [250, 170, 30]
colormap[7] = [220, 220, 0]
colormap[8] = [107, 142, 35]
colormap[9] = [152, 251, 152]
colormap[10] = [70, 130, 180]
colormap[11] = [220, 20, 60]
colormap[12] = [255, 0, 0]
colormap[13] = [0, 0, 142]
colormap[14] = [0, 0, 70]
colormap[15] = [0, 60, 100]
colormap[16] = [0, 80, 100]
colormap[17] = [0, 0, 230]
colormap[18] = [119, 11, 32]

def get_cityscapes_data(
    mode,
    split,
    root_dir='data',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,
    num_workers=2

):
    data = CityscapesDataset(
        mode=mode, split=split, target_type=target_type,transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    return data_loaded

def decode_segmap(temp):
    temp = temp.cpu().numpy()[0]
    #convert gray scale to color
    # temp=temp.numpy()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 19):
        r[temp == l] = colormap[l][0]
        g[temp == l] = colormap[l][1]
        b[temp == l] = colormap[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    rgb  = cv2.cvtColor(rgb.astype('float32'), cv2.COLOR_RGB2BGR) 
    return rgb

def save_predictions(data, model, save_path):    
    model.eval()
    model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(data)):

            X, y, s, _ = batch
            X, y = X.to(device), y.to(device)
            predictions = model(X) 
            predictions = torch.nn.functional.softmax(predictions, dim=1)
            pred_labels = torch.argmax(predictions, dim=1) 
            pred_labels = pred_labels.float()

            pred_labels = pred_labels.to('cpu')
            pred_labels.apply_(lambda x: t2l[x].id)
            pred_labels = pred_labels.to(device)

            pred_labels = transforms.Resize((1024, 2048))(pred_labels) 
            s = str(s)

            pos = s.rfind('/', 0, len(s))
            name = s[pos+1:-18]    

            # global location
            # location = 'saved_images\multiclass_1'

            save_as_images(pred_labels, save_path, name)                


def save_as_images(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}/{image_name}.png"
    tensor_pred.save(filename)

def evaluate_(data_loader, model, path, save_path):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'{path} has been loaded and initialized')
    save_predictions(data_loader, model, save_path)

###########################################################################################################
def get_tensor(path, transform = None):
    frame = Image.open(path)
    if transform is not None:
            og_img = np.array(frame)
            transformed = transform(image = og_img)
            image = transformed["image"]
            image = image.unsqueeze(0)
            return image, og_img
    return None, None

def get_labels(predictions):
    predictions = torch.nn.functional.softmax(predictions, dim=1)
    pred_labels = torch.argmax(predictions, dim=1) 
    pred_labels = pred_labels.float()
    return pred_labels

def real_time_segmentation(model, device, weight_path, video_path, transform = None):

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    s = str(video_path)
    pos = s.rfind('/', 0, len(s))
    vid_name = s[pos+1:]  
    video_output = cv2.VideoWriter('test_results/'+vid_name+"_1"+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (2048, 1024))

    model.eval()
    model.to(device)
    alpha = 0.5
    for frame_name in tqdm(sorted(os.listdir(video_path))):
        new_frame, og_frame = get_tensor(video_path+'/'+frame_name, transform)
        start_time = time.time()
        segmentations = rt_predict(model, new_frame, transform)
        end_time = time.time()
        segmentations = cv2.normalize(segmentations, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) 
        blend = cv2.addWeighted(segmentations, alpha,  og_frame, abs(1-alpha), 0, og_frame)
        time_taken = end_time - start_time
        fps = round(1/time_taken, 2)

        cv2.putText(blend, (str(fps)+'FPS'), (2048-500, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("img", blend)
        cv2.waitKey(25) 
        cv2.destroyAllWindows()
        video_output.write(blend)
    video_output.release()
    
def rt_predict(model, image, transform):
    with torch.no_grad():
        image = image.to(device)
        predictions = model(image) 
        pred_labels = get_labels(predictions)
        pred_labels = transforms.Resize((1024, 2048))(pred_labels)
        color_labels = decode_segmap(pred_labels)
        return color_labels