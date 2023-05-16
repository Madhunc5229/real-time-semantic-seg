import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from model.model import SegmentNet
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
import train

# from torchsummary import summary
torch.manual_seed(0)
LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
BACKBONE = "densenet169"
EPOCHS = 100

EXP_NAME = BACKBONE + "_" + str(time.time())
# EXP_NAME = "log_dir/densenet169_1670792529.2218642"
SAVE_PATH = os.path.join("log_dir",EXP_NAME)
def main():
    train_transform = A.Compose([
    # A.RandomSizedCrop((256 - 32, 512 - 64), 256, 512, p=1),
    # A.RandomRotate90(p=0.3),
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 

    train_loader = get_cityscapes_data(mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 4, transforms = train_transform, shuffle=True)
    val_loader = get_cityscapes_data(mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform, shuffle=True)
    test_loader = get_cityscapes_data(mode='fine', split='test', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")
    model = SegmentNet().to(device)
    print("Model loaded")
    # ckpt = torch.load("log_dir/densenet169_1670792529.2218642/densenet169_40.pt")
    # model.load_state_dict(ckpt["model_state_dict"])
    
    # optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=2)
    criterion = nn.CrossEntropyLoss(ignore_index = 255) 

    # print("############### Start Training ################")
    # train.train_model(num_epochs=EPOCHS, model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_function=criterion, scheduler=scheduler, save_path = SAVE_PATH)
    rt_vid_path = "stuttgart_01"
    # real_time_segmentation(model, device, "densenet169_40.pt", rt_vid_path, transform = val_transform)
    evaluate_(data_loader=test_loader, model=model, path="densenet169_40.pt", save_path="saved_images")
if __name__ == '__main__':
    main()