from tqdm import tqdm
from eval_test import mIOU
import torch
from torch.utils.tensorboard import SummaryWriter
import os


def train(model, device, train_loader, optimizer, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0

    for batch in tqdm(train_loader):
        image, labels, _, _ = batch
        image, labels = image.to(device), labels.to(device)

        prediction = model(image)
        optimizer.zero_grad()
        loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*image.size(0)
        running_mIOU += mIOU(labels, prediction)

    # calculate average loss
    running_loss = running_loss/len(train_loader)
    running_mIOU = running_mIOU/len(train_loader)
    return running_loss, running_mIOU



def evaluate(model, data_loader, device, loss_function):
    running_loss = 0.0
    running_mIOU = 0.0
    with torch.no_grad():
        model.eval()
        for image, labels, _, _ in data_loader:
            image, labels = image.to(device), labels.to(device)
            prediction = model(image)
            loss = 0.8*loss_function(prediction, labels)  - 0.2*mIOU(labels, prediction) 
            running_loss += loss.item()*image.size(0)
            running_mIOU += mIOU(labels, prediction)
        running_loss = running_loss/len(data_loader)
        running_mIOU = running_mIOU/len(data_loader)

    return running_loss, running_mIOU

def train_model(num_epochs, model, device, train_loader, optimizer, loss_function,  save_path, scheduler = None, val_loader = None):
    print("Start training...")

    writer = SummaryWriter(os.path.join("runs", save_path.split("/")[-1]))
    # print("Writer : ", os.path.join("runs", save_path.split("/")[-1]))
    # images, labels, _, _ = next(iter(train_loader))

    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)
    # writer.add_graph(model, images)
    model = model.to(device)
    for epoch in range(1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print("Starting Epoch "+str(epoch))
        train_loss, running_mIOU = train(model, device, train_loader, optimizer, loss_function)
        val_loss, val_mIOU = evaluate(model, val_loader, device, loss_function)
        if scheduler is not None:

            scheduler.step(val_loss)

        print('Epoch [{}/{}], Train Loss: {:.4f}, Train IOU: {:.4f}, Val Loss: {:.4f}, Val IOU: {:.4f}'.format(epoch, num_epochs, train_loss, running_mIOU, val_loss, val_mIOU))
        if epoch%10 == 0:
            save_checkpoint(save_path=save_path, model=model, optimizer=optimizer, val_loss=0, epoch=epoch)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("mIOU/Train", running_mIOU, epoch)
        writer.add_scalar("mIOU/Validation", val_mIOU, epoch)

def save_checkpoint(save_path, model, optimizer, val_loss, epoch):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    file_name = save_path.split("/")[-1].split("_")[0] + "_" + str(epoch) + ".pt"
    # print("file_name : ", file_name)
    # print("ckpt_path : ", os.path.join(save_path, file_name))
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, os.path.join(save_path, file_name))


    