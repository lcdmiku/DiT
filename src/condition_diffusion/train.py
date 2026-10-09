import torch
from src.condition_diffusion.ddpm import DDPM
import tqdm
from torch.utils.data.dataloader import DataLoader

def train(
        model:DDPM,
        optimizer:torch.optim.Optimizer,
        epochs:int,
        device:str,
        train_loader:DataLoader,
        val_loader:DataLoader,
        save_path='cond_model.pth',
        save_interval:int=10,
):
    train_losses = []
    val_losses = []
    time = 0
    for epoch in range(epochs):
        model.train(True)
        train_loss = 0
        val_loss = 0
        pbar = tqdm.tqdm(train_loader)
        for index,(img,label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            loss = model(img,label)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            time += 1
            if time % save_interval == 0:
                torch.save(model.state_dict(), save_path)
            pbar.set_description(f"loss for epoch {epoch}: {train_loss / (index + 1):.4f}")

        # model.eval()
        # with torch.no_grad():
        #     for (imgs, labels) in val_loader:
        #         imgs = imgs.to(device)
        #
        #         loss = model(imgs)
        #
        #         val_loss += loss.item()
        # train_losses.append(train_loss / len(train_loader))
        # val_losses.append(val_loss / len(val_loader))
    return train_losses, val_losses
