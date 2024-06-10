from dataset import IRDataset, create_dataloader
from dataset import create_splits
from dataset import IRContrastStretching
from utils import Increment_dir
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import argparse
import os

from models import model_dict

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    total_loss = 0

    for batch_idx, (inputs, labels_A, labels_B, motion_centers) in enumerate(dataloader):
        inputs, labels_A, labels_B = inputs.to(device), labels_A.to(device), labels_B.to(device)

        # Forward pass
        outputs_A, outputs_B = model(inputs)

        # Compute loss
        loss_A = loss_fn(outputs_A, labels_A)
        loss_B = loss_fn(outputs_B, labels_B)
        loss = loss_A + loss_B
        total_loss += loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)

def validate(model, dataloader, loss_fn, device):
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_rmse_A = 0
        total_rmse_B = 0

        with torch.no_grad():  # Perform operations without saving gradients
            for inputs, labels_A, labels_B, motion_centers in dataloader:
                inputs, labels_A, labels_B = inputs.to(device), labels_A.to(device), labels_B.to(device)

                # Forward pass
                outputs_A, outputs_B = model(inputs)

                # Compute loss
                loss_A = loss_fn(outputs_A, labels_A)
                loss_B = loss_fn(outputs_B, labels_B)
                loss = loss_A + loss_B
                total_loss += loss.item()

                # Calculate RMSE
                rmse_A = torch.sqrt(loss_A)
                rmse_B = torch.sqrt(loss_B)
                total_rmse_A += rmse_A.item()
                total_rmse_B += rmse_B.item()

        average_loss = total_loss / len(dataloader)
        average_rmse_A = total_rmse_A / len(dataloader)
        average_rmse_B = total_rmse_B / len(dataloader)
        return average_loss, average_rmse_A, average_rmse_B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--model', type=str, default='', help='which model to use')
    parser.add_argument('--config', type=str, default='sequence_v2.json', help='sequence json path')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', type=str, default='', help='renames results.txt to results name.txt if supplied')
    parser.add_argument('--device', type=str, default='0,1,2,3 or cpu', help='cuda device, i.e. 0,1,2,3 or cpu')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    opt = parser.parse_args()

    torch.multiprocessing.freeze_support()

    # Logging options
    writer = SummaryWriter(log_dir=increment_dir(Path(opt.logdir) / opt.model / 'exp' / opt.name))
    log_dir = Path(writer.log_dir)
    wdir = str(log_dir / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = str(log_dir / 'results.txt')

    # Load dataset
    dataset = IRDataset(opt.config)
    train_dataset, val_dataset, test_dataset = create_splits(dataset)

    train_loader = create_dataloader(train_dataset, opt.batch_size, shuffle=True, num_workers=4)
    val_loader = create_dataloader(val_dataset, opt.batch_size, shuffle=False)
    test_loader = create_dataloader(test_dataset, opt.batch_size, shuffle=False)

    print("Dataloaders ready!")

    # Assuming model and dataloaders are defined
    if opt.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)

    model = model_dict[opt.model](dataset.crop_size).to(device)
    # model = SceneUNCNet2(model_dict[opt.model], dataset.crop_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    loss_fn = nn.MSELoss()

    with open(results_file, 'a') as f:
        f.write('%s, %s, %f, %d/n'%(opt.model, opt.config, opt.lr, opt.batch_size))
        f.write("Epoch, Train Loss, Val Loss, Val RMSE A, Val RMSE B\n")
        f.write("---------------------------------------------------\n")


    best_fitness = 1e10
    # Training and validation loop
    for epoch in tqdm.tqdm(range(opt.epochs)):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_rmse_A, val_rmse_B = validate(model, val_loader, loss_fn, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('RMSE/A', val_rmse_A, epoch)
        writer.add_scalar('RMSE/B', val_rmse_B, epoch)
        # Print epoch results
        #print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE A: {val_rmse_A:.4f}, Val RMSE B: {val_rmse_B:.4f}")
        with open(results_file, 'a') as f:
            f.write(f"{epoch+1}: {train_loss:.4f}, {val_loss:.4f}, {val_rmse_A:.4f}, {val_rmse_B:.4f}\n")

        if (val_rmse_A + val_rmse_B) < best_fitness:
            best_fitness = val_rmse_A + val_rmse_B

            save = (not opt.nosave) or (epoch == opt.epochs - 1)
            if save:
                # Save last, best and delete old checkpoints
                torch.save(model.state_dict(), last)

                if best_fitness == (val_rmse_A + val_rmse_B):
                    torch.save(model.state_dict(), best)
                # if near the end, save model checkpoints
                #if epoch >= (opt.epochs - 30):
                #    torch.save(model.state_dict(), last.replace('.pt', f':{epoch:03d}.pt'))

    torch.cuda.empty_cache()