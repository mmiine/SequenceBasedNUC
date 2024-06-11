import torch
import argparse
from pathlib import Path
import cv2
import numpy as np
from math import log10, sqrt 
from dataset import IRContrastStretching
from dataset import IRDataset, create_splits, create_dataloader

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

# Function to test the model
def test(model, dataloader, loss_fn, device, log_dir):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_rmse_A = 0
    total_rmse_B = 0
    total_psnr = 0

    with torch.no_grad():
        for i, (inputs, labels_A, labels_B, motion_centers) in enumerate(dataloader):
            inputs, labels_A, labels_B = inputs.to(device), labels_A.to(device), labels_B.to(device)
            
            # Forward pass
            outputs_A, outputs_B = model(inputs)
            
            # Compute loss
            loss_A = loss_fn(outputs_A, labels_A)
            loss_B = loss_fn(outputs_B, labels_B)
            total_loss += (loss_A.item() + loss_B.item())
            total_rmse_A += torch.sqrt(loss_A).item()
            total_rmse_B += torch.sqrt(loss_B).item()
            
            outputs_A = outputs_A.cpu().numpy()
            outputs_B = outputs_B.cpu().numpy()
            
            A_pred, B_pred = dataloader.dataset.inverse_transform(outputs_A[0,:], outputs_B[0,:], motion_centers[0][0], motion_centers[1][0])
            A_, B_ = dataloader.dataset.crop(A_pred, B_pred, motion_centers[0][0], motion_centers[1][0])
            A_ = A_.numpy()
            B_ = B_.numpy()
            
            img = inputs.cpu().numpy()[0,0,:] * (2**14 - 1)
            corr = A_ * img + B_
            pred = A_pred * img + B_pred
            pred[pred < 0] = 0
            pred[pred > 2**14 - 1] = 2**14 - 1
            
            psnr = PSNR(corr, img)
            total_psnr += psnr
            
            if (i % 4 == 0 or i % 45 == 0 or i % 88 == 0 or i == 123):
                cv2.imwrite(f"{log_dir}/rawImgbatch_{i}.png", IRContrastStretching(img))
                cv2.imwrite(f"{log_dir}/gtImgbatch_{i}.png", IRContrastStretching(corr))
                cv2.imwrite(f"{log_dir}/predImgbatch_{i}.png", IRContrastStretching(pred))
                mean = pred.mean()
                pred[pred < 1500] = mean
                pred[pred > 13000] = mean
                cv2.imwrite(f"{log_dir}/predImgbatch_{i}_noise_mean.png", IRContrastStretching(pred))
    
    average_rmse_A = total_rmse_A / len(dataloader)
    average_rmse_B = total_rmse_B / len(dataloader)
    
    return average_rmse_A, average_rmse_B, total_psnr / len(dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model4', help='which model to use')
    parser.add_argument('--weights', type=str, default='runs/model4/exp0/weights/best.pt', help='weights path to load')
    parser.add_argument('--data', type=str, default='sequence_vs_json', help='sequence.json path')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--rename', action='store_true', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--logdir', type=str, default='test/', help='logging directory')
    
    opt = parser.parse_args()
    torch.multiprocessing.freeze_support()

    # Logging options
    log_dir = Path(opt.weights).parent.parent / 'test'
    os.makedirs(log_dir, exist_ok=True)
    results_file = str(Path(log_dir) / 'results.txt')
    
    # Load dataset
    dataset = IRDataset(opt.data)
    print(f"Dataset size: {len(dataset)}")
    
    train_sampler, val_sampler, test_sampler = create_splits()
    test_loader = create_dataloader(dataset, opt.batch_size, sampler=test_sampler)
    
    print("Dataloaders ready!")
    
    # Assuming model and dataloaders are defined
    if opt.device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)
    
    model = model_dict[opt.model](dataset.crop_size).to(device)
    state_dict = torch.load(opt.weights)
    model.load_state_dict(state_dict)
    loss_fn = nn.MSELoss()
    
    with open(results_file, 'w') as f:
        val_rmse_A, val_rmse_B, val_psnr = test(model, test_loader, loss_fn, device, log_dir)
        f.write(f"val_rmse_A: {val_rmse_A:.4f}, val_rmse_B: {val_rmse_B:.4f}, val_psnr: {val_psnr:.6f}\n")
        print(f"RMSE A: {val_rmse_A:.4f}, RMSE B: {val_rmse_B:.4f}, PSNR: {val_psnr:.6f}\n")
    torch.cuda.empty_cache()
