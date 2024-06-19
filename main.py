import argparse
import os
import pandas as pd
import wandb
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.constants import data_fbp5500, batch_size, res_dir
from data import FBP5500
from model.fbp import FBP
from loss.crloss import CRLoss

def load_data(dataset_name, mode):
    transform = None
    shuffle = True
    if mode == "train":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
        ])
    elif mode == "test":
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[1, 1, 1])
        ])
        shuffle = False
      
    dataset = None  
    if dataset_name == "fbp5500":
        df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"{mode}.xlsx"), sheet_name=mode)
        dataset = FBP5500(names=df['filename'].tolist(), scores=df['score'].to_list, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=50)
 
def test(model, test_dataloader, log=None):
    model.eval()
    
    print("Start testing...")
    pred_score_list = []
    gt_score_list = []
    image_name_list = []
    for data in test_dataloader:
        images, scores, _, names = data['image'], data['score'], data['class'], data['filename']
        images = images.to(device)
        
        pred_scores, _ = model.forward(images)
        
        pred_score_list += pred_scores.to('cpu').detach().numpy().tolist()
        gt_score_list += scores.to('cpu').detach().numpy().tolist()
        image_name_list += names
    
    mae = round(mean_absolute_error(np.array(gt_score_list), np.array(pred_score_list).ravel()), 4)
    rmse = round(np.math.sqrt(mean_squared_error(np.array(gt_score_list), np.array(pred_score_list).ravel()), 4))
    pc = round(np.corrcoef(np.array()))
    print('===============The Mean Absolute Error of CRNet is {0}===================='.format(mae))
    print('===============The Root Mean Square Error of CRNet is {0}===================='.format(rmse))
    print('===============The Pearson Correlation of CRNet is {0}===================='.format(pc))
    
    if log is not None:
        col = ['filename', 'gt', 'pred']
        df = pd.DataFrame([[image_name_list[i], gt_score_list[i], pred_score_list[i][0]] for i in range(len(image_name_list))],
                          columns=col)
        df.to_excel(os.path.join(res_dir, f"{log}.xlsx"), sheet_name='result', index=False)
        print(f"Saved results in {log}.xlsx")
        
    return mae
    
def train(model, train_dataloader, val_dataloader, num_epochs=25, save_name="model"):
    cirterion = CRLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    min_mae = 0
    recorded_mae = False
    
    print("Start training...")
    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        
        loss_list = 0.0
        for _, data in enumerate(train_dataloader, 0):
            X, y_s, y_c = data['image'], data['score'], data['class']
            X = X.to(device)
            X = X.float()
            y_s = y_s.to(device)
            y_s = y_s.float().view(batch_size, 1)
            y_c = y_c.to(device)
            
            optimizer.zero_grad()
            
            y_s_pred, y_c_pred = model(X)
            loss = cirterion(y_s_pred, y_s, y_c_pred, y_c)
            loss.backward()
            optimizer.step()
            
            loss_list += loss.item()
        
        mean_loss_train = sum(loss_list) / len(loss_list)
        wandb.log({"train loss": sum(loss_list) / len(loss_list)})
        print(f"Train loss: {mean_loss_train:.4f}")
        mae_val = test(model, val_dataloader, False)
        wandb.log({"val mae": mae_val})
        
        if not recorded_mae:
            min_mae = mae_val
            recorded_mae = True
            torch.save(model.state_dict(), os.path.join(res_dir, f"{save_name}.pt"))
            print("Saved model.")
        elif mae_val < min_mae:
            min_mae = mae_val
            torch.save(model.state_dict(), os.path.join(res_dir, f"{save_name}.pt"))
            print(f"Updated best model at epoch {epoch}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("code")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--maml", action="store_true")
    parser.add_argument("--dataset", metavar="file", action="store", default="fbp5500")
    parser.add_argument("--save-name", metavar="file", action="store", default="model")
    parser.add_argument("--load-from", metavar="file", action="store", default=None)

    options = parser.parse_args()
    
    model = FBP()
    if options.load_from:
        model.load_state_dict(torch.load(os.path.join(res_dir, f"{options.load_from}.pt")))
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)