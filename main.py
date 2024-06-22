import argparse
import os
import pandas as pd
import imageio
import wandb
import numpy as np
import cv2
import scipy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from config.constants import data_fbp5500, batch_size, res_dir
from data import FBP5500
from model.fbp import FBP, FBC
from loss.crloss import CRLoss
from utils import load_image

def load_data(dataset_name, mode, person=None):
    transform = None
    shuffle = True
    drop_last = True
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
        drop_last = False
      
    dataset = None  
    if dataset_name == "fbp5500":
        if person is not None:
            df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"{mode}_maml.xlsx"), sheet_name=mode)
            df = df[df['user'] == person]
        else:
            df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"{mode}.xlsx"), sheet_name=mode)
        dataset = FBP5500(names=df['filename'].tolist(), scores=df['score'].tolist(), transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=50)
    return dataloader

def load_and_split_data(dataset_name, person=None):
    mode = "train"
    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1, 1, 1])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1, 1, 1])
    ])
      
    train_dataset = None  
    val_dataset = None
    if dataset_name == "fbp5500":
        if person is not None:
            df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"{mode}_maml.xlsx"), sheet_name=mode)
            df = df[df['user'] == person]
        else:
            df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"{mode}.xlsx"), sheet_name=mode)
        X_train, X_val, y_train, y_val = train_test_split(df['filename'].tolist(), df['score'].tolist(), test_size=0.15, random_state=0)
        train_dataset = FBP5500(X_train, y_train, transform=train_transform)
        val_dataset = FBP5500(X_val, y_val, transform=val_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=50)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=50)
    return train_dataloader, val_dataloader

def explain(model, dataset_name, save_name, person=None):
    model.eval()
    if not os.path.exists(os.path.join(res_dir, save_name)):
        os.makedirs(os.path.join(res_dir, save_name))
        
    image_dir = None
    if dataset_name == "fbp5500":
        image_dir = os.path.join(data_fbp5500['dir'], "faces")

    for filename in data_fbp5500['visualize_images']:
        # 获取被解释图像数据的tensor和array形式
        input_tensor, input_array = load_image(os.path.join(image_dir, f"{filename}.jpg"))
        input_tensor = input_tensor.unsqueeze(dim=0)
        input_tensor = input_tensor.to(device)
        
        for name, module in model.named_children():
            if name == 'backbone':
                for name_backbone, module_backbone in module.named_children():
                    if name_backbone != 'avgpool':
                        input_tensor = module_backbone(input_tensor)
                    else:
                        matrix = np.transpose(input_tensor[0, :, :, :].data.cpu().numpy(), [1, 2, 0])
                        matrix = np.mean(matrix, axis=2).reshape([matrix.shape[0], matrix.shape[0], 1])
                        matrix = cv2.resize(matrix, (224, 224))
                        
                        distance = np.zeros([224, 224, 3])
                        for i in range(3):
                            distance[:, :, i] = 0.2 * input_array[:, :, i] + 0.8 * matrix
                    
                        imageio.imwrite(os.path.join(res_dir, save_name, f"{filename}_gradcam.jpg"), distance.astype(np.uint8))
                        break

# def explain(model, dataset_name, save_name, person=None):
#     model.eval()
    
#     if not os.path.exists(os.path.join(res_dir, save_name)):
#         os.makedirs(os.path.join(res_dir, save_name))
    
#     target_layer = [model.backbone.avgpool]
    
#     image_dir = None
#     if dataset_name == "fbp5500":
#         image_dir = os.path.join(data_fbp5500['dir'], "faces")
    
#     df = None
#     if person is not None:
#         df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"train_maml.xlsx"), sheet_name="train")
#         df = df[df['user'] == person]
#     else:
#         df = pd.read_excel(os.path.join(data_fbp5500['dir'], f"train.xlsx"), sheet_name="train")
    
#     for filename in data_fbp5500['visualize_images']:
#         # 获取被解释图像数据的tensor和array形式
#         input_tensor, input_array = load_image(os.path.join(image_dir, f"{filename}.jpg"))
#         input_tensor = input_tensor.unsqueeze(dim=0)

#         # 构建GradCAM模型
#         model_c = FBC(model.backbone, model.classifier)
#         cam = GradCAM(model=model_c, target_layers=target_layer)
            
#         img_idx = round(df[df['filename'] == f'{filename}.jpg']['score'].tolist()[0]) - 1
#         target = [ClassifierOutputTarget(img_idx)]
        
#         # 获得热力图
#         grayscale_cam = cam(input_tensor=input_tensor, targets=target)
        
#         # 在原图上绘制热力图并保存
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = Image.fromarray(show_cam_on_image(input_array, grayscale_cam, use_rgb=True))
#         visualization.save(os.path.join(res_dir, save_name, f"{filename}_gradcam.png"))
 
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
    rmse = round(np.math.sqrt(mean_squared_error(np.array(gt_score_list), np.array(pred_score_list).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_score_list), np.array(pred_score_list).ravel())[0, 1], 4)
    print('===============The Mean Absolute Error is {0}===================='.format(mae))
    print('===============The Root Mean Square Error is {0}===================='.format(rmse))
    print('===============The Pearson Correlation is {0}===================='.format(pc))
    
    if log is not None:
        if not os.path.exists(os.path.join(res_dir, log)):
            os.makedirs(os.path.join(res_dir, log))
        with pd.ExcelWriter(os.path.join(res_dir, log, "pred_results.xlsx"), engine='openpyxl') as writer:
            col = ['filename', 'gt', 'pred']
            df = pd.DataFrame([[image_name_list[i], gt_score_list[i], pred_score_list[i][0]] for i in range(len(image_name_list))],
                            columns=col)
            df.to_excel(writer, sheet_name='result_one', index=False)
            df = pd.DataFrame({"criteria": ["MAE", "RMSE", "PC"], "value": [mae, rmse, pc]})
            df.to_excel(writer, sheet_name="result_all", index=False)
        print(f"Saved results in {res_dir}/{log}/results.xlsx")
        
    return mae
    
def train(model, train_dataloader, val_dataloader, num_epochs=25, save_name="model", weight_classifier=0.4, weight_regressor=0.6):
    cirterion = CRLoss(weight_classifier, weight_regressor)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    min_mae = 0
    recorded_mae = False
    
    print("Start training...")
    for epoch in range(num_epochs):
        model.train()
        
        loss_list = []
        for _, data in enumerate(train_dataloader, 0):
            X, y_s, y_c = data['image'], data['score'], data['class']
            X = X.to(device)
            y_s = y_s.to(device)
            y_c = y_c.to(device)
            
            optimizer.zero_grad()
            
            X = X.float()
            y_s = y_s.float().view(batch_size, 1)
            
            y_s_pred, y_c_pred = model(X)
            loss = cirterion(y_s_pred, y_s, y_c_pred, y_c)
            loss.backward()
            optimizer.step()
            
            loss_list.append(loss.item())
        
        mean_loss_train = sum(loss_list) / len(loss_list)
        wandb.log({"train loss": sum(loss_list) / len(loss_list)})
        print(f"Train loss: {mean_loss_train:.4f}")
        mae_val = test(model, val_dataloader)
        wandb.log({"val mae": mae_val})
        
        if not os.path.exists(os.path.join(res_dir, "models")):
            os.makedirs(os.path.join(res_dir, "models"))
        if not recorded_mae:
            min_mae = mae_val
            recorded_mae = True
            torch.save(model.state_dict(), os.path.join(res_dir, "models", f"{save_name}.pt"))
            print("Saved model.")
        elif mae_val < min_mae:
            min_mae = mae_val
            torch.save(model.state_dict(), os.path.join(res_dir, "models", f"{save_name}.pt"))
            print(f"Updated best model at epoch {epoch}.")
        
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("code")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--explain", action="store_true")
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--person", metavar="name", type=int, action="store", default=0)
    parser.add_argument("--dataset", metavar="file", action="store", default="fbp5500")
    parser.add_argument("--save-name", metavar="file", action="store", default="model")
    parser.add_argument("--load-from", metavar="file", action="store", default=None)
    parser.add_argument("--weight-classifier", metavar="value", type=float, action="store", default=0.4)
    parser.add_argument("--weight-regressor", metavar="value", type=float, action="store", default=0.6)

    options = parser.parse_args()
    
    model = FBP()
    if options.load_from:
        model.load_state_dict(torch.load(os.path.join(res_dir, "models", f"{options.load_from}.pt")))
    model = model.float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if options.maml:
        person = options.person
    else:
        person = None
    
    if options.train:
        wandb.init(project="cv_project_fbp", entity="yyxxyy574", name=options.save_name)
        if options.maml:
            train_dataloader, val_dataloader = load_and_split_data(options.dataset, person=person)
        else:
            train_dataloader = load_data(options.dataset, mode="train", person=person)
            val_dataloader = load_data(options.dataset, mode="test", person=person)
        train(model, train_dataloader, val_dataloader, save_name=options.save_name, weight_classifier=options.weight_classifier, weight_regressor=options.weight_regressor)
        
    if options.test:
        test_dataloader = load_data(options.dataset, mode="test", person=person)
        test(model, test_dataloader, log=options.save_name)
        
    if options.explain:
        explain(model, options.dataset, save_name=options.save_name, person=person)