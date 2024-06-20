import argparse
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from dataset import EmbeddingDataset
from transformer import Transformer

def validate_model(model, val_loader, criterion, device, epoch, save_path):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    # att = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # att.append(attention_map)

    # print(att)
    # exit()
    # np.save(os.path.join(save_path, f'att_maps_{epoch}.npy'))#, [am.cpu().numpy() for am in att])
    accuracy = correct / total
    return val_loss / len(val_loader), accuracy

def evaluate_test_set(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    auroc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    return test_loss / len(test_loader), accuracy, auroc, f1, precision, recall, cm

# def save_attention_maps(model, val_loader, save_path):
#     model.eval()
#     with torch.no_grad():
#         for inputs, _ in val_loader:
#             inputs = inputs.to(device)
#             attention_maps = model.get_attention_maps(inputs)
#             np.save(os.path.join(save_path, 'attention_maps.npy'), [am.cpu().numpy() for am in attention_maps])
#             break  # Just save for the first batch for simplicity

def main(args):
    # load csv file
    data_df = pd.read_csv(args.csv_path)
    train_df = data_df[data_df['split']=='train']
    test_df = data_df[data_df['split']=='test']
    # data = data_df['folder_id'].to_list()
    # cls_info = data_df['her2'].to_list()
    # train_df, val_df = train_test_split(data_df, random_state=cfg.seed, test_size=0.15)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_dataset = EmbeddingDataset(args.dir_path, train_df, train=True)
    test_dataset = EmbeddingDataset(args.dir_path, test_df, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('# of train dataset:', len(train_dataset), '# of test dataset', len(test_dataset))

    # Initialize the model
    model = Transformer(input_dim=768, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2.0e-05) 

    save_path = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader: #tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            test_loss, test_accuracy, test_auroc, test_f1, test_precision, test_recall, test_cm = evaluate_test_set(model, test_loader, criterion, device)
            print(f'Epoch [{epoch+1}/{args.epochs}], Train Loss: {running_loss/len(train_loader)}, test Loss: {test_loss}, test Accuracy: {test_accuracy}, test auroc: {test_auroc}, test f1: {test_f1}, test precision: {test_precision}, test recall: {test_recall}, test confusion matrix: {test_cm}')
            # save_attention_maps(model, val_loader, save_path, device)

            # Save the trained model
            model_path = os.path.join(save_path, f'ckpt_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transformer model for binary classification')
    parser.add_argument('--csv_path', type=str, default='D:/TCGA_STAD/her2_class_train_test_split.csv')
    parser.add_argument('--dir_path', type=str, default='D:/TCGA_STAD/CTransPath_feat_test')
    parser.add_argument('--save_dir', type=str, default='D:/TCGA_STAD/transformer_ckpt', help='Path to save the trained model')
    parser.add_argument('--exp', type=str, default='trial3')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2.0e-05, help='Learning rate for training')

    args = parser.parse_args()
    main(args)