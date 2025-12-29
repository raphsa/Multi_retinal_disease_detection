import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# File paths MYDRIVE USED FOR COLAB
train_csv = "/content/drive/MyDrive/final_project_resources/train.csv"
val_csv   = "/content/drive/MyDrive/final_project_resources/val.csv"
test_csv  = "/content/drive/MyDrive/final_project_resources/offsite_test.csv"
train_image_dir ="/content/drive/MyDrive/final_project_resources/images/train"
val_image_dir = "/content/drive/MyDrive/final_project_resources/images/val"
test_image_dir = "/content/drive/MyDrive/final_project_resources/images/offsite_test"

onsite_csv = "/content/drive/MyDrive/final_project_resources/onsite_test_submission.csv"
onsite_image_dir = "/content/drive/MyDrive/final_project_resources/images/onsite_test"

saved_models_path = "/content/drive/MyDrive/final_project_resources/saved_models"
# COLAB os.makedirs(saved_models_path, exist_ok=True)
submissions_path = "/content/drive/MyDrive/final_project_resources/submissions"
# COLAB os.makedirs(submissions_path, exist_ok=True)
save_dir="/content/drive/MyDrive/final_project_resources/checkpoints"
# COLAB os.makedirs(save_dir, exist_ok=True)


# Dataset preparation
# For offsite dataset
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels

# Build model
def build_model(backbone="resnet18", num_classes=3, pretrained=True):

    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# Dataset preparation
# For onsite dataset
class OnsiteDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["id"]  # first column
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, img_name

def generate_onsite_loader():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Create dataset + dataloader
    onsite_ds = OnsiteDataset(onsite_csv, onsite_image_dir, transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=32, shuffle=False)

    return onsite_loader

# For task 1.1 and 1.2: using 0.5 as threshold for every class
def generate_onsite_result(backbone, checkpoint_path, onsite_loader, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []

    with torch.no_grad():
        for imgs, names in onsite_loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            for name, pred in zip(names, preds):
                predictions.append([name] + pred.tolist())

    df = pd.DataFrame(predictions, columns=["id", "D", "G", "A"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission file: {output_csv}")

# For task 1.3: using different lower thresholds
def generate_onsite_result_task13(backbone, checkpoint_path, onsite_loader, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    predictions = []

    thresholds = [0.4, 0.45, 0.4]

    with torch.no_grad():
        for imgs, names in onsite_loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > thresholds).astype(int)

            for name, pred in zip(names, preds):
                predictions.append([name] + pred.tolist())

    df = pd.DataFrame(predictions, columns=["id", "D", "G", "A"])
    df.to_csv(output_csv, index=False)
    print(f"Saved submission file: {output_csv}")

# Helper functions for predictions and metrics for offsite test set
def get_predictions(model, dataloader, device, thresholds=0.5):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.cpu().numpy()

            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > thresholds).astype(int)

            y_true.extend(labels)
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)

def print_disease_metrics(y_true, y_pred, disease_names, backbone=None):
    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        title = f"{disease} Results"
        if backbone is not None:
            title += f" [{backbone}]"

        print(title)
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}\n")

# TASK 1

# Task 1.1 - no training 
def train_test_task11(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints", pretrained_backbone=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)

    print("No training: only evaluating pretrained model.")

    # testing
    print("Loading pretrained backbone.")
    model.load_state_dict(torch.load(pretrained_backbone, map_location=device))
    
    disease_names = ["DR", "Glaucoma", "AMD"]

    y_true, y_pred = get_predictions(model, test_loader, device)
    print_disease_metrics(y_true, y_pred, disease_names, backbone)

    return model

# Task 1.2 - freezing backbone and training only classifier 
def train_test_task12(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints", pretrained_backbone=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)
        # freeze the backbone for task 1.2
        if backbone == "resnet18":
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        elif backbone == "efficientnet":
            for name, param in model.named_parameters():
                if "classifier.1" not in name:
                    param.requires_grad = False
        print("Backbone frozen, classifier will be trained only.")

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= 8:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # testing
    print("Loading best checkpoint from training.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    disease_names = ["DR", "Glaucoma", "AMD"]

    y_true, y_pred = get_predictions(model, test_loader, device)
    print_disease_metrics(y_true, y_pred, disease_names, backbone)

    return model

# Task 1.3 - fine tuning of full model 
def train_test_task13(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
                       epochs=10, batch_size=32, lr=1e-4, img_size=256, save_dir="checkpoints", pretrained_backbone=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # dataset & dataloader
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, train_transform)
    val_ds   = RetinaMultiLabelDataset(val_csv, val_image_dir, val_test_transform)
    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, val_test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict)

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()

    if pretrained_backbone is not None:
        if backbone == "resnet18":
          optimizer = optim.Adam([
          {"params": model.fc.parameters(), "lr": 1e-6},           # Fast for new head
          {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-4} # Slow for backbone
        ])
        elif backbone == "efficientnet":
          # optimizer = optim.Adam(model.parameters(), lr=4e-5)
          optimizer = optim.Adam([
            {"params": model.classifier.parameters(), "lr": 5e-6},   # Fast for new head
            {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": 1e-4} # Slow for backbone
        ])

    label_smoothing = 0.05

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[{backbone}] Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model for {backbone} at {ckpt_path}")
        else:
            epochs_no_improve += 1

        # early stopping
        if epochs_no_improve >= 10:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # testing
    print("Loading best checkpoint from training.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    thresholds = [0.4, 0.45, 0.4]
    disease_names = ["DR", "Glaucoma", "AMD"]

    y_true, y_pred = get_predictions(model, test_loader, device, thresholds)
    print_disease_metrics(y_true, y_pred, disease_names, backbone)
    
    return model







# main
if __name__ == "__main__":
    # Task 1.1
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'

    print("\nOffsite results with ResNet18, no training:\n")

    model_task11 = train_test_task11(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=0,
        batch_size=32,
        lr=1e-5,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task11.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task11.pt")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'

    print("\nOffsite results with EfficientNet, no training:\n")

    model_task11 = train_test_task11(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=0,
        batch_size=32,
        lr=1e-5,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task11.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task11.pt")

    generate_onsite_result(
        backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task11.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task11_resnet18.csv"
    )

    generate_onsite_result(
        backbone="efficientnet",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task11.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task11_efficientnet.csv"
    )

    # Task 1.2
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'

    print("\nOffsite results with ResNet18, training only classifier:\n")

    model_task12 = train_test_task12(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task12.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task12.pt")


    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'

    print("\nOffsite results with EfficientNet, training only classifier:\n")

    model_task12 = train_test_task12(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task12.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task12.pt")

    generate_onsite_result(
        backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task12.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task12_resnet18.csv"
    )

    generate_onsite_result(
        backbone="efficientnet",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task12.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task12_efficientnet.csv"
    )

    # Task 1.3
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'

    print("\nOffsite results with ResNet18, training full model:\n")

    model_task13 = train_test_task13(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=50,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task13.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task13.pt")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'

    print("\nOffsite results with EfficientNet, training full model:\n")

    model_task13 = train_test_task13(
        backbone,
        train_csv,
        val_csv,
        test_csv,
        train_image_dir,
        val_image_dir,
        test_image_dir,
        epochs=50,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        pretrained_backbone=pretrained_backbone)

    torch.save(model_task13.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task13.pt")

    generate_onsite_result_task13(
        backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task13.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task13_resnet18.csv"
    )

    generate_onsite_result_task13(
        backbone="efficientnet",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task13.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task13_efficientnet.csv"
    )










