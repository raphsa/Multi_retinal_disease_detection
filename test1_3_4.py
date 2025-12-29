import os
import pandas as pd
import numpy as np
import shutil
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from torchvision.models import ResNet18_Weights, EfficientNet_B0_Weights
from torchvision.models import ResNet, efficientnet_b0
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


# TASK 2

# Functions for task 2
# ...













# TASK 3

# Functions for task 3
# SE Block for task 3.1
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

class SEBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(SEBasicBlock, self).__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out) # SE attention

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# SE block for EfficientNet
def get_out_channels(module):
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            return m.out_channels
    raise RuntimeError("No Conv2d found in block")

class SEEfficientNet(nn.Module):
    def __init__(self, num_classes=3, reduction=16, pretrained=True):
        super().__init__()

        base = efficientnet_b0(
            weights="DEFAULT" if pretrained else None
        )

        self.features = base.features
        self.avgpool = base.avgpool

        self.se_blocks = nn.ModuleList([
            SEBlock(get_out_channels(block), reduction)
            for block in self.features
        ])

        self.classifier = nn.Linear(
            base.classifier[1].in_features,
            num_classes
        )

    def forward(self, x):
        for block, se in zip(self.features, self.se_blocks):
            x = block(x)
            x = se(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Build model with SE attention for ResNet
def build_model_task31(num_classes=3):

    model = ResNet(
        block=SEBasicBlock,
        layers=[2, 2, 2, 2]
    )

    # Replace classifier head for multi-label classification
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

# Build model with SE attention for EfficientNet
def build_model_task31_efficientnet(num_classes=3, pretrained=True):
    return SEEfficientNet(
        num_classes=num_classes,
        reduction=16,
        pretrained=pretrained
    )

# Onsite predictions for ResNet18
def generate_onsite_result_task31(checkpoint_path, onsite_loader, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build SE-ResNet18 
    model = build_model_task31(num_classes=3).to(device)

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

# Onsite predictions for EfficientNet
def generate_onsite_result_task31_effnet(checkpoint_path, onsite_loader, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build SE-efficientNet 
    model = build_model_task31_efficientnet(num_classes=3).to(device)

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


# Task 3.1 - Squeeze-and-Excitation (SE) for both ResNet18 and EfficientNet
def train_test_task31(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
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
    if backbone == "efficientnet":
        model = build_model_task31_efficientnet(num_classes=3).to(device)
    else:
        model = build_model_task31(num_classes=3).to(device)


    for p in model.parameters():
        p.requires_grad = True

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

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
    epochs_no_improve = 0

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

# ResNet18 with MHA for task 3.2
class ResNet18_MHA(nn.Module):
    def __init__(self, backbone, num_classes=3, num_heads=4):
        super().__init__()

        self.backbone = backbone

        self.mha = nn.MultiheadAttention(512, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(512)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 512))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = self.backbone(x)            
        B, C, H, W = x.shape

        # Spatial tokens
        x = x.view(B, C, H * W).permute(0, 2, 1)  
        x = x + self.pos_embed
        x_norm = self.norm(x)

        # Multi-Head Self-Attention
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pooling and classification
        x = x.mean(dim=1)
        x = self.classifier(x)

        return x

# EfficientNet with MHA for task 3.2
class EfficientNet_MHA(nn.Module):
    def __init__(self, num_classes=3, num_heads=8, pretrained=True):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = efficientnet_b0(weights=weights)

        self.features = base.features
        self.avgpool = base.avgpool

        embed_dim = 1280   

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim))

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # CNN backbone
        x = self.features(x)             
        B, C, H, W = x.shape

        # Spatial tokens
        x = x.view(B, C, H * W).permute(0, 2, 1)  
        x = x + self.pos_embed
        x_norm = self.norm(x)

        # Self-attention
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pool and classifier
        x = x.mean(dim=1)
        return self.classifier(x)

# Build model for Task 3.2
def build_model_task32(backbone="resnet18",
                       num_classes=3,
                       num_heads=4,
                       pretrained=True):

    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        backbone_net = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        return ResNet18_MHA(
            backbone=backbone_net,
            num_classes=num_classes,
            num_heads=num_heads
        )

    elif backbone == "efficientnet":
        return EfficientNet_MHA(
            num_classes=num_classes,
            num_heads=8,   # MUST divide 1280
            pretrained=pretrained
        )

    else:
        raise ValueError("Unsupported backbone")

# Onsite predictions 
def generate_onsite_result_task32(backbone, checkpoint_path, onsite_loader, output_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build MHA-model
    model = build_model_task32(
        backbone=backbone,
        num_classes=3,
        num_heads=4,
        pretrained=True  
    ).to(device)

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

# Task 3.2 - Multi-head Attention (MHA) for both ResNet18 and EfficientNet
def train_test_task32(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir, test_image_dir,
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
    model = build_model_task32(
        backbone=backbone,
        num_classes=3,
        num_heads=4,
        pretrained=True   
    ).to(device)

    for p in model.parameters():
        p.requires_grad = True

    # training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"best_{backbone}.pt")

    # load pretrained backbone
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    if backbone == "efficientnet":
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


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


# TASK 4

# Functions for task 4

# WEIGHTED AVERAGE ENSEMBLE LEARNING METHOD
def generate_onsite_result_task4(ckpt_resnet18, ckpt_effnet, ckpt_resnet18_se, onsite_loader, output_csv):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    model_resnet = build_model("resnet18", num_classes=3, pretrained=False).to(device)
    model_effnet = build_model("efficientnet", num_classes=3, pretrained=False).to(device)
    model_resnet_se = build_model_task31(num_classes=3).to(device)

    model_resnet.load_state_dict(torch.load(ckpt_resnet18, map_location=device)) # resnet task 1.3
    model_effnet.load_state_dict(torch.load(ckpt_effnet, map_location=device)) #efficientnet task 1.3
    model_resnet_se.load_state_dict(torch.load(ckpt_resnet18_se, map_location=device)) # resnet with SE task 3.1

    model_resnet.eval()
    model_effnet.eval()
    model_resnet_se.eval()

    # ensemble weights
    w_resnet = 0.40 # best performance
    w_resnet_se = 0.30
    w_effnet = 0.30

    thresholds = [0.4, 0.45, 0.4]
    predictions = []

    # inference
    with torch.no_grad():
        for imgs, names in onsite_loader:
            imgs = imgs.to(device)

            p_resnet = torch.sigmoid(model_resnet(imgs))
            p_effnet = torch.sigmoid(model_effnet(imgs))
            p_resnet_se = torch.sigmoid(model_resnet_se(imgs))

            ensemble_prob = (
                w_resnet * p_resnet +
                w_effnet * p_effnet +
                w_resnet_se * p_resnet_se
            )

            preds = (ensemble_prob.cpu().numpy() > thresholds).astype(int)

            for name, pred in zip(names, preds):
                predictions.append([name] + pred.tolist())

    # save CSV
    df = pd.DataFrame(predictions, columns=["id", "D", "G", "A"])
    df.to_csv(output_csv, index=False)

    print(f"Saved ensemble submission file: {output_csv}")


def weighted_average(model_paths):
    # Load the models saved before
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    test_ds  = RetinaMultiLabelDataset(test_csv, test_image_dir, test_transform)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    # ResNet18 (Task 1.3)
    model_resnet = build_model(backbone="resnet18", num_classes=3, pretrained=False).to(device)
    model_resnet.load_state_dict(torch.load(model_paths["resnet"], map_location=device))
    model_resnet.eval()

    # EfficientNet (Task 1.3)
    model_effnet = build_model(backbone="efficientnet", num_classes=3, pretrained=False).to(device)
    model_effnet.load_state_dict(torch.load(model_paths["effnet"], map_location=device))
    model_effnet.eval()

    # ResNet18 + SE (Task 3.1)
    model_resnet_se = build_model_task31(num_classes=3).to(device)
    model_resnet_se.load_state_dict(torch.load(model_paths["resnet_se"], map_location=device))
    model_resnet_se.eval()

    # define weights for the models
    w_resnet = 0.40 # best performance
    w_resnet_se = 0.30
    w_effnet = 0.30

    # calculate weighted average
    y_true, y_pred = [], []

    thresholds = [0.4, 0.45, 0.4]

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)

            # Get probabilities
            p_resnet = torch.sigmoid(model_resnet(imgs))
            p_effnet = torch.sigmoid(model_effnet(imgs))
            p_resnet_se = torch.sigmoid(model_resnet_se(imgs))

            # Weighted average
            ensemble_prob = (
                w_resnet * p_resnet +
                w_effnet * p_effnet +
                w_resnet_se * p_resnet_se
            )

            # Apply thresholds
            preds = (ensemble_prob.cpu().numpy() > thresholds).astype(int)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # compute metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    disease_names = ["DR", "Glaucoma", "AMD"]

    print("\n=== Ensemble Results (Weighted Average) ===\n")

    for i, disease in enumerate(disease_names):  # compute metrics for every disease
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        print(f"{disease} Results [Weighted average Ensemble]")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")
        print(f"Kappa    : {kappa:.4f}\n")

    # no model to save, no additional training was done

# AUTOENCODERS
# define the CVAE
class CVAE(nn.Module):
    def __init__(self, latent_dim=128, num_classes=3):
        super(CVAE, self).__init__()
        self.num_classes = num_classes

        # encoder: compresses 256x256 image -> latent vector
        # input channels = 3 (RGB of images) + num_classes (label conditioning)
        self.enc_conv1 = nn.Conv2d(3 + num_classes, 32, 4, 2, 1)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)

        # parameters to sample
        self.fc_mu = nn.Linear(256*16*16, latent_dim)
        self.fc_logvar = nn.Linear(256*16*16, latent_dim)

        # decoder: expands latent vector -> 256x256 image
        self.dec_fc = nn.Linear(latent_dim + num_classes, 256*16*16)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encode(self, x, labels):
        # broadcast label to image size and concatenate
        labels = labels.view(labels.size(0), self.num_classes, 1, 1)
        ones = torch.ones(x.size(0), self.num_classes, x.size(2), x.size(3)).to(x.device)
        x = torch.cat([x, labels * ones], dim=1)

        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    # reparametrization is a trick used because we need to sample a random code from the distribution
    # instead of sampling directly, random noise is sampled and then shifted by mean and
    # variance of the model predicted
    # to not break the chain of derivatives
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        z = torch.cat([z, labels], dim=1)
        x = self.dec_fc(z)
        x = x.view(x.size(0), 256, 16, 16)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        return torch.sigmoid(self.dec_conv4(x))

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# loss function is made of two parts
def loss_function_vae(recon_x, x, mu, logvar):

    # first part, wich is the reconstruction loss --> MSE: accuracy term
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # second part is the KL Divergence
    # forces the hidden codes to look like a standard normal distribution
    # needed to organize the latent space smoothly
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # 0.5 * KLD to give more importance to reconstructing proper images
    return BCE + 0.5 * KLD


# train the model
def train_cvae_model(train_loader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training CVAE (Generation Model)...")
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data, labels)
            loss = loss_function_vae(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f'VAE Epoch: {epoch+1} average loss: {train_loss / len(train_loader.dataset):.4f}')
    return model



# main
if __name__ == "__main__":
    # Task 1.1
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'
    print("\nOffsite results with ResNet18, no training:\n")
    
    model_task11 = train_test_task11(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=0, batch_size=32, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)
    
    torch.save(model_task11.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task11.pt")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'
    print("\nOffsite results with EfficientNet, no training:\n")

    model_task11 = train_test_task11(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=0, batch_size=32, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

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

    model_task12 = train_test_task12(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=30, batch_size=32, lr=1e-3, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task12.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task12.pt")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'
    print("\nOffsite results with EfficientNet, training only classifier:\n")

    model_task12 = train_test_task12(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=30, batch_size=32, lr=1e-3, img_size=256, pretrained_backbone=pretrained_backbone)

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

    model_task13 = train_test_task13(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task13.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task13.pt")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'
    print("\nOffsite results with EfficientNet, training full model:\n")

    model_task13 = train_test_task13(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

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

    # Task 2.1
    # ...
    





    # Task 3.1
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'

    print("\nOffsite results with SE-ResNet18, training full model with Squeeze-and-Excitation (SE) attention method:\n")

    model_task31 = train_test_task31(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task31.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task31.pt")

    generate_onsite_result_task31(
        #backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task31.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task31_resnet18.csv"
    )

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'
    print("\nOffsite results with SE-EfficientNet, training full model with Squeeze-and-Excitation (SE) attention method:\n")

    model_task31 = train_test_task31(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task31.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task31.pt")

    generate_onsite_result_task31_effnet(
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task31.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task31_efficientnet.csv"
    )

    # Task 3.2
    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'
    backbone = 'resnet18'
    print("\nOffsite results with MHA-ResNet18, training full model with Multi-head Attention (MHA) attention method:\n")

    model_task32 = train_test_task32(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-4, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task32.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task32.pt")

    generate_onsite_result_task32(
        backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task32.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task32_resnet18.csv"
    )

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'
    backbone = 'efficientnet'
    print("\nOffsite results with MHA-EfficientNet, training full model with Multi-head Attention (MHA) attention method:\n")

    model_task32 = train_test_task32(backbone, train_csv, val_csv, test_csv, train_image_dir, val_image_dir,
        test_image_dir, epochs=50, batch_size=16, lr=1e-5, img_size=256, pretrained_backbone=pretrained_backbone)

    torch.save(model_task32.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task32.pt")

    generate_onsite_result_task32(
        backbone="efficientnet",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task32.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task32_efficientnet.csv"
    )

    # Task 4
    # WEIGHTED AVERAGE
    model_paths = {
        "resnet": "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task13.pt",
        "effnet": "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task13.pt",
        "resnet_se": "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task31.pt",
    }

    # Create weighted average of models in model_paths and print metrics for offsite
    weighted_average(model_paths)

    generate_onsite_result_task4(
        ckpt_resnet18=model_paths["resnet"],
        ckpt_effnet=model_paths["effnet"],
        ckpt_resnet18_se=model_paths["resnet_se"],
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task4_ensemble.csv"
    )

    # AUTOENCODERS
    # setup data for training the autoencoder
    transform_vae = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_ds_vae = RetinaMultiLabelDataset(train_csv, train_image_dir, transform_vae)
    train_loader_vae = DataLoader(train_ds_vae, batch_size=32, shuffle=True, num_workers=2)

    # train the VAE
    cvae_model = train_cvae_model(train_loader_vae, epochs=30)

    # generate synthetic data and merge with the rest
    # create a new directory for the augmented dataset
    aug_img_dir = "/content/drive/MyDrive/final_project_resources/images/train_aug"
    if os.path.exists(aug_img_dir):
        shutil.rmtree(aug_img_dir)
    os.makedirs(aug_img_dir, exist_ok=True)

    # copy original images images to this new folder
    print("Copying original images to new training folder...")
    df_orig = pd.read_csv(train_csv)
    for img_name in df_orig.iloc[:, 0]:
        src = os.path.join(train_image_dir, img_name)
        dst = os.path.join(aug_img_dir, img_name)
        shutil.copy(src, dst)

    # generate synthetic images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae_model.eval()

    # we augment Glaucoma (index 1) and AMD (index 2)
    # we generate 350 images for G and AMD to balance them with DR
    num_aug = 350
    new_rows = []

    print(f"Generating {num_aug} synthetic images per minority class...")
    with torch.no_grad():
        for class_idx, class_name in [(1, "Glaucoma"), (2, "AMD")]:
            label_tensor = torch.zeros(num_aug, 3).to(device)
            label_tensor[:, class_idx] = 1.0

            z = torch.randn(num_aug, 128).to(device)
            gen_imgs = cvae_model.decode(z, label_tensor).cpu()

            for i in range(num_aug):
                file_name = f"aug_{class_name}_{i}.jpg"
                save_path = os.path.join(aug_img_dir, file_name)
                save_image(gen_imgs[i], save_path)

                # create CSV row: [id, DR, G, A]
                row = [file_name, 0.0, 0.0, 0.0]
                row[class_idx + 1] = 1.0
                new_rows.append(row)

    # create new CSV
    df_aug = pd.DataFrame(new_rows, columns=df_orig.columns)
    df_combined = pd.concat([df_orig, df_aug], ignore_index=True)
    aug_csv_path = "/content/drive/MyDrive/final_project_resources/train_aug.csv"
    df_combined.to_csv(aug_csv_path, index=False)

    print(f"Augmented Dataset Ready! CSV: {aug_csv_path}")
    print(f"Total Training Images: {len(df_combined)}")

    # train - ResNet18
    print("STARTING TASK 4 CLASSIFICATION")
    print("Using ResNet18 on VAE-Augmented Dataset")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_resnet18_ep50.pt'

    # we reuse train_test_task13 as function
    model_task4 = train_test_task13(
        backbone='resnet18',
        train_csv=aug_csv_path, # augmented set
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=aug_img_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        epochs=40,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        save_dir="checkpoints",
        pretrained_backbone=pretrained_backbone
    )

    # save the final Task 4 model
    torch.save(model_task4.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task4_vae_2.pt")

    # generate onsite submission
    generate_onsite_result_task13(
        backbone="resnet18",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_resnet18_task4_vae_2.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task4_vae_resnet18_2.csv"
    )

    # train - EfficientNet
    print("STARTING TASK 4 CLASSIFICATION")
    print("Using Efficient on VAE-Augmented Dataset")

    pretrained_backbone = '/content/drive/MyDrive/final_project_resources/pretrained_backbone/ckpt_efficientnet_ep50.pt'

    # we reuse train_test_task13 as function
    model_task4 = train_test_task13(
        backbone='efficientnet',
        train_csv=aug_csv_path, # augmented set
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=aug_img_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        epochs=40,
        batch_size=16,
        lr=1e-5,
        img_size=256,
        save_dir="checkpoints",
        pretrained_backbone=pretrained_backbone
    )

    # save the final Task 4 model
    torch.save(model_task4.state_dict(), "/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task4_vae_2.pt")

    # generate onsite submission
    generate_onsite_result_task13(
        backbone="efficientnet",
        checkpoint_path="/content/drive/MyDrive/final_project_resources/saved_models/best_efficientnet_task4_vae_2.pt",
        onsite_loader=generate_onsite_loader(),
        output_csv="/content/drive/MyDrive/final_project_resources/submissions/pred_task4_vae_efficientnet_2.csv"
    )





















