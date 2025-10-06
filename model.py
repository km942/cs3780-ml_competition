# For the project, used Google Colab Pro and mounted Google Drive to access the training and test data stored as pickle files.

# Defined an EnhancedRPSDataset class to load paired 24×24 grayscale images, applying resizing,
#  normalization, and optional augmentations including rotations, affine transforms, color jitter, 
# flips, perspective distortions, and Gaussian blur.
#
# Implemented ImprovedResNet34Siamese, a Siamese network based on a pretrained ResNet-34 with a modified one-channel input convolution, 
# an attention mechanism over feature maps, and fully connected layers with batch normalization, dropout, and a sigmoid output for binary classification.
#
# Structured the training into three stages: first freezing the backbone to train only the fully connected layers and attention with a 
# CosineAnnealingWarmRestarts scheduler, then unfreezing deeper layers (layer3 and layer4) and training with a OneCycleLR scheduler and early stopping, 
# and finally fine-tuning all parameters with CosineAnnealingLR and early stopping. I used a Focal Loss to address class imbalance and saved the best model
#  checkpoints after each stage.
#
# Performed test-time augmentation (horizontal flips and small rotations),  aggregated TTA predictions with a 0.5 threshold mapped to {+1, -1},
# and then outputted the final submission CSV,


import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pickle
import sys

sys.modules['numpy._core.numeric'] = np.core.numeric
sys.modules['numpy.core.numeric'] = np.core.numeric
np.int = int  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EnhancedRPSDataset(Dataset):
   def __init__(self, img1_data, img2_data, labels=None, augment=False):
       self.img1_data = img1_data
       self.img2_data = img2_data
       self.labels = labels
       self.augment = augment
       self.mean = [0.485]
       self.std = [0.229]
       self.basic_transform = transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),  
           transforms.ToTensor(),
           transforms.Normalize(mean=self.mean, std=self.std)
       ])

       if self.augment:
           self.transform = transforms.Compose([
               transforms.ToPILImage(),
               transforms.Resize((224, 224)),
               transforms.RandomRotation(30),  
               transforms.RandomAffine(
                   degrees=0,
                   translate=(0.15, 0.15), 
                   scale=(0.85, 1.15)  
               ),
               transforms.ColorJitter(brightness=0.25, contrast=0.25),
               transforms.RandomHorizontalFlip(p=0.5),
               transforms.RandomPerspective(distortion_scale=0.25, p=0.3),
               transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
               transforms.ToTensor(),
               transforms.Normalize(mean=self.mean, std=self.std)
           ])
       else:
           self.transform = self.basic_transform


   def __len__(self):
       return len(self.img1_data)


   def __getitem__(self, idx):
       
       img1 = np.array(self.img1_data[idx]).reshape(24, 24).astype(np.uint8)
       img2 = np.array(self.img2_data[idx]).reshape(24, 24).astype(np.uint8)



       img1 = self.transform(img1)
       img2 = self.transform(img2)


       if self.labels is not None:
           label = torch.tensor(1.0 if self.labels[idx] == 1 else 0.0, dtype=torch.float32)
           return img1, img2, label
       else:
           return img1, img2



class ImprovedResNet34Siamese(nn.Module):
   def __init__(self, pretrained=True):
       super(ImprovedResNet34Siamese, self).__init__()

       resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
       self.feature_dim = 512
      
     
       original_conv = resnet.conv1
       self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
      
      
       if pretrained:
           with torch.no_grad():
               self.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
      
     
       self.bn1 = resnet.bn1
       self.relu = resnet.relu
       self.maxpool = resnet.maxpool
       self.layer1 = resnet.layer1
       self.layer2 = resnet.layer2
       self.layer3 = resnet.layer3
       self.layer4 = resnet.layer4
       self.avgpool = resnet.avgpool
      
  
       self.attention = nn.Sequential(
           nn.Conv2d(512, 64, kernel_size=1),
           nn.ReLU(),
           nn.Conv2d(64, 512, kernel_size=1),
           nn.Sigmoid()
       )
      
       
       self.fc = nn.Sequential(
           nn.Linear(self.feature_dim * 2, 768),
           nn.BatchNorm1d(768),  
           nn.ReLU(inplace=True),
           nn.Dropout(0.4),
           nn.Linear(768, 384),
           nn.BatchNorm1d(384),  
           nn.ReLU(inplace=True),
           nn.Dropout(0.4),
           nn.Linear(384, 128),
           nn.BatchNorm1d(128), 
           nn.ReLU(inplace=True),
           nn.Dropout(0.3),
           nn.Linear(128, 1),
           nn.Sigmoid()
       )
  
   def forward_one(self, x):
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)
       x = self.maxpool(x)
      
       x = self.layer1(x)
       x = self.layer2(x)
       x = self.layer3(x)
       x = self.layer4(x)
      
    
       attention_weights = self.attention(x)
       x = x * attention_weights
      
       x = self.avgpool(x)
       x = torch.flatten(x, 1)
      
       return x
  
   def forward(self, img1, img2):
      
       feat1 = self.forward_one(img1)
       feat2 = self.forward_one(img2)
       combined = torch.cat((feat1, feat2), 1)
      
       output = self.fc(combined)
       return output


def train_model_2(model, train_loader, val_loader, criterion,
                initial_lr=0.0003,  
                num_epochs_stage1=8,  
                num_epochs_stage2=20,  
                num_epochs_stage3=12,  
                patience=6,  
                model_name="model_2_resnet34"):
  
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
  
  
   all_train_losses = []
   all_val_losses = []
   all_train_accs = []
   all_val_accs = []
   best_val_acc = 0.0
  
   
   print("Stage 1: Training FC layers and attention mechanism")
   for name, param in model.named_parameters():
       if not (name.startswith('fc') or name.startswith('attention')):
           param.requires_grad = False
  
   optimizer = torch.optim.AdamW(
       filter(lambda p: p.requires_grad, model.parameters()),
       lr=initial_lr,
       weight_decay=0.01  
   )
  
   
   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, T_0=3, T_mult=2, eta_min=initial_lr/20
   )
  
   for epoch in range(num_epochs_stage1):
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0
      
       train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage1} [Train]")
       for img1, img2, labels in train_bar:
           img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
          
           optimizer.zero_grad()
          
           outputs = model(img1, img2).squeeze()
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           running_loss += loss.item() * img1.size(0)
           predicted = (outputs >= 0.5).float()
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
           train_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       train_loss = running_loss / len(train_loader.dataset)
       train_acc = correct / total
       all_train_losses.append(train_loss)
       all_train_accs.append(train_acc)
      
       model.eval()
       running_loss = 0.0
       correct = 0
       total = 0
      
       with torch.no_grad():
           val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage1} [Val]")
           for img1, img2, labels in val_bar:
               img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
              
               outputs = model(img1, img2).squeeze()
               loss = criterion(outputs, labels)
              
               running_loss += loss.item() * img1.size(0)
               predicted = (outputs >= 0.5).float()
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
              
               val_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       val_loss = running_loss / len(val_loader.dataset)
       val_acc = correct / total
       all_val_losses.append(val_loss)
       all_val_accs.append(val_acc)
      
       print(f"Epoch {epoch+1}/{num_epochs_stage1}: "
             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
      
       scheduler.step()

       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), f'best_{model_name}_stage1.pth')
           print(f"Model saved with validation accuracy: {best_val_acc:.4f}")
  
   print("\nStage 2: Unfreezing layers 3 and 4 (Extended Training)")
   for name, param in model.named_parameters():
       if name.startswith('layer3') or name.startswith('layer4'):
           param.requires_grad = True
  
   optimizer = torch.optim.AdamW(
       filter(lambda p: p.requires_grad, model.parameters()),
       lr=initial_lr / 5,
       weight_decay=0.005  
   )
  
   scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer,
       max_lr=initial_lr / 2,
       steps_per_epoch=len(train_loader),
       epochs=num_epochs_stage2,
       pct_start=0.3,  
       div_factor=25.0
   )
  
   no_improve_epochs = 0
  
   for epoch in range(num_epochs_stage2):
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0
      
       train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage2} [Train]")
       for img1, img2, labels in train_bar:
           img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
          
           optimizer.zero_grad()
          
           outputs = model(img1, img2).squeeze()
           loss = criterion(outputs, labels)
          
           loss.backward()
          
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          
           optimizer.step()
           scheduler.step()  
          
           running_loss += loss.item() * img1.size(0)
           predicted = (outputs >= 0.5).float()
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
          
           train_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       train_loss = running_loss / len(train_loader.dataset)
       train_acc = correct / total
       all_train_losses.append(train_loss)
       all_train_accs.append(train_acc)
      
       model.eval()
       running_loss = 0.0
       correct = 0
       total = 0
      
       with torch.no_grad():
           val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage2} [Val]")
           for img1, img2, labels in val_bar:
               img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
              
               outputs = model(img1, img2).squeeze()
               loss = criterion(outputs, labels)
              
               running_loss += loss.item() * img1.size(0)
               predicted = (outputs >= 0.5).float()
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
              
               val_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       val_loss = running_loss / len(val_loader.dataset)
       val_acc = correct / total
       all_val_losses.append(val_loss)
       all_val_accs.append(val_acc)
      
       print(f"Epoch {epoch+1}/{num_epochs_stage2}: "
             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
      
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), f'best_{model_name}_stage2.pth')
           print(f"Model saved with validation accuracy: {best_val_acc:.4f}")
           no_improve_epochs = 0
       else:
           no_improve_epochs += 1
           if no_improve_epochs >= patience:
               print(f"Early stopping at epoch {epoch+1}")
               break
  
   print("\nStage 3: Fine-tuning entire model")
   # Unfreeze all parameters
   for param in model.parameters():
       param.requires_grad = True
  
   optimizer = torch.optim.AdamW(
       model.parameters(),
       lr=initial_lr / 20,
       weight_decay=0.001
   )
  
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer,
       T_max=num_epochs_stage3,
       eta_min=initial_lr / 200
   )

   no_improve_epochs = 0
  
   for epoch in range(num_epochs_stage3):
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0
      
       train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage3} [Train]")
       for img1, img2, labels in train_bar:
           img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
          
           optimizer.zero_grad()
          
           outputs = model(img1, img2).squeeze()
           loss = criterion(outputs, labels)
          
           loss.backward()
          
           torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          
           optimizer.step()
          
           running_loss += loss.item() * img1.size(0)
           predicted = (outputs >= 0.5).float()
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
          
           train_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       train_loss = running_loss / len(train_loader.dataset)
       train_acc = correct / total
       all_train_losses.append(train_loss)
       all_train_accs.append(train_acc)
      
       model.eval()
       running_loss = 0.0
       correct = 0
       total = 0
      
       with torch.no_grad():
           val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs_stage3} [Val]")
           for img1, img2, labels in val_bar:
               img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
              
               outputs = model(img1, img2).squeeze()
               loss = criterion(outputs, labels)
              
               running_loss += loss.item() * img1.size(0)
               predicted = (outputs >= 0.5).float()
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
              
               val_bar.set_postfix(loss=loss.item(), acc=correct/total)
      
       val_loss = running_loss / len(val_loader.dataset)
       val_acc = correct / total
       all_val_losses.append(val_loss)
       all_val_accs.append(val_acc)
      
       print(f"Epoch {epoch+1}/{num_epochs_stage3}: "
             f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
             f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
      
       scheduler.step()
      
       if val_acc > best_val_acc:
           best_val_acc = val_acc
           torch.save(model.state_dict(), f'best_{model_name}_final.pth')
           print(f"Model saved with validation accuracy: {best_val_acc:.4f}")
           no_improve_epochs = 0
       else:
           no_improve_epochs += 1
           if no_improve_epochs >= patience:
               print(f"Early stopping at epoch {epoch+1}")
               break
  
   epochs = range(1, len(all_train_losses) + 1)
   plt.figure(figsize=(12, 5))
  
   plt.subplot(1, 2, 1)
   plt.plot(epochs, all_train_losses, 'b-', label='Training Loss')
   plt.plot(epochs, all_val_losses, 'r-', label='Validation Loss')
   plt.axvline(x=num_epochs_stage1, color='g', linestyle='--')
   plt.axvline(x=num_epochs_stage1 + num_epochs_stage2, color='g', linestyle='--')
   plt.title('Training and Validation Loss')
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
  
   plt.subplot(1, 2, 2)
   plt.plot(epochs, all_train_accs, 'b-', label='Training Accuracy')
   plt.plot(epochs, all_val_accs, 'r-', label='Validation Accuracy')
   plt.axvline(x=num_epochs_stage1, color='g', linestyle='--')
   plt.axvline(x=num_epochs_stage1 + num_epochs_stage2, color='g', linestyle='--')
   plt.title('Training and Validation Accuracy')
   plt.xlabel('Epochs')
   plt.ylabel('Accuracy')
   plt.legend()
  
   plt.tight_layout()
   plt.savefig(f'{model_name}_training_curves.png')
   plt.show()
  
   return best_val_acc, all_train_losses, all_val_losses, all_train_accs, all_val_accs

def train_and_predict_model_2(train_data, test_data):
   train_indices, val_indices = train_test_split(
       range(len(train_data['img1'])),
       test_size=0.15,  
       random_state=100,  
       stratify=train_data['label']
   )
  
   train_dataset = EnhancedRPSDataset(
       [train_data['img1'][i] for i in train_indices],
       [train_data['img2'][i] for i in train_indices],
       [train_data['label'][i] for i in train_indices],
       augment=True
   )
  
   val_dataset = EnhancedRPSDataset(
       [train_data['img1'][i] for i in val_indices],
       [train_data['img2'][i] for i in val_indices],
       [train_data['label'][i] for i in val_indices],
       augment=False
   )
  
   test_dataset = EnhancedRPSDataset(
       test_data['img1'],
       test_data['img2'],
       None,
       augment=False
   )
  
   batch_size = 32  
  
   train_loader = DataLoader(
       train_dataset,
       batch_size=batch_size,
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )
  
   val_loader = DataLoader(
       val_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=4,
       pin_memory=True
   )
  
   test_loader = DataLoader(
       test_dataset,
       batch_size=batch_size,
       shuffle=False,
       num_workers=4,
       pin_memory=True
   )
  
   model = ImprovedResNet34Siamese(pretrained=True).to(device)
  
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
           super(FocalLoss, self).__init__()
           self.alpha = alpha
           self.gamma = gamma
           self.reduction = reduction
           self.bce = nn.BCELoss(reduction='none')
          
       def forward(self, inputs, targets):
           BCE_loss = self.bce(inputs, targets)
           pt = torch.exp(-BCE_loss)
           F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
          
           if self.reduction == 'mean':
               return torch.mean(F_loss)
           elif self.reduction == 'sum':
               return torch.sum(F_loss)
           else:
               return F_loss
  
   criterion = FocalLoss(alpha=0.25, gamma=2.0)

   model_name = "model_2_resnet34_improved"
   best_acc, _, _, _, _ = train_model_2(
       model, train_loader, val_loader, criterion,
       initial_lr=0.0003,
       num_epochs_stage1=8,
       num_epochs_stage2=20,
       num_epochs_stage3=12,
       patience=6,
       model_name=model_name
   )
  
   model.load_state_dict(torch.load(f'best_{model_name}_final.pth'))
   print(f"Model 2 best accuracy: {best_acc:.4f}")
  
   print("Generating predictions with test-time augmentation...")
   model.eval()
  
   tta_transforms = [
       transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])
       ]),
       transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),
           transforms.RandomHorizontalFlip(p=1.0),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])
       ]),
       transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),
           transforms.RandomRotation(degrees=(5, 5)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])
       ]),
       transforms.Compose([
           transforms.ToPILImage(),
           transforms.Resize((224, 224)),
           transforms.RandomRotation(degrees=(-5, -5)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229])
       ])
   ]
  
   all_predictions = []
  
   for tta_idx, transform in enumerate(tta_transforms):
       print(f"Running TTA {tta_idx+1}/{len(tta_transforms)}...")
       predictions = []
      
       with torch.no_grad():
           for img1, img2 in tqdm(test_loader, desc=f"Generating predictions (TTA {tta_idx+1})"):
               img1, img2 = img1.to(device), img2.to(device)
               outputs = model(img1, img2).squeeze()
               predicted = (outputs >= 0.5).float()
               predictions.extend([1 if pred == 1.0 else -1 for pred in predicted.cpu().numpy()])
      
       all_predictions.append(predictions)
  
   final_predictions = []
   for i in range(len(all_predictions[0])):
       avg_pred = np.mean([1 if all_predictions[j][i] == 1 else 0 for j in range(len(tta_transforms))])
       final_predictions.append(1 if avg_pred >= 0.5 else -1)
  
   submission = pd.DataFrame({
       'id': test_data['id'],
       'label': final_predictions
   })
  

   submission.to_csv('model_2_improved_submission.csv', index=False)
   print("Saved predictions to 'model_2_improved_submission.csv'")
  

   unique_preds, counts = np.unique(final_predictions, return_counts=True)
   print("\nPrediction distribution:")
   for value, count in zip(unique_preds, counts):
       print(f"  {value}: {count} ({count/len(final_predictions)*100:.2f}%)")
  
   return submission, model, best_acc


def load_data(train_path, test_path):
   base_path = '/content/drive/MyDrive/expanded-rock-paper-scissors/'

   with open(base_path + 'train.pkl', 'rb') as f:
       train_data = pickle.load(f)

   with open(base_path + 'test.pkl', 'rb') as f:
       test_data = pickle.load(f)
  
   print(f"Loaded {len(train_data['img1'])} training samples and {len(test_data['img1'])} test samples")
   return train_data, test_data

if __name__ == "__main__":
   train_path = 'train_data.pkl'
   test_path = 'test_data.pkl'
  

   train_data, test_data = load_data(train_path, test_path)
  
   submission, model, accuracy = train_and_predict_model_2(train_data, test_data)
  
   print(f"Model 2 training completed with accuracy: {accuracy:.4f}")
   print("Final predictions saved to 'model_2_improved_submission.csv'")

