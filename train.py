import os
import argparse
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import FullDataset
from SAM3UNet import SAM3UNet

# Argument configuration
parser = argparse.ArgumentParser("SAM3-UNet Binary Segmentation")
parser.add_argument("--sam3_path", type=str, required=True, 
                    help="path to the sam3 pretrained pth")
parser.add_argument("--train_image_path", type=str, required=True, 
                    help="path to the training images")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the training masks")
parser.add_argument("--val_image_path", type=str, required=True, 
                    help="path to the validation images")
parser.add_argument("--val_mask_path", type=str, required=True,
                    help="path to the validation masks")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoints")
parser.add_argument("--epoch", type=int, default=20, 
                    help="number of training epochs")
parser.add_argument("--lr", type=float, default=0.0002, 
                    help="learning rate")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--input_size", default=336, type=int, 
                    help="input image size")
args = parser.parse_args()


# Binary IoU computation
def compute_iou(pred, mask, threshold=0.5):
    """
    Compute IoU for binary segmentation
    pred: [B, 1, H, W] - logits
    mask: [B, 1, H, W] - binary mask
    """
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * mask).sum()
    union = pred_binary.sum() + mask.sum() - intersection
    
    if union > 0:
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou.item()
    return 0.0


# Binary Dice computation
def compute_dice(pred, mask, threshold=0.5):
    """
    Compute Dice coefficient for binary segmentation
    pred: [B, 1, H, W] - logits
    mask: [B, 1, H, W] - binary mask
    """
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    
    intersection = (pred_binary * mask).sum()
    dice = (2. * intersection + 1e-7) / (pred_binary.sum() + mask.sum() + 1e-7)
    
    return dice.item()


# Binary segmentation loss (keeping original loss function)
def structure_loss(pred, mask):
    # mask is already float with shape [B, 1, H, W] from dataset
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=61, stride=1, padding=30) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()

# Train the model for one epoch
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    epoch_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        x = batch['image'].to(device)
        target = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        pred = model(x)
        loss = structure_loss(pred, target)
        
        # Compute metrics
        iou = compute_iou(pred, target)
        dice = compute_dice(pred, target)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        iou_score += iou
        dice_score += dice
        
        # Print progress every 50 batches
        if (i + 1) % 50 == 0:
            print(f"  Batch [{i+1}/{len(dataloader)}] - Loss: {loss.item():.4f}, IoU: {iou:.4f}, Dice: {dice:.4f}")
    
    avg_epoch_loss = epoch_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    return avg_epoch_loss, avg_iou, avg_dice


# Validate the model
def validate(model, dataloader, device):
    model.eval()
    val_loss, iou_score, dice_score = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            
            pred = model(x)
            
            loss = structure_loss(pred, target)
            iou = compute_iou(pred, target)
            dice = compute_dice(pred, target)
            
            val_loss += loss.item()
            iou_score += iou
            dice_score += dice
    
    avg_val_loss = val_loss / len(dataloader)
    avg_iou = iou_score / len(dataloader)
    avg_dice = dice_score / len(dataloader)
    
    return avg_val_loss, avg_iou, avg_dice


# Main training function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input size: {args.input_size}x{args.input_size}")
    
    # Load training dataset
    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 
                                args.input_size, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=8)
    
    # Load validation dataset
    val_dataset = FullDataset(args.val_image_path, args.val_mask_path, 
                              args.input_size, mode='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=8)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = SAM3UNet(args.sam3_path, args.input_size).to(device)
    
    optimizer = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}], 
                          lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=1.0e-7)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    # Track best validation IoU
    best_val_iou = 0.0
    best_model_path = None
    
    # Log file for tracking
    log_file = os.path.join(args.save_path, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_IoU,Train_Dice,Val_Loss,Val_IoU,Val_Dice\n")
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(args.epoch):
        print(f"\nEpoch [{epoch + 1}/{args.epoch}]")
        print("-" * 80)
        
        # Training
        print("Training...")
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validation
        print("Validating...")
        val_loss, val_iou, val_dice = validate(
            model, val_loader, device
        )
        
        # Update learning rate
        scheduler.step()
        
        # Print summary
        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{train_iou:.4f},{train_dice:.4f},"
                   f"{val_loss:.4f},{val_iou:.4f},{val_dice:.4f}\n")
        
        # Save model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            
            # Delete previous best model
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_model_path = os.path.join(args.save_path, f'best_model_epoch-{epoch + 1}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ NEW BEST MODEL! Val IoU: {best_val_iou:.4f} (saved to {best_model_path})")
        
        print("-" * 80)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best Model: {best_model_path}")
    print(f"Training log: {log_file}")
    print("="*80)


if __name__ == "__main__":
    main(args)