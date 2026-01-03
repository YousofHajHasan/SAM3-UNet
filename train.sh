CUDA_VISIBLE_DEVICES="0" \
# Model and data paths
SAM3_PATH="/root/SAM3-UNet/sam3.pt"
TRAIN_IMAGE_PATH="/root/sam3/spondy-29-sam/train/images/"
TRAIN_MASK_PATH="/root/sam3/spondy-29-sam/train/masks/"
VAL_IMAGE_PATH="/root/sam3/spondy-29-sam/val/images/"
VAL_MASK_PATH="/root/sam3/spondy-29-sam/val/masks/"
SAVE_PATH="./checkpoints/sam3unet_binary"

# Training hyperparameters
EPOCHS=20
LEARNING_RATE=0.0002
BATCH_SIZE=32
INPUT_SIZE=576

# Run training
python train.py \
    --sam3_path $SAM3_PATH \
    --train_image_path $TRAIN_IMAGE_PATH \
    --train_mask_path $TRAIN_MASK_PATH \
    --val_image_path $VAL_IMAGE_PATH \
    --val_mask_path $VAL_MASK_PATH \
    --save_path $SAVE_PATH \
    --epoch $EPOCHS \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --input_size $INPUT_SIZE
