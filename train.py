# train.py
import argparse
import os
import torch
import torch.optim as optim
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader
from tqdm import tqdm

from utils import get_train_data_loader, get_data_loader, dice
from networks.Unet import UNet3D
from networks.transunet import TransUNet3D


def main():
    parser = argparse.ArgumentParser(description='Train UNet/TransUNet on BTCV dataset.')
    # Data paths
    parser.add_argument('--data_path', type=str, default='/path/to/train/images')
    parser.add_argument('--label_path', type=str, default='/path/to/train/labels')
    parser.add_argument('--val_data_path', type=str, default='/path/to/val/images')
    parser.add_argument('--val_label_path', type=str, default='/path/to/val/labels')

    # Training & logging
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save models and logs.')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--eval_interval', type=int, default=5)

    # Model and data parameters
    parser.add_argument('--roi_width', type=int, default=96)
    parser.add_argument('--roi_height', type=int, default=96)
    parser.add_argument('--roi_depth', type=int, default=96)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=14)

    # Distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)

    # Resume checkpoint
    parser.add_argument('--resume_checkpoint', type=str, default=None)

    # Supported models
    parser.add_argument('--model', type=str, required=True, choices=['unet3d', 'transunet'])

    # TransUNet parameters
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_dim', type=int, default=3072)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Data preprocessing parameters
    parser.add_argument('--a_min', type=float, default=-175.0)
    parser.add_argument('--a_max', type=float, default=250.0)
    parser.add_argument('--b_min', type=float, default=0.0)
    parser.add_argument('--b_max', type=float, default=1.0)
    parser.add_argument('--num_samples_per_image', type=int, default=2)

    args = parser.parse_args()

    # -------------------- Distributed / device setup --------------------
    local_rank = args.local_rank
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"Distributed training with {world_size} GPUs. Rank: {rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
        rank = 0

    # Output directory
    if rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # TensorBoard writer
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    else:
        writer = None

    # Check ROI compatibility for TransUNet
    if args.model == 'transunet':
        ps = args.patch_size
        if (args.roi_width % ps != 0) or (args.roi_height % ps != 0) or (args.roi_depth % ps != 0):
            raise ValueError("For transunet, each of roi_width/roi_height/roi_depth must be divisible by patch_size.")

    # -------------------- Build model --------------------
    if args.model == 'unet3d':
        model = UNet3D(n_channels=1, n_classes=args.num_classes, base_features=16).to(device)
    elif args.model == 'transunet':
        model = TransUNet3D(
            img_size=(args.roi_width, args.roi_height, args.roi_depth),
            in_channels=1,
            num_classes=args.num_classes,
            base_channels=args.patch_size,
            embed_dim=args.embed_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.depth,
            dropout=args.dropout
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        print(f"Model wrapped with DDP on device {local_rank}")

    # -------------------- Loss & optimizer --------------------
    criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, batch=True)
    optimizer = (
        optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.model == 'unet3d'
        else optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    )

    # -------------------- Resume checkpoint --------------------
    start_epoch = 0
    if args.resume_checkpoint is not None and os.path.isfile(args.resume_checkpoint):
        map_location = {f'cuda:{0}': f'cuda:{local_rank}'} if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume_checkpoint, map_location=map_location)
        model_state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_state)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"[Rank {rank}] Resumed from epoch {start_epoch}")

    # -------------------- DataLoader --------------------
    shuffle = not args.distributed
    train_loader = get_train_data_loader(
        data_path=args.data_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        roi_depth=args.roi_depth,
        roi_height=args.roi_height,
        roi_width=args.roi_width,
        a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
        num_samples_per_image=args.num_samples_per_image,
        transforms=None
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            dataset=train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        train_sampler = None

    if rank == 0:
        val_loader = get_data_loader(
            data_path=args.val_data_path,
            label_path=args.val_label_path,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            roi_depth=args.roi_depth,
            roi_height=args.roi_height,
            roi_width=args.roi_width,
            a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max,
            transforms=None
        )
    else:
        val_loader = None

    # -------------------- Training loop --------------------
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0

        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(total=len(train_loader), desc=f"Epoch [{epoch+1}/{args.num_epochs}]", ncols=100) if rank == 0 else None

        for batch_data in train_loader:
            images = batch_data['image'].to(device, non_blocking=True)
            labels = batch_data['label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if pbar is not None:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        epoch_loss /= len(train_loader)

        if args.distributed:
            loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = (loss_tensor.item() / world_size)

        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            print(f"Epoch [{epoch+1}/{args.num_epochs}]  AvgLoss: {epoch_loss:.4f}")

        # Validation
        if ((epoch + 1) % args.eval_interval == 0) and rank == 0:
            model.eval()
            dice_list_case = []
            with torch.no_grad():
                val_pbar = tqdm(total=len(val_loader), desc="Validation", ncols=100)
                for val_data in val_loader:
                    val_images = val_data['image'].to(device, non_blocking=True)
                    val_labels = val_data['label'].to(device, non_blocking=True)

                    val_outputs = sliding_window_inference(
                        val_images,
                        roi_size=(args.roi_width, args.roi_height, args.roi_depth),
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode='gaussian'
                    )

                    outputs_seg = torch.softmax(val_outputs, dim=1).cpu().numpy()
                    outputs_seg = np.argmax(outputs_seg, axis=1).astype(np.uint8)
                    labels_np = val_labels.cpu().numpy()[:, 0, :, :, :]

                    dice_list_sub = []
                    for organ_idx in range(1, args.num_classes):
                        organ_dice = dice(outputs_seg[0] == organ_idx, labels_np[0] == organ_idx)
                        dice_list_sub.append(organ_dice)
                    mean_dice = float(np.mean(dice_list_sub))
                    dice_list_case.append(mean_dice)

                    val_pbar.set_postfix({'Mean Dice': f"{mean_dice:.4f}"})
                    val_pbar.update(1)

                val_pbar.close()
                mean_dice_all = float(np.mean(dice_list_case))
                if writer is not None:
                    writer.add_scalar('Dice/validation', mean_dice_all, epoch)
                print(f"Epoch [{epoch+1}/{args.num_epochs}]  Val Mean Dice: {mean_dice_all:.4f}")

        # Save checkpoint
        if ((epoch + 1) % args.save_interval == 0) and rank == 0:
            ckpt_path = os.path.join(args.output_dir, f'{args.model}_epoch_{epoch+1}.pth')
            state_dict = model.module.state_dict() if args.distributed else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    if rank == 0 and writer is not None:
        writer.close()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
