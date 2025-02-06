from argparse import Namespace, ArgumentParser
from pathlib import Path

from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from timm.data import Mixup

# Dataset
from dataloader import SkinDataset

# Models
from models import Resnet10, EfficientNetB0, DenseNet, EfficientNetB1, SeNet50, Resnet50


def main(args: Namespace) -> None:
    # Transforms
    SelectedModel = DenseNet
    base_aug = [
        transforms.ToTensor(),
        transforms.Resize(args.img_size)
    ]

    tr_aug = [
        transforms.ColorJitter(brightness=.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-20, 20))
    ]
    valid_df = pd.read_csv(args.valid_gt)

    valid_ds = SkinDataset(valid_df, args.valid_root, n_sample=None, transform=transforms.Compose(base_aug))
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_worker)

    device = torch.device("cuda")

    pause_idx = len(list(args.save_root.glob('*')))
    for cls_idx in range(pause_idx, args.n_classifier):
        print(f'[*] Classifier {cls_idx + 1} is running')
        cls_save_model: Path = args.save_root.joinpath(f'Model_{cls_idx}')
        cls_save_model.mkdir(parents=True, exist_ok=True)

        prefix_name = f"{cls_save_model.parent.parent.name}-{cls_save_model.parent.name}-{cls_save_model.name}"

        train_df = pd.read_csv(args.train_gt)
        train_ds = SkinDataset(train_df, args.train_root, n_sample=args.n_sample,
                               transform=transforms.Compose(base_aug + tr_aug))

        n_select = np.random.choice(np.arange(len(train_ds)), size=args.n_img, replace=False)
        train_sub = Subset(train_ds, n_select)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_worker)

        mix_up_fn = None
        if args.mix_up:
            # mix_up_fn = Mixup(num_classes=7, mixup_alpha=1.5, cutmix_alpha=1)
            mix_up_fn = Mixup(num_classes=7, mixup_alpha=args.mx_alpha, cutmix_alpha=args.cm_alpha)

        img_ls = []
        for idx in range(len(train_sub)):
            img, _ = train_sub[idx]
            img_ls.append(img)

        img_ls = torch.stack(img_ls, dim=0)
        if args.mix_up:
            img_ls, _ = mix_up_fn(img_ls, torch.tensor(np.ones(len(img_ls, ))))
        img_grid = make_grid(img_ls)
        save_image(img_grid, cls_save_model.joinpath('samples.png'))
        model = SelectedModel()
        model.zero_grad()
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30], gamma=0.5, last_epoch=-1)

        best_thresh = 0
        best_auc = 0
        best_acc = 0
        criterion = nn.MSELoss()

        ep_tr_acc = []
        ep_tr_loss = []
        ep_vl_acc = []
        ep_vl_loss = []

        # Train
        break_cnt = 0
        for counter in range(args.epochs):

            epoch_iterator_train = tqdm(train_dl)
            tr_loss = 0.0
            tr_acc = 0.0
            step = 0

            str_ep = f"[{counter + 1}/{args.epochs}]"
            for idx, batch in enumerate(epoch_iterator_train):
                str_st = f"[{idx + 1}/{len(train_dl)}]"
                model.train()
                images, targets = batch[0].to(device), batch[1]

                # MixUp
                if args.mix_up and images.shape[0] % 2 == 0:
                    targets = targets.to(device)
                    images, targets = mix_up_fn(images, targets)
                else:
                    targets = F.one_hot(targets, num_classes=7)
                    targets = targets.to(device)

                outputs = model(images)
                targets = targets  # .view(-1, 1)
                loss = criterion(outputs.squeeze(1), targets.float())

                loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()

                tr_loss += loss.item()
                batch_acc = metrics.accuracy_score(np.argmax(targets.detach().cpu().numpy(), axis=-1),
                                                   np.argmax(outputs.detach().cpu().numpy(), axis=-1))
                tr_acc += batch_acc

                str_postfix = f'Epoch: {str_ep} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(tr_acc / (idx + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(tr_loss / (idx + 1), 3)}'
                show = OrderedDict({f'Model_{cls_idx}': str_postfix})
                epoch_iterator_train.set_postfix(show)
                step = idx
            scheduler.step()  # Update learning rate schedule
            ep_tr_acc.append(tr_acc / (step + 1))
            ep_tr_loss.append(tr_loss / (step + 1))

            with torch.no_grad():
                val_loss = 0.0
                val_acc = 0.0
                # preds = []
                # true_labels = []
                epoch_iterator_val = tqdm(valid_dl)
                step = 0
                for idx, batch in enumerate(epoch_iterator_val):
                    str_st = f"[{idx + 1}/{len(valid_dl)}]"
                    model.eval()
                    images, targets = batch[0].to(device), batch[1]
                    targets = F.one_hot(targets, num_classes=7)
                    targets = targets.to(device)
                    outputs = model(images)
                    targets = targets  # .view(-1, 1)
                    loss = criterion(outputs.squeeze(1), targets.float())
                    val_loss += loss.item()
                    batch_acc = metrics.accuracy_score(np.argmax(targets.detach().cpu().numpy(), axis=-1),
                                                       np.argmax(outputs.detach().cpu().numpy(), axis=-1))
                    val_acc += batch_acc

                    str_postfix = f'Epoch: {str_ep} Step: {str_st} Batch Acc: {round(batch_acc, 3)} Acc: {round(val_acc / (idx + 1), 3)} Batch Loss: {round(loss.item(), 3)} Loss: {round(val_loss / (idx + 1), 3)}'
                    show = OrderedDict({f'Model_{cls_idx}': str_postfix})
                    epoch_iterator_val.set_postfix(show)
                    # preds.append(outputs.sigmoid().detach().cpu().numpy())
                    # true_labels.append(targets.cpu().numpy())
                    step = idx
            ep_vl_acc.append(val_acc / (step + 1))
            ep_vl_loss.append(val_loss / (step + 1))
            break_cnt += 1
            if val_acc / (step + 1) > best_acc:
                print("Saving the model...")

                print(
                    f"[*] {best_acc} -> {val_acc / (step + 1)}"
                )
                best_acc = val_acc / (step + 1)

                torch.save(model,
                           cls_save_model.joinpath(f'{prefix_name}-best.pth'))
                break_cnt = 0

            if break_cnt >= args.imp == 0:
                break

        cls_save_asset = cls_save_model.joinpath('assets')
        cls_save_asset.mkdir(parents=True, exist_ok=True)

        # Save Assets
        subs = pd.DataFrame({
            "train_accuracy": np.array(ep_tr_acc),
            "train_loss": np.array(ep_tr_loss),
            "valid_accuracy": np.array(ep_vl_acc),
            "valid_loss": np.array(ep_vl_loss),

        })

        subs.to_csv(cls_save_asset.joinpath('submission.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    # Data
    parser.add_argument('--train_root', help='Train Image root', type=Path, default='data/train_valid_center')
    # parser.add_argument('--valid_root', help='Valid Image root', type=Path, default='data/valid_center')
    parser.add_argument('--train_gt', help='Train Ground Truth', type=Path,
                        default='data/Data/train_valid.csv')
    # parser.add_argument('--valid_gt', help='Valid Ground Truth', type=Path,
    #                     default='data/Data/Valid/ISIC2018_Task3_Validation_GroundTruth.csv')

    parser.add_argument('--valid_root', help='Valid Image root', type=Path, default='data/test_center')
    parser.add_argument('--valid_gt', help='Valid Ground Truth', type=Path,
                        default='data/Data/Test/ISIC2018_Task3_Test_GroundTruth.csv')

    # parser.add_argument('--test_root', help='Path to test images', type=Path, default='data/test_center')
    # parser.add_argument('--test_gt', help='Path to test ground truth', type=Path,
    #                     default='data/Data/Test/ISIC2018_Task3_Test_GroundTruth.csv')

    # Save
    parser.add_argument('--save_root', help='Save Root', type=Path, default='experience/train_23 ')

    # Hyperparameter
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=40)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=10)
    parser.add_argument('--lr', help='Learning Rate', type=float, default=1e-4)
    parser.add_argument('--n_sample', help='Number of samples', type=int, default=200)
    parser.add_argument('--n_classifier', help='Number of classifiers', type=int, default=20)
    parser.add_argument('--imp', help='No.Improvement', type=int, default=10)

    # Image
    parser.add_argument('--img_size', help='Image size', type=int, default=256)
    parser.add_argument('--n_worker', help='Number of workers', type=int, default=6)
    parser.add_argument('--n_img', help='Number of image to save', type=int, default=64)
    parser.add_argument('--mix_up', help='Enable mix-up', type=bool, default=False)
    parser.add_argument('--mx-alpha', '--mx_alpha', help='MixUp Alpha', type=float, default=.5)
    parser.add_argument('--cm-alpha', '--cm_alpha', help='CutMix alpha', type=float, default=0.)

    main(args=parser.parse_args())
