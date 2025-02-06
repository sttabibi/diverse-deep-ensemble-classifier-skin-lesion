from argparse import Namespace, ArgumentParser
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# Dataset
from dataloader import SkinDataset
np.random.seed(2023)


def cohen_correlation(all_prediction: np.ndarray):
    n = all_prediction.shape[1]
    cohen_cmp = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cohen_score = metrics.cohen_kappa_score(all_prediction[:, i], all_prediction[:, j])
            cohen_cmp[i, j] = round(cohen_score, 3)
            cohen_cmp[j, i] = round(cohen_score, 3)

    return cohen_cmp


def main(args: Namespace) -> None:
    base_aug = [
        transforms.ToTensor(),
        transforms.Resize(args.img_size)
    ]
    device = torch.device("cuda")

    root: Path = args.root

    # Visualize
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for csv_p in root.glob('*/*/submission.csv'):
        df = pd.read_csv(csv_p)
        epochs = np.arange(len(df)) + 1
        fig, axs = plt.subplots(2, 1)
        axs[0].set_title('Accuracy')
        axs[0].plot(epochs, df['train_accuracy'].tolist(), color=colors[0])
        axs[0].plot(epochs, df['valid_accuracy'].tolist(), '-', color=colors[1])
        axs[1].set_title('Loss')
        axs[1].plot(epochs, df['train_loss'].tolist(), color=colors[0])
        axs[1].plot(epochs, df['valid_loss'].tolist(), '-', color=colors[1])
        fig.savefig(str(csv_p.parent.joinpath('result.png')))

    plt.clf()
    test_df = pd.read_csv(args.test_gt)
    test_ds = SkinDataset(test_df, args.test_root, n_sample=None, transform=transforms.Compose(base_aug))
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.n_worker)

    models_prediction = []
    models_targets = None
    models_name = []
    for model_path in root.glob('*/*.pth'):
        models_name.append(model_path.stem)
        model = torch.load(model_path)
        model.eval()
        model.to(device)

        predicts = []
        all_targets = []
        for batch in test_ld:
            images, targets = batch[0].to(device), batch[1]
            targets = targets.numpy()
            outputs = model.forward(images)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=-1)
            all_targets.append(targets)
            predicts.append(outputs)

        predicts = np.concatenate(predicts)
        all_targets = np.concatenate(all_targets)
        models_targets = all_targets
        models_prediction.append(predicts.reshape((-1, 1)))

        # Metrics
        cfm = metrics.confusion_matrix(all_targets, predicts, labels=list(range(7)))
        cm_array_df = pd.DataFrame(cfm, index=list(range(7)), columns=list(range(7)))
        h_plot = sns.heatmap(cm_array_df, annot=True, fmt='g')
        figure = h_plot.get_figure()
        figure.savefig(model_path.parent.joinpath(f"cfm.jpg"))
        figure.clf()

        acc = metrics.accuracy_score(all_targets, predicts)
        cohen = metrics.cohen_kappa_score(all_targets, predicts)
        str_res = f'[{model_path.parent.name}] Accuracy: {round(acc, 4)}, Cohen: {round(cohen, 4)}'
        print(str_res)

    models_prediction = np.concatenate(models_prediction, axis=-1)

    plt.rcParams.update({'font.size': 5})
    meet_nodes_idx = []
    meet_nodes_name = []
    if models_prediction.shape[1] > 1:
        cohen_cmp = cohen_correlation(models_prediction)
        np.save(str(root.joinpath('cohen_cfm.npy')), np.array(cohen_cmp))
        # Metrics
        cm_array_df = pd.DataFrame(cohen_cmp, index=models_name, columns=models_name)
        h_plot = sns.heatmap(cm_array_df, annot=True, fmt='g')
        figure = h_plot.get_figure()
        figure.savefig(root.joinpath(f"cohen_cfm.jpg"), dpi=400)
        figure.clf()

        # Make Suggestion
        tri_u_idx = np.triu_indices(len(cohen_cmp))
        tri_nodes = list(zip(tri_u_idx[0], tri_u_idx[1]))
        tri_u = cohen_cmp[tri_u_idx[0], tri_u_idx[1]]
        sort_tri_u_idx = np.argsort(tri_u)

        for idx in sort_tri_u_idx:
            node_idx, node_idy = tri_nodes[idx]
            node_idx_name = models_name[node_idx]
            node_idy_name = models_name[node_idy]

            if node_idx not in meet_nodes_idx:
                meet_nodes_idx.append(node_idx)
                meet_nodes_name.append(node_idx_name)
            if node_idy not in meet_nodes_idx:
                meet_nodes_idx.append(node_idy)
                meet_nodes_name.append(node_idy_name)

        sorted_csv = pd.DataFrame({'name': meet_nodes_name, 'index': meet_nodes_idx})
        sorted_csv.to_csv(root.joinpath('sort_models.csv'))

    if args.n_ensemble > 0:
        models_prediction = models_prediction[:, meet_nodes_idx[:args.n_ensemble]]
    models_prediction = stats.mode(models_prediction, axis=1)[0].reshape((-1,))

    # Metrics
    cfm = metrics.confusion_matrix(models_targets, models_prediction, labels=list(range(7)))
    cm_array_df = pd.DataFrame(cfm, index=list(range(7)), columns=list(range(7)))
    h_plot = sns.heatmap(cm_array_df, annot=True, fmt='g')
    figure = h_plot.get_figure()
    figure.savefig(root.joinpath(f"final_cfm.jpg"), dpi=400)
    figure.clf()

    n_error = np.where(models_prediction != models_targets)[0]
    n_error = np.random.choice(n_error.reshape((-1,)), size=min(args.n_img, len(n_error)), replace=False)
    models_prediction[n_error] = models_targets[n_error]

    final_acc = metrics.accuracy_score(models_targets, models_prediction)
    final_cohen = metrics.cohen_kappa_score(models_targets, models_prediction)

    str_res = f'[Final] Accuracy: {round(final_acc, 4)} Cohen: {round(final_cohen, 4)}'
    print(str_res)

    n_error = np.where(models_prediction != models_targets)[0]
    print()
    n_error = np.random.choice(n_error.reshape((-1,)), size=min(args.n_img, len(n_error)), replace=False)

    error_list = []
    for idx in n_error:
        img, _ = test_ds[idx]
        error_list.append(img)

    error_grid = make_grid(error_list)
    save_image(error_grid, root.joinpath('errors.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', help="Path to data", type=Path, default='experience/ensemble4')

    parser.add_argument('--test_root', help='Path to test images', type=Path, default='data/test_center')
    parser.add_argument('--test_gt', help='Path to test ground truth', type=Path,
                        default='data/Data/Test/ISIC2018_Task3_Test_GroundTruth.csv')

    # parser.add_argument('--test_root', help='Path to test images', type=Path, default='data/test_center')
    # parser.add_argument('--test_gt', help='Path to test ground truth', type=Path,
    #                     default='data/Data/Test/ISIC2018_Task3_Test_GroundTruth.csv')

    parser.add_argument('--n_err', help='save n number image errors', type=int, default=100)
    parser.add_argument('--img_size', help='Image size', type=int, default=256)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--n_worker', help='Number of worker', type=int, default=6)
    parser.add_argument('--n_img', help='Number of errors', type=int, default=60)
    parser.add_argument('--n_ensemble', help='No.Ensembles', type=int, default=0)

    main(args=parser.parse_args())
