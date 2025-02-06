from argparse import Namespace, ArgumentParser
from pathlib import Path

import shutil


def main(args: Namespace) -> None:
    """

    :param args:
    :return:
    """
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    rename_idx = len(list(dest.glob('*')))

    for root_path in args.list:
        for dir_path in root_path.glob('*'):
            bs_save = dest.joinpath(f'Model-{rename_idx + 1}')
            for model_path in dir_path.glob('*.pth'):
                bs_save.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(model_path), bs_save.joinpath(model_path.name))

            for asset_path in dir_path.glob('assets'):
                bs_save.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(asset_path), str(bs_save.joinpath('assets')))

            if bs_save.exists():
                rename_idx += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dest', help='Destination Directory', type=Path)
    parser.add_argument('--list', nargs='+', help='merge directory into the source', type=Path)
    main(args=parser.parse_args())
