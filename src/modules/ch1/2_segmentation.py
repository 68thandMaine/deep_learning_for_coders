from fastai.vision.all import (
    untar_data,
    URLs,
    SegmentationDataLoaders,
    get_image_files,
    unet_learner,
    resnet34,
)
from pathlib import Path
import numpy as np


def get_camvid_data() -> (Path, list):
    """
    Downloads and returns the CAMVID dataset path and corresponding img files from fastai.
    """
    path = untar_data(URLs.CAMVID_TINY)
    img_files = get_image_files(path / "images")
    return path, img_files


def train_segmentation(path: Path, img_files: list) -> unet_learner:
    """
    Fine tunes a pretrained segmentation model using the CAMVID dataset from fastai.

    Args:
        path (Path): The path to the CAMVID dataset.
        img_files (list): List of image file paths.

    Returns:
        learn: Trained segmentation model.
    """

    dls = SegmentationDataLoaders.from_label_func(
        path,
        bs=8,
        fnames=img_files,
        label_func=lambda o: path / "labels" / f"{o.stem}_P{o.suffix}",
        codes=np.loadtxt(path / "codes.txt", dtype=str),
    )
    learn = unet_learner(dls, resnet34)
    learn.fine_tune(8)
    return learn


if __name__ == "__main__":
    path, img_files = get_camvid_data()
    trained_model = train_segmentation(path, img_files)
    trained_model.show_results(max_n=6, figsize=(7, 8))
