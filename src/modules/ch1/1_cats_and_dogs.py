from fastai.vision.all import (
    untar_data,
    ImageDataLoaders,
    cnn_learner,
    get_image_files,
    Resize,
    PILImage,
    URLs,
)
from fastai.metrics import error_rate
from fastai.vision.models import resnet34


path = untar_data(URLs.PETS) / "images"


def is_cat(x):
    return x[0].isupper()


def train_pet_classifier():
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        seed=42,
        label_func=is_cat,
        item_tfms=Resize(224),
    )

    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    return learn


if __name__ == "__main__":
    trained_model = train_pet_classifier()

    img = PILImage.create(uploader.data[0])
    is_cat, _, probs = trained_model.predict(img)
