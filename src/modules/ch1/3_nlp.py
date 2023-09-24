from fastai.text.all import (
    TextDataLoaders,
    URLs,
    untar_data,
    text_classifier_learner,
    AWD_LSTM,
    accuracy,
    TextLearner,
)
from pathlib import Path


def get_IMDB_urls() -> Path:
    """
    Get the location of the IMDB urls from fastai.
    """
    imdb_urls = URLs.IMDB
    urls = untar_data(imdb_urls)
    return urls


def trained_model(url_paths: Path) -> TextLearner:
    """
    Fine tune a NLP model from fastai to do do sentiment analysis on movie reviews

    Args:
        url_paths (Path): location of movie review ursl
    """
    dls = TextDataLoaders.from_folder(url_paths, valid="test")
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)
    return learn


if __name__ == "__main__":
    urls = get_IMDB_urls()
    model = trained_model(urls)
    model.predict("I really liked that movie!")
