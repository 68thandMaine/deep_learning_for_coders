# Deep Learning for Coders with Fastai and Pytorch Dump

This repository contains tutorials and a personal blog written while going through the content in [Deep Learning for Coders with Fastai and Pytorch](https://course.fast.ai/Resources/book.html). The book covers a variety of subjects, and upon completion readers should know enough to build and train their own AI models.

## Requirements and Installation Instructions

- This dependencies for this project are managed with Python Poetry. Poetry helps create an isolated environment by creating a lockfile that has all the dependency versions locked.

- One thing to note is that you will need access to a GPU in order to run any of the code in this repository.

## Directory Structure

The content of each chapter in the book is located under `src/modules`. Each chapter in the book has a directory within modules, and each tutorial within the chapter is represented by a python script:

```
- src
  |_ modules
     |_ ch1
     |  |_ 1_classification.py
     |_ ch2
```

## Design Decisions

- As a software engineer by trade, I have opted to not use the Juypter Notebook approach and instead write python modules.
- To visualize the content of the tutorials I am using [Gradio](https://www.gradio.app/guides/quickstart).

## Blog

- In the Blog directory of this repository you can find the notes that I've taken as I complete each chapter of this book.
