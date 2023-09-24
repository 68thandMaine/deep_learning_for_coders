# Your Deep Learning Journey

First thing is first - what is a _dataset_? Well, it's simply a bumch of data - it could be images, emails, financial data, biological data. Pretty much any sort of data that can be assembled into a collective group that our models will be trained with.

In the first example of the book we write code that is capable of identifying whether or not a picture contains a cat or a dog - commonly called a **classifier**. This is done by using the **Oxford-IIT Pet Dataset** and a pretrained model.
Finally, we fine tune the pretrained model to create a custom model for recognizing dogs and cats.

The code is incredibly simple:

```python
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224)
    )

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

#### Line by line breakdown of this code

```python
from fastai.vision.all import *
```

- This gives us all of the functions and classes we will need to create a wide variety of computer vision models.

```python
path = untar_data(URLS.PETS)/'images'
```

- Downloads the dataset, and returns a [PATH object](https://docs.python.org/3/library/pathlib.html) with the extracted location. This is a really useful class from the Python 3 standard library that makes accessing files and directories much easier.

```python
def is_cat(x): return x[0].isupper()
```

- Pretty simple function that labels cats based on a filename rule provided by the dataset curators.

```python
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224)
  )
```

- Tell Fastai what kind of dataset we are using, and how it is structured. Note that we use the `ImageDataLoaders` class to identify the type of data we are using. In Fastai, usually the first part of a classname will be the type of data we have.
- We also tell Fastai how to get data labels from the dataset by passing it the label function we used earlier.
- We define the `Transform` that we need. `Transform` is code that is applied automatically during training. In Fastai there are two types:
  - `item_tfms` which are applied to each item.
  - `batch_tfms` which are applied to a _batch_ of items using the GPU.
- One of the most important pieces of this code is the `valid_pct=0.2`. This creates the validation set by setting aside 20% of the data. The other 80% is used during training. We use the validation set to measure the accuracy of the model
- The parameter seed=42 sets the random seed to the same value every time we run this code, which means we get the same validation set every time we run it—this way, if we change our model and retrain it, we know that any differences are due to the changes to the model, not due to having a different random validation set.

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```

- Tell Fastai to create a _convolutional neural network_ and specifies the architecture to use.
- `resnet34` is a standard architecture. The number in `resnet34` refers to the number of layers in this variant of `resnet`. Modles with more layers take longer to train and are prone to overfitting, but when using more data they are more accurate.

```python
learn.fine_tune(1)
```

- This line tells Fastai how many epochs to train the model. We call it `fine_tune` as opposed to `fit` because Fastai has a method called `fit` that is used when training a model from scratch. In the case of our classifier we are using pretrained models.
- `fine_tun` performs two steps in the default form:
  1. Use one epoch to fit the parts of the model necessary to get the new random head to work with the dataset.
  2. Use the number of epochs to fit the entire model, updating the weights of the later layers faster than earlier layers.

---

Once we have fine tuned the model, we can begin to provide our model with pictures of cats and dogs.

```python
img = PILImage.create(uploader.data[0])
is_cat, _, probs = learn.predict(img)
```

!!! Note that we need a way to capture image data still

---

Checking to see if the code is "any good" involves investigating the _error rate_. This value provides a metric (measure of model quality) and identifies the proportion of data that was correctly identified.

What are some of the basic concepts involved in machine learning?

- The idea of a "weight assignment".
- Every "weight assignment" has some actual performance.
- There must be an automatic means of testing the performance.
- There must be a mechanism (another automatic process) for improving the performance by updating the weight assignments.

Weights are just variables, and weight assignments are particular choices of values for those variables. A program's weight assignment define how the program will operate. Once a model has been trained, the weights become a part of the model.

The training loop of a model essentially gives models weights and inputs. The model then produces results, and we can measure the performance of those results. If we are not happy with the performance, then we can use the values we measured to "tweak" the weights. This produces a bit of a loop between the model output and the model input.

A **neural network** is a kind of function that is so flexible that it can be used to solve any given problem just by varying the weights. In order to do this, we use _stochastic gradient descent_ (SGD) which is just a fancy word for the mathematical process of choosing values to update weights.

In the cat and dog classifier, the images we provide the model are the "inputs", the weights are predefined within the model, the model is a neural network, and our results are the values that are calculated (cat vs dog).

There are two main types of models in machine learning that are investigated in this book. A classification model, like the cats and dog one, attempts to predict a class or category. A regression model attempts to predict one or more numeric qualities such as temperature or a location.

When we train a large enough model for a long time, it will memorize the label of every item in the training dataset. This is why it is important to save at least 20% of the inital dataset for validation. Validation set accuracy can also improve for a while, but it will begin to get worse as the model memorizes training data. This is a prime example of overfitting.

**Overfitting is the single most important and challenging issue when training**.

Models "learn" by passing data through each layer of their neural net. Each layer is responsible for a reconstructing some feature that is required for an overarching goal. For example, a computer vision model might have a layer for understanding diagonal lines, another layer for identifying corners, repeating lines, circles, and other simple patterns, and another layer for higher semantic components such as car wheels, cats, dogs, text, etc. As data is passed through each layer, the layers become more and more higher level.

One of the cool things about working with computer vision is that we can use it for a wide array of tasks. Sound can be converted into a wave, which can then be identified through computer vision. The same can be done with anything that produces an image of any sort. This could potentially be a great usecase for scientific instruments that produce images for analysis.

Something to consider while working with AI, is that AI is a prediction machine. It can only make recommendations - not actions. If we provide our models with poor data and a model learns to identify patterns from this data, then it will produce inaccurate predictions. That's why we need to have quality data when training models.

As humans in charge of creating AI, it is our responsibility to curate the data that we provide our models. For example, if we act on data predicted by a model, and update our model with new data based on these predictions - we will create a positive feedback loop. Essentially the more the model is used, the more biased it can become. We don't like bias.

## Segmentation Models

> Please see `/src/modules/ch1/2_segmentation.py`

Although not discussed as thoroughly as the classifier model, we also create a pretrained model for localizing objects in a picture. The task of recognizing the content of every individual pixel in an image is called segmentation. This concept is a powerful tool for creating self driving cars because in this cse it helps the model understand what is a pedestrian vs a static object.

## NLP Models

> Please see `/src/modules/ch1/3_nlp.py`

We can use Natural Language Processing (NLP) to analyze comments, label words in a sentence, and much more. The tutorial provided uses a pretrained model to classify the sentiment of a movie review by using the IMDb Large Movie Review dataset. It can work with a large amount of text (also called tokens). Using the `predict` method from the model will return several items

```python
('pos', tensor(1), tensor([0.0041, 0.9959]))
```

- `pos`: Here we can see the model has considered the review to be positive.
- `tensor(1)`: The index of “pos” in our data vocabulary.
- `tensor([0.0041, 0.9959])`: The probabilities attributed to each class (99.6% for “pos” and 0.4% for “neg”).

## Models with Tabular Data

Tabular data is that which is in the form of a table like a csv. We can create models that try to predict one column of a table based off of information in other columns of the table.

In the code tutorial we need to tell Fastai which columns are categorical versus continuous. In the example we build a tool that can predict whether or not someone is a high-income earner based on socioeconomic background. We use values that represent a discrete set of choices for categorical data, and values like age for continuous data.

In general, there are not very many pretrained models for any tabular modeling class, so we will use a new method called `fit_one_cycle` for training Fastai models from scratch.

The rest of the examples in the chapter iterate on how to work with tabular data. One thing to note is that we might want to control the range of numbers for continuous data. To do this we can use the `y_range` parameter.

## Validation Sets and Test Sets

To make sure that models predict well on new data we split our datasets into two sets: the training set, and the validation set.

The training set is what we see in training, and up to this point in the book is defined as 80% of the initial dataset. The validation dataset is also known as the development set lets us test that the model learns lessons from the training data.

Even though the validation set is kept separate from the training set, there is still a risk of overfitting through human trail/error and exploration. As modlers, our job is to explore different model choices or hyperparameters to produce the best outcome. Subsequent versions of a model are shaped by modlers seeing the validation code - so it stands to reason we can be a source of overfitting. To combat this we can use a test set.

Test sets are pieces of data that are held from the researcher. This means that out of a dataset, some is used for training, and the data that is witheld in trainging is used for validation. The data that is witheld from modelers is the test set. We cannot use the test set to improve the model - its sole purpose is to assess the accuracy of our model at the very end of our efforts. To summarize: training data is fully exposed, validation data is less exposed, and test data is totally hidden.

Both the test and validation set should have enough data to ensure we create a good estimate of the accuracy. A key property of both sets is that they must be representative of new data that we can swee in the future. Selecting what data belongs in what set can involve more than a random grab of the original dataset. Sometimes (especially with time series data) if we choose a random subset of previous data, we aren't doing ourselves any favors because it is too easy to fill in the gaps. A better strategy would be to use earlier data for training, and the later data for validation. Essentially you must consider the task at hand when creating your subclasses of datasets.

---

## Questionnaire

1. Do you need these for deep learning:

- Lots of math = False
- Lots of data = False
- Lots of expensive computers = False
- A PhD = False

2. Name five areas where deep learning is now the best tool in the world

- Medicine (Designing Drugs)
- Language Processing (ChatGPT)
- Computer Vision (self driving cars)
- Recommendation Systems (Netflix, Amazon)
- Robotics

3. What was the name of the first device that was based on the principle of the artificial neuron?

- The Mark I Perceptron

4. Based on the book of the same name, what are the requirements for parallel distributed processing (PDP)?

- A set of processing units
- A state of activation
- An output function for each unit
- A pattern of connectivity among units
- A propagation rule for propagating patterns of activities through a network of conductivities.
- An activation rule for combining th einputs impinging on a unit with the current state of that unit to produce an output for the unit.
- A learning rate whereby patterns of connectivity are modified by experience.
- An environment within which the system must operate.

5. What were the two theoretical misunderstandings that held back the field of neural networks?

- There was a belief that a single layer of neurons could appoximate any mathematical functions.
- Not using multiple layers of devices to allow learning limitations to be addressed / lack of understanding of deep architectures.

6. What is a GPU?

- A Graphic Processing Unit or graphics card is a special kind of processor in a computer that can handle thousands of single tasks at the same time.

7. Why is it hard to use a traditional computer program to recognize images in a photo?

- Traditional computer programs have a difficult time recognizing images in a photo because they cannot learn. They cannot self correct.

8. What did Samuel mean by "weight assignment"?

- Weight assignments are variables that define how a program will operate

9. What term do we normally use in deep learning for what Samuel called "weights"?

- Weights are simply parameters.

10. Why is it hard to understand why a deep learning model makes a particular prediction?

- The complexity of a deep network and the nature the inner workings of a model not being easily interpretable are some reasons why it's hard to understand why a model makes a prediction.

11. What is the name of the theorem that shows that a neural network can solve any mathematical problem to any level of accuracy?

- The name of the theorem that shows that a neural network can solve any mathematical problem wo any level of accuracy is the universal approximation theorem.

12. What do you need in order to train a model?

- To train a model you need data.
- You need a way to update your weights using the results of a prediction.

13. How can a feedback loop impact the rollout of a predictive policing model?

- A positive feedback loop can occur if a model predicts areas where arrests happen, police record arrests in this area, and then use the records as new data for training again. This will cause the model to get really good at predicting arrests in one area.

14. Do we always have to use 224x224 pixel images with the cat recognition model?

- This is the standard size for historical reasons. If you increase the size you might get better results, but at the price of speed and memory consumption.

15. What is the difference between classification and regression?

- Classification is used to assign labels to things, and regression is used to predict numerical data.

16. What is a validation set? What is a test set? Why do we need them?

- A validation set is a portion of the initial dataset that is hidden from model training. A test set is a portion of data that is completely hidden from training and validation.
- We need both these sets of data to prevent overfitting. This occurs when a model memorizes the answeres to its training data, or we end up making so many tweak to training with the validation data that the model cannot accurately predict new input.

17. What will fastai do if you don't provide a validation set?

- Fastai will attempt to create a validation set from the training data if it is not explicitly given a validation set.

18. Can we always use a random sample for a validation set? Why or why not?

- We cannot always use a random sample for the validation set because random sampling can make predictions too easy. In a time series if we use random sampling the pattern between known and unknown data is too easy to discover. Therefore, it is best to purposely choose an earlier portion of your data for training, and a later portion for validation.

19. What is overfitting? Provide an example.

- Overfitting occurs when a model learns the data that it is given in training so well that it can no longer predict data it has not seen before.

20. What is a metric? How does it differ from loss?

- A metric is a function that measures the quality of the models predictions using the validation set, and will be printed at the end of each epoch. A loss is a measure of the accuracy of the model that is used by the program to adjust weights.

21. How can pretrained models help?

- Pretrained models come with their own weights which prevents the need for us to run training.

22. What is the "head" of a model?

- The head of the model is the part that is newly added to be specific to the new dataset.

23. What kinds of features do the early layers of a CNN find? How about the later layers?

- The early layers of a convolutional neural network learn to identify basic shapes and patterns. For example one layer might learn to identify specific types of lines, and another layer might use that data to understand the edges of shapes.
- Later layers use the data from the previous layers to learn higher semantic things such as car tires, types of signs, types of plants.

24. Are image models only useful for photos?

- Image models can be used in an array of fields that are not just for photos. Any data that can be represented as a picture - be it sound waves, heatmap data, or patterns created by measuring output can be run through an image model.

25. What is an architecture?

- An architecture is the internal of a model or the actual mathematical function that receives input data and hyperparameters.

26. What is the `y_range` used for? When do we need it?

- When working with continuous data we can use the `y_range` to limit our range of acceptable data.

27. What are hyperparameters?

- Hyperparameters are the various inputs that are given to a model.

28. What's the best way to avoid failures when using AI in an organization?

- Having data that a vendor or model developer never sees that we can use to double check the accuracy of a model is a great way to avoid failures. We can define our own metrics based on what actually matters and decide if the model is adequate.

---

## Deep Learning Jargon

- The functional form of a model is referred to as it's architecture.
- Weights are a type of parameter.
- The results of a model are called predictions.
- The predictions are calculated from independent variables (data without labels).
- The measure of performance is called loss.

| Term             | Meaning                                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Label            | The data that we’re trying to predict, such as “dog” or “cat”                                                                                        |
| Architecture     | The template of the model that we’re trying to fit; i.e., the actual mathematical function that we’re passing the input data and parameters to       |
| Model            | The combination of the architecture with a particular set of parameters                                                                              |
| Parameters       | The values in the model that change what task it can do and that are updated through model training                                                  |
| Fit              | Update the parameters of the model such that the predictions of the model using the input data match the target labels                               |
| Train            | A synonym for fit                                                                                                                                    |
| Pretrained model | A model that has already been trained, generally using a large dataset, and will be fine-tuned                                                       |
| Fine-tune        | Update a pretrained model for a different task                                                                                                       |
| Epoch            | One complete pass through the input data                                                                                                             |
| Loss             | A measure of how good the model is, chosen to drive training via SGD                                                                                 |
| Metric           | A measurement of how good the model is using the validation set, chosen for human consumption                                                        |
| Validation set   | A set of data held out from training, used only for measuring how good the model is                                                                  |
| Training set     | The data used for fitting the model; does not include any data from the validation set                                                               |
| Overfitting      | Training a model in such a way that it remembers specific features of the input data, rather than generalizing well to data not seen during training |
| CNN              | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks                                        |

## Key Takeaways

- Fastai has a bunch of different modules used for different deep learning types. In this chapter we used the following modules to access classes specific to the types of jobs required for each module:
  - Tabular module
  - Text module
  - Vision module
