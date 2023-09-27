# From Model to Production

Now that we have a handle on what we can use deep learning for, and a few examples under our belt let's explore the end-to-end process of creating a deep learning application. It's only through working on your own projects that you will get real experience building and using models aferall!

> While working through the book, try to think about a potential deep learning project of your own that you would like to work on. Just remember, **the most important consideration is data availability**.

### Computer Vision

Computers can recognize items in an image largely as well as humans can. This task is referred to as _object recognition_. When computers use _object recognition_ to analyze where items in an image are and highlight various items within an image it is referred to as _object detection_. Unfortunately models are only as good as the data they are trained on, so if you train an image recognition model with just black and white photos, it might not be very good at making predictions against color photos.

A major challenge to _object detection_ is that labeling images can be slow and expensive. One approach to increase the object detection's efficiency is to synthetically generate variations of input images by rotating them or changing their brightness/contrast. This process is called _data augmentation_.

### Text (Natural Language Processing)

I think we all are aware of the power of ChatGPT, so suffice to say that computers are becoming just as good as humans at classifying, summarizing, and writing text. That being said, as of 2020 (when the book was written), computers are not good at generating correct responses. They might sound confident in their output, but it can oftentimes be misleading.

The saying "the masses are asses" (Alexander Hamilton?) has never been more correct, and in the age of AI this can be a huge problem. Highly compelling AI generated context spread on social media can be used to cause damage at massive scale. Chances are the average person does not fact check what they read online and bad actors can use AI to encourage conflict. Unfortunately, **text generation models will always be technically a bit ahead of models that can recognize artifically generated text**. Hell, in 2023 we heard reports that AI text detectors are impossible, and teachers are trying to find new ways to get their students to not use ChatGPT.

That being said, text generators have been adopted in mass as translators, summarizers, and content generators.

### Combining text and images

We can create a single model that combines texts and images quite easily. A model could be trained on input images with output captions, and learn to generate captions for new images! Be aware though, there is no guarantee that the generated captions will be correct.

### Recommendation systems

Recommendation systems are a special type of tabular data that has a high-cardinality categorical variable which represents users, and another one representing a product of some kind. Using a giant sparse matrix and combining these variables with other kinds of data like language or image is a powerful way to build up a better picture of who a user is, and what they like.

### The Drivetrain Approach

Ensuring that modeling work is useful is something we need to consider. There are many accurate models with no use to anyone just as there are many inaccurate models which are highly useful. The Drivetrain Approach can help us frame this issue by doing the following:

- Start by considering the objective and what actions can be taken to achieve the objective.
- What data do you have or is easily acquired
- Build a model to determine the best actions to take to get the best results in terms of the objective.

**Data is used to produce actionable outcomes. This is the goal of the Drivetrain Approach.**

Once the three goals have been achieved, then begin to build a predictive model. By determining what the objective is, what layers we will need, what data we have and can get access to we will know what "levers" and independent variables we can provide as inputs. The output of the models can be combined to predict the final state for the objective.

## The downside to recommendation systems is that they tell you which products a user might like, rather than which products would be helpful for a user.

## Notes

### Creating your own project

- If you are open to the possibility that deep learning might solve **part** of your problem with less data or complexity than you expect, you can design a process through which you can find the specific capabilities and constraints related to your particular problem.
- Iterate from end to end on a project. Don't spend months fine tuning a model or polishing a UI. Start with the small pieces of your project and build on them. You will undoubtedly learn what the trickiest pieces of building an AI solution are, and which bits make the biggest difference to the final result.
- When thinking about the project you want to do, try not to solve something that has already been done because you won't be able to easily tell if there is something wrong with your code, or if the problem is not solvable with deep learning.

### Computer vision

- There is no general way to check which types of images are missing in your training set, but we can use different strategies to help a model recognize unfamiliar data in production. This unfamiliar data is called _out-of-domain_ data.
- You may be able to take a problem that does not seem like it can be solved with computer vision and do it anyway. We talked a bit about this in Ch1 where sound waves were turned in to images and analyzed with computer vision.

### Text (Natural Language Processing)

- We don’t have a reliable way to, for instance, combine a knowledge base of medical information with a deep learning model for generating medically correct natural language responses.

### Combining text and images

- Not a good idea to use deep learning as a fully automated process. It's safer to include a human to double check the AI to decrease the likelihood of a compelling and incorrect response.

### Tabular data and recommendation systems

- Deep learning models generally take longer to train than random forests or gradient boosting machines.

### Recommendation systems

- Data is represented as a giant sparse matrix which is a table populated by mostly zeros. Usually customers are the rows and products are the columns.

### The Drivetrain Approach

- Take a recommendation system.
  - You will need the objective of driving sales with recommendations of items they would not have purchased. New data will need to be collected to generate recommendations for new sales. This requires many experiments in order to collect data about the wide range of recommendations for a wide range of customers.
  - Next you can build two models for purchase probabilities conditional on seeing or not seeing a recommendation. The difference is the utility function for a given recommendation.

---

## Key Takeaways

---

## Questionnaire

1. Where do text models currently have a major deficiency?

- Text models currently have a major deficiency at generating correct content. Responses may sound compelling, but can be entirely incorrect.

2. What are possible negative societal implications of text generation models?

- Some possible negative implications of text generation models are that they can be used by bad actors to seed dissent, cause societal unrest, and stir up conflict by creating incorrect content.

3. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?

- We can decouple the automation by including a human reviewer during the model training. Humans can assess whether or not a response is correct or not.

4. What kind of tabular data is deep learning particularly good at?

- Deep learning is particularly good at analyzing tabular data that has columns that contain natural language and high cardinality categorical columns. For example a book title, review, zipcode, and product id in one table.

5. What's a key downside of directly using a deep learning model for recommendation systems?

- Recommendation systems do a good job of reporting what products a user will like, rather than what products they might find helpful. The example given uses books and a singular author. If a user likes a particular author, then a recommendation system might recommend more books by the same author as opposed to recommending books in the same genre by different authors.

6. What are the steps of the Drivetrain Approach?

- The steps to the Drivetrain Approach are to create an objective and steps to achieve this objective that AI can solve, identify sources of data that we can use, create models to determine the best actions to take to ge the best results in relation to the objective.
- Next we want to combine the output of these models to predict the final state of the objective.

7. How do the steps of the Drivetrain Approach map to a recommendation system?
8. Create an image recognition model using data you curate, and deploy it somewhere.
9. What is `Dataloaders`?
10. What four things do we need to tell fastai to create `Dataloaders`?
11. What does the `splitter` parameter to `DataBlock` do?
12. How do we ensure a random split always gives the same validation set?
13. What letters are often used to signify the independent and dependent variables?
14. What are the differences between the crop, pad, and squish resize approaches?
15. What is data augmentation? Why is it needed?

- Data augmentation is a way to help models reduce the amount of data they need by making changes to existing data. For example manipulating the properties of an image

16. Provide an example of where the bear classification model might work poorly.
17. What is the difference between `item_tfms` and `batch_tfms`?
18. What is a confusion matrix?
19. What does `export` save?
20. What is it called when we use a model for making predictions, instead of training?
21. What ar IPython widgets?
22. When would you use a CPU for deployment? When might a GPU be better?
23. What are the downsides of deploying your app to a server, instead of to a client device such as a phone or PC?
24. What are three examples of problems that could occur when rolling out a bear warning system in practice?
25. What is out-of-domain data?
26. What is domain shift?
27. What are the three steps in the deployment process?
