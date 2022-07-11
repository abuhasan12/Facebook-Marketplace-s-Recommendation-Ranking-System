# Marketplace Text and Image Classifier

A multi-modal text and image classifier that mimics the process behind facebook marketplace for identifying the category of product images uploaded by the user and descriptions given.
<br/>
Use-case: A customer would search the marketplace for an item and the model will return the closest matching items.

## Demo

![demo](https://github.com/abuhasan12/Facebook-Marketplace-s-Recommendation-Ranking-System/blob/main/demo/Demo.gif)

## Docker

To run:
```Command Line
$ docker run --name ml-container -p 8080:8080 abuh12/fb-mp-ml
```
<br/>

To test:
<br/>
Navigate to [host]:8080/docs
<br/>
e.g. localhost:8080/docs
<br/>
<br/>

To stop:
```Command Line
$ docker stop [CHOSEN_CONTAINER_NAME]
```

## The Model

The model is composed of layers of a text classifier and an image classifier.
<br/>
<br/>
The text classifier is made up of 1D convolutional layers, each succeeded with rectified linear activation functions (ReLU layers).
<br/>
<br/>
The image classifier uses a pretrained ResNet-50 model.
<br/>
<br/>
The networks of both are then flattened before configured to output the same size for easy concatenation.
Then the final layers of each classifiers are concatenated for the combined model which has an output size of the number of product categories (13).

## Build

Building the classifiers were done using Pytorch.

### Getting Familiar with CNN Architecture

Before creating the combined model, an initial attempt to create a custom image classifier was taken to learn more about convolutional neural networks.
<br/>
<br/>
The first task was to create a Pytorch Dataset class. The Dataset reads a Pandas dataframe containing image arrays and their categories, and it allows for transformations to be applied to the image arrays. To get the features and labels from the Dataset, they are returned as Torch Tensors (the data-types used for Pytorch models).
<br/>
<br/>
Next, the CNN model was constructed using multiple convolutional 2d layers, rectified linear activation layers in-between, and multiple dropout layers for regularisation.
<br/>
<br/>
To train the model, stochastic gradient descent was the chosen optimiser with a learning rate of 0.1. The loss calculated for evaluation was Cross Entropy.
<br/>
Due to the purpose of the task being to have an initial attempt at creating a CNN, there was no attempt at fine-tuning the model.

### Image Classifier

For the final image classifier model, I leveraged transfer learning, by using a pretrained ResNet-50 model (from torchvision library).
<br/>
The fully connected layer was redesigned to a linear layer outputting the size of the number of categories of the products (13).
<br/>
<br/>
Training was done using an SGD optimiser again, but this time with a learning rate scheduler that decayed the learning rate every X number of epochs. Loss was measured using Cross Entropy. During training, a tensorboard graph visualised the change in training and validation accuracy, and the training method returning the model with the best weights and biases (using Pytorch's state_dict).
<br/>
<br/>
The datasets were created using the images in a folders of their respective categories. Torchvision's datasets.ImageFolder creates tensor objects of these images (converted to arrays) and automatically sets their labels as the category of the folder they were in. The image transormation is also done in this step.
<br/>
<br/>
Finally, the model was trained to reach a somewhat acceptable validation accuracy (66%) after fine-tuning the hyper-parameters for multiple runs (compared using Tensorboard). The optimal hyper-parameters were:
<ul>
<li>Validation Split: 0.2</li>
<li>Batch Size (Training and Validation): 32</li>
<li>SGD Learning Rate: 0.002</li>
<li>SGD Momentum: 0.93</li>
<li>LR Scheduler Step Size (X number of epochs): 5</li>
<li>LR Scheduler Decay: 0.2</li>
</ul>

### Text Classifier

For the text classifier, I also used a convolutional architecture comprised of the hidden convolutional layers described earlier (combined model).
<br/>
The Dataset class was constructed to take the Product Description of the Products dataframe return them vectorise using BERT (BertTokenizer and BertModel), with their labels (categories encoded).
<br/>
<br/>
The training was done using an Adam Optimser instead of SGD which proved to output better accuracy in this case. Loss was again measured using Cross Entropy, with training optimised by the use of Tensorboard for visualisation and saving the best weight and biases using Pytorch's state_dict.
<br/>
<br/>
The model was trained multiple times until the accepted accuracy was 83% and the optimal hyper-parameters as:
<ul>
<li>Validation Split: 0.2</li>
<li>Batch Size (Training and Validation): 32</li>
<li>Adam Learning Rate: 0.001</li>
</ul>

### Combined Model

Finally, the combined model could be created and trained. The architecture of the Image Classifier and Text Classifier was constructed and adjusted (removing and adding layers, trying out dropout for regularisation etc.) to return the best achievable accuracies on their validation sets.
<br/>
<br/>
The combined model makes use of both these archictures, with only their output layers adjusted to output the same size (128) for concatenation. Then, the true output layer was added to complete the combined classifier. The inputs (an image and associated text) would go through the appropriate initial architecture (image through the image layers and text through the text layers) and then the outputs through the final year concatenated.
<br/>
<br/>
The Datasets were loaded as a combination of their respective methods used in the previous tasks.
<br/>
<br/>
The chosen optimiser for training was SGD along with a step scheduler for learning rate decay. Loss was measured with Cross Entropy.
<br/>
<br/>
The accuracy score achieved was 65% with the following hyper-parameters:
<ul>
<li>Validation Split: 0.2</li>
<li>Batch Size (Training and Validation): 32</li>
<li>SGD Learning Rate: 0.001</li>
<li>SGD Momentum: 0.93</li>
<li>LR Scheduler Step Size (X number of epochs): 4</li>
<li>LR Scheduler Decay: 0.2</li>
</ul>

## Simple Machine Learning Models Build

Initially, more simpler models were designed (Linear Regression & KNN Classifier) to explore the datasets and get familiar with machine learning models.

### Data

Training data was provided of products in a csv file (/data/Products.csv). Data included were:
<ul>
<li>Product ID</li>
<li>Product Name</li>
<li>Category</li>
<li>Product Description</li>
<li>Price</li>
<li>Location</li>
<li>URL</li>
<li>Page ID</li>
<li>Create Time</li>
</ul>

/data_cleaning/clean_tabular.py provides a function `clean` of the class `CleanTabular` that performs a number of data cleaning tasks specific for the Products.csv file (and any other similar file).
```python
from data_cleaning.clean_tabular import CleanTabular
clean_products = CleanTabular("./data/Products.csv")
clean_products.clean()
clean_products.df       # dataframe after data cleaning
```

`clean()` performs the following operations:
<ol>
<li>Drops nulls</li>
<li>Cleans the Product Name field of each instance to read more tidy</li>
<li>Cleans the Price field of each instance to remove pound sign (£) and comma for thousand (,). It also converts the price from string to float</li>
<li>Cleans text for Product Name, Product Description and Location by removing unwanted characters, decapitalising, performing lemmatisation and removing stop words.</li>
</ol>
<br/>

The images of the products were downloaded from an AWS S3 bucket, using the links provided in /data/Images.csv. Each image has an ID and the Product ID associated with it (some products have multiple images), as well as the S3 URL to download it from, image_ref and create_time.
<br/>
After downloading the images to a folder, data_cleaning/clean_images.py provides a function `clean_and_save` of the class `CleanImages` that resizes images to 512 by 512 and saves the results in a new directory called resized_images.
```python
from data_cleaning.clean_images import CleanImages
clean_images = CleanImages("images/")       # images folder
clean_images.clean_and_save()       # results in new folder called resized_images
```

### Regression and Classification

In the ml_models folder there are attempts at using simple machine learning models to build a price predictor and an image classifier. These generated poor accuracy scores but were used to gain familiarity creating models.

#### Price Predictor - Linear Regression

First, the Products.csv file was cleaned using the cleaning method mentioned above. The features chosen for the predictor variables were Product Name, Product Description, and Location. The target variable obviously being price.
<br/>
A column transformer was constructed to vectorise the features using TF-IDF, and the column transformer incorporated into an SKLearn pipeline, completed with the Linear Regression model.
GridSearchCV helped with tuning the hyper-parameters of the pipeline.

#### Image Classifier - K-Nearest Neighbour

For the image classifier, the images needed to be first converted to multi-dimensional arrays. This was done with `imread` from `skimage.io`.
<br/>
Combining these arrays with Images.csv and Products.csv, the arrays could be mapped to their categories. All other columns were dropped.
<br/>
The image arrays were flattened and scaled before fitting to the KNN model by SKLearn (using GridSearchCV to fine-tune the hyper-parameters).

#### Evaluation

As mentioned above, the accuracy scores of both models were really bad due to the complexity of the data. The true purpose of this project was to explore deep neural networks for the use-case.