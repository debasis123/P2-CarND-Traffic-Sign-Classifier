## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we will use deep neural networks and convolutional neural networks to classify German traffic signs. We will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, we will try out the model on random images of German traffic signs from the web.


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/sample.png "Sample images"
[image2]: ./output_images/train_hist.png "training histogram"
[image3]: ./output_images/test_hist.png "testing histogram"
[image4]: ./output_images/gray.png "gray samples"
[image5]: ./output_images/augmented.png "augmented samples"
[image6]: ./output_images/augmented_hist.png "histogram after training augmentation"
[image7]: ./output_images/lecun.png "LeCun model"
[image8]: ./output_images/web_images.png "Web images"
[image9]: ./output_images/web_resized.png "Web images resized"

---

### Data Set Summary & Exploration

<!-- #### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. -->

I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:
* The size of training set is `39209`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

Traffic signs in the dataset look as follows:

![alt text][image1]

<!-- #### 2. Include an exploratory visualization of the dataset. -->

Following is an exploratory visualization of training data; it is a bar chart showing the frequencies of each sign.

![alt text][image2]

Following is the explanatory visualization of the test data.

![alt text][image3]


### Design and Test a Model Architecture

<!-- #### 3. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.) -->


#### Data Preprocessing

As a preprocessing step on images before we train the model, I normalized and gray-scaled the images. Normalizing means having mean zero and equal variance for all images. Here is a sample of how they look after that.

![alt text][image4]


#### Training Data Augmentation

Since some of the signs have a low frequency, as can be seen in the above histogram, there is a chance that those images will not be predicted with high accuracy. So we augment those images randomly. The rule is to augment an image twice if the `freq(sign)<300` and once if `freq(sign)<400`. We choose the augmentation step randomly from one of the five methods: rotation, translation, affine transformation to the left and right, and zooming. We use opencv apis for that. This raises the min frequency for a sign/class to `420` from `210` and the median from `600` to `960`.

Following is a sample of applying data augmentation on an image.

![alt text][image5]

Here is the new historgram for the frequencies of the signs in training data.

![alt text][image6]


<!-- #### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model. -->


#### Model Architecture

I used modified LeNet architecture involving convolutional neural networks from Sermanet/LeCunn traffic sign classification paper to build my convolutional model. Here is the summary of the model.
```
Layer 1: Convolutional.     Input = 32x32x1.    Output = 28x28x6.   Activation = relu
MaxPooling.                 Input = 28x28x6.    Output = 14x14x6
Layer 2: Convolutional.     Input = 14x14x6.    Output = 10x10x16.  Activation = relu
MaxPooling.                 Input = 10x10x16.   Output = 5x5x16
Layer 3: Convolutional.     Input = 5x5x16.     Output = 1x1x400.   Activation = relu
Flatten convolution.        Input = 5x5x16.     Output = 400
Flatten convolution.        Input = 1x1x400.    Output = 400
Concatenate above two layers.                   Output = 800
Dropout regularization
Layer 4: Fully Connected.   Input = 800.        Output = 43.
```

and a graphic representation from the paper.

![alt text][image7]


#### Training / Validation split

Now, we split the training dataset into training and validation sets so as to validate the model while training. This is required to keep the model from overfitting. Number of validation examples is `2387` and number of training examples is `45342` after this split.


<!-- #### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. -->


#### Hyperparameters

I choose the following hyperparamters to train the model, after some experimentation. We choose the number of epochs high here so as to infer the right number of epochs before the overfitting starts.
```python
number of epochs = 50
batch size = 128
learning rate = 0.001
keep_prob for dropout layer = 0.8
L2 regularization factor = 5e-4
```

#### Optimizer and loss function

I used Adam optimizer with learning rate 0.001 for quickly converging to the lowest training and validation loss. `softmax_cross_entropy()` from tensorflow is used as the loss function.


<!-- #### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. -->

#### Training, validation, and test

Finally, I trained my convolutional model with the above hyperparameters and epochs = 30 seemed to be a sweet spot where the overfitting started. At this point I had
* training set accuracy of `0.998`
* validation set accuracy of `0.981`.

So I chose this value for the final training of the model on the entire training + validation data. The training accuracy did not change, but I had
* test set accuracy of `0.921`.



<!-- If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? -->

<!-- If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? -->



### Test a Model on New Images

<!-- #### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. -->

Here are ten German traffic signs that I found on the web:

![alt text][image8]

The images are not of the same shape. So I resized those images first to the size `(32x32x3)` as required by our pipeline. They look as follows after that.

![alt text][image9]

Then I ran the same preprocessing pipeline, namely grayscaling and normalizing, to those images, as I did for the training the model. The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of `40%`. This does compare favorably to the accuracy on the test set which was `92%`.

The top 5 probabilities and predicted classes for each image (in the same order as above, left to right) is as follows:
```
| Image                   |  Predicted Probabilities                    |  Predicted Class   | Correct class |
|:-----------------------:|:-------------------------------------------:|-------------------:|--------------:|
| Speed limit (30km/h)    |  [25.6669 24.0977 20.335  17.501  13.6295]  |  [22 18 24 14 37]  |  1            |
| Children crossing       |  [31.4411 27.8007 26.5787 26.0859 21.2881]  |  [24 27 26 20 11]  |  28           |
| Stop                    |  [18.9173 12.1106 10.7846 10.6355  9.0884]  |  [14 33 20 39 34]  |  14           |
| Roundabout mandatory    |  [13.2608 11.2747  9.9937  9.2092  8.3866]  |  [21 40  0 12 28]  |  40           |
| Turn right ahead        |  [14.2172 10.324   9.9464  9.0774  7.4673]  |  [33 11 34 28  1]  |  33           |
| Speed limit (70km/h)    |  [32.396  16.2778 15.2482 14.5263 14.3444]  |  [22 39 21 14 20]  |  4            |
| No entry                |  [43.616  20.0873 13.91   13.7417 13.3675]  |  [17 14 33 36 34]  |  17           |
| No passing              |  [37.6866 27.0559 22.2123 20.3657 17.2654]  |  [ 9 41 16 28 10]  |  9            |
| Slippery road           |  [30.0859 29.4393 24.771  20.5017 19.6111]  |  [18 24 29 11 26]  |  23           |
| Speed limit (100km/h)   |  [15.6811 14.0558 13.7544 13.7401  8.0946]  |  [18 27 26 40 12]  |  7            |
```

As can be seen above, Stop, Turn right ahead, No entry, and No passing were detected correctly, and Roundabout mandatory came a close second. The reason these images did better than others is because the entire image consists of the signs only, whereas for other images, they sometime less than 50 % of the image size. This tells that cropping the images before throwing them to the pipeline could be a better option.

Second, the resizing really distored some of the images and could be a reason for not doing good in the prediction. More careful resizing techniques could be other option.

<!-- The first image might be difficult to classify because ... -->

<!-- #### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). -->

<!-- Here are the results of the prediction:

| Image                 |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| Stop Sign             | Stop sign                                     |
| U-turn                | U-turn                                        |
| Yield                 | Yield                                         |
| 100 km/h              | Bumpy Road                                    |
| Slippery Road         | Slippery Road                                 | -->



<!-- #### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts) -->

<!-- The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. -->

<!-- For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability           |     Prediction                                |
|:---------------------:|:---------------------------------------------:|
| .60                   | Stop sign                                     |
| .20                   | U-turn                                        |
| .05                   | Yield                                         |
| .04                   | Bumpy Road                                    |
| .01                   | Slippery Road                                 |


For the second image ... -->

<!-- ### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details) -->
<!-- #### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications? -->
