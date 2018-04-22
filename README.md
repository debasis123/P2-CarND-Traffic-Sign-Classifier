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
[image7]: ./output_images/lecunn.png "LeCunn model"
[image8]: ./output_images/web_images.png "Web images"
[image9]: ./output_images/web_resized.png "Web images resized"
[image10]: ./output_images/train_test_hist.png "training-testing histogram"

---

### Data Set Summary & Exploration

Please look at the `Data Set Summary & Exploration` section in the jupyter notebook for the implementation.

I used the numpy and pandas libraries to calculate summary statistics of the traffic signs data set:
* The size of training set is `39209`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

Traffic signs in the input dataset look as follows:

![alt text][image1]

Following is the statistics on the training and testing data.
```
| Data      | Min frequency   | Max frequency | Mean     | Median     | Std. Dev.  |
|-----------|-----------------|---------------|----------|------------|------------|
| Training  | 210             | 2250          | 911.84   | 600.0      | 687.72     |
| Testing   | 60              | 750           | 293.72   | 180.0      | 230.71     |
```
Following is an exploratory visualization of training data and test data; it is a bar chart showing the frequencies of each sign.

![alt text][image10]


### Design and Test a Model Architecture

My pipeline consisted of the following steps in the same order:
* Data preprocessing
* Training data augmentation
* Defining model architecture (convolutional + Maxpooling + Dropout + Activation)
* Splitting training data into training and validation data
* Choosing optimizer and loss function
* Training the model in batches of data while tuning hyperparameters
* Output train, validate, and test accuracy

Now, I explain how I performed each of the above steps. Please look at the `Design and Test a Model Architecture` section and corresponding subsections in jupyter notebook for the implementation for each of them.


#### Data Preprocessing

The first step was to preprocess the data before they are fed into the model. This is required to avoid having various ranges of pixel values in input images (as will be evident in test images from web) and thus converging by the model will be difficult. My preprocessing consisted of two steps in the following order:
* grayscaling the image: This removes unncessary noises in the images in general and uniformizes various types of images (RGB, BGR, ...) that may come from the web. It also helps to train faster.
* normalizing the image: Normalizing usually means having mean zero and equal variance for all images. My normalization step was just to divide the pixels by 255, so that each value is in the range `[0,1]`. Following was the result before and after the normalization step.
```
| Data          | Mean Before Preprocessing  | Mean After Preprocessing  |
|---------------|----------------------------|---------------------------|
| Training      |  82.67                     |  0.32                     |
| Testing       |  82.15                     |  0.32                     |
```

Here is a sample of the images after the preprocessing on some images.

![alt text][image4]


#### Training Data Augmentation

Since some of the signs have a low frequency, as can be seen in the above histogram, there is a chance that those images will not be predicted with high accuracy and the model will be biased towards signs with higher frequency. So the next step was to augment the images with lower frequency so that they are well represented in the training set. The idea was to augment an image twice if the `freq(sign)<300` and once if `freq(sign)<400`, and so on. I chose the augmentation step randomly from one of the five well-known methods: rotation, translation, affine transformation to the left and right, and zooming. I used opencv APIs for that.

This raises the min frequency for a sign/class from `210` to `420` and the median from `600` to `960`.

Following is a sample of applying data augmentation on an image.

![alt text][image5]

Here is the comparison between the old and new training dataset.
```
| Data          | Min frequency   | Max frequency | Mean     | Median     | Std. Dev.  |
|---------------|-----------------|---------------|----------|------------|------------|
| Old Training  | 210             | 2250          | 911.84   | 600.0      | 687.72     |
| New Training  | 420             | 2250          | 1109.98  | 960.0      | 550.38     |
```

and the historgram showing the frequencies of the signs in the augmented training data.

![alt text][image6]



#### Model Architecture

LeNet model was my starting point. I tried the LeNet architecture without any data augmentation and found that the validation accuracy was below 90%. As mentioned in the Sermanet/LeCunn paper, their new model works better than the original LeNet architecture involving convolutional neural networks for the purpose of traffic sign classification. So I used the their modified LeNet architecture for my project. As mentioned in the paper, dropout and maxpooling are important layers to avoid the overfitting of the model. Here is the summary of the model.
```
Layer 1:
5x5 Convolutional.          Input = 32x32x1.    Output = 28x28x6.   Activation = relu
2x2 MaxPooling.             Input = 28x28x6.    Output = 14x14x6
Layer 2:
5x5 Convolutional.          Input = 14x14x6.    Output = 10x10x16.  Activation = relu
2x2 MaxPooling.             Input = 10x10x16.   Output = 5x5x16
Layer 3:
5x5 Convolutional.          Input = 5x5x16.     Output = 1x1x400.   Activation = relu
Flatten convolution.        Input = 5x5x16.     Output = 400
Flatten convolution.        Input = 1x1x400.    Output = 400
Concatenate above two layers.                   Output = 800
Dropout regularization
Layer 4: Fully Connected.   Input = 800.        Output = 43.
```

and a graphic representation from the above paper.

![alt text][image7]


#### Training / Validation split

Now I split the training dataset into training and validation sets so as to cross-validate the model while training. I used the standar left-out validation method for this purpose. Cross-validation is an important step to keep the overfitting at check, along with the dropout and maxpooling layers, as mentioned above. After this split,
* number of validation examples: `2387`
* number of training examples: `45342`.


#### Optimizer and loss function

I used `Adam optimizer` with learning rate `0.001` since it helps to converge the model faster while minimizing the loss function. I used `softmax_cross_entropy()` from tensorflow as my primary loss function, as mentioned in the lesson. To avoid overfitting, I also used L2 regulariztion on the filter weights of the two fully connected networks ('fc1' and 'fc2' below) using `tf.nn.l2_loss()` API, along with the above cross-entroy loss. The following code snippet explains this.
```python
logits = build_cnn_model(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# L2_regularization
l2_loss = tf.nn.l2_loss(filter_weight['fc1']) + tf.nn.l2_loss(filter_bias['fc1'])
loss += L2_REGU_FACTOR * l2_loss  # L2_REGU_FACTOR is a hyperparamter to be tuned

# Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=RATE).minimize(loss)
```


#### Training the model & hypterparameter tuning

Finally, I was ready to train the model with the above training and validation data in batches. An important step while training the model was to experiment with various hyperparamters for the model and settle down on values that provide good validation score, as well as the right epoch number where the model starts overfitting.

After several iterations on their values, I finally chose the following hyperparamters for my model that provided consistent high accuracy on validation data.
```python
number of epochs = 30
batch size = 128
learning rate = 0.001
keep_prob for dropout layer = 0.8
L2 regularization factor = 5e-4
```

#### Training, validation, and test accuracy

The final accuracy of the model with the above hyperparamerters, optimizer, and loss function was as follows:
* training set accuracy: `0.998`
* validation set accuracy: `0.981`.

At this point, as always recommended, I trained my model on the entire training + validation dataset for the final testing of the model on the unseen test data, and I had
* test set accuracy: `0.921`.



### Test a Model on New Images

For implementation of this section, please refer to the `Test a Model on New Images` section in jupyter notebook.

#### Loading and resizing images

I randomly picked the following ten German traffic signs from the web.

![alt text][image8]

The images are not of the same shape unlike Udacity provided dataset (Here, we go!). For example, the shape of the first sign (30 km/h) is `(259, 194, 3)`, whereas the shape of the second image (Childrens crossing) is `(168, 299, 3)`. So I resized those images first to the size `(32x32x3)` as required by our pipeline. They look as follows after this step.

![alt text][image9]

#### Preprocessing

Then I ran the same preprocessing pipeline, namely grayscaling and normalizing, to those images, as I did for the training the model.

#### Performance of the model

Finally, I ran my model on these new images. The model was able to correctly guess 4 of the 10 traffic signs, which gives an accuracy of `40%`. This does compare favorably to the accuracy on the test set which was `92%`.

The top 5 probabilities and predicted classes for each image (in the same order as above, left to right) is as follows:
```
| Image                 | Predicted Probabilities                   | Predicted Class  | Actual class | Correct? |
|:----------------------|:------------------------------------------|------------------|--------------|----------|
| Speed limit (30km/h)  | [25.6669 24.0977 20.335  17.501  13.6295] | [22 18 24 14 37] |  1           |          |
| Speed limit (70km/h)  | [32.396  16.2778 15.2482 14.5263 14.3444] | [22 39 21 14 20] |  4           |          |
| Speed limit (100km/h) | [15.6811 14.0558 13.7544 13.7401  8.0946] | [18 27 26 40 12] |  7           |          |
| No passing            | [37.6866 27.0559 22.2123 20.3657 17.2654] | [ 9 41 16 28 10] |  9           |    Y     |
| Stop                  | [18.9173 12.1106 10.7846 10.6355  9.0884] | [14 33 20 39 34] |  14          |    Y     |
| No entry              | [43.616  20.0873 13.91   13.7417 13.3675] | [17 14 33 36 34] |  17          |    Y     |
| Slippery road         | [30.0859 29.4393 24.771  20.5017 19.6111] | [18 24 29 11 26] |  23          |          |
| Children crossing     | [31.4411 27.8007 26.5787 26.0859 21.2881] | [24 27 26 20 11] |  28          |          |
| Turn right ahead      | [14.2172 10.324   9.9464  9.0774  7.4673] | [33 11 34 28  1] |  33          |    Y     |
| Roundabout mandatory  | [13.2608 11.2747  9.9937  9.2092  8.3866] | [21 40  0 12 28] |  40          |          |
```

#### Discussion

As can be seen above, `Stop`, `Turn right ahead`, `No entry`, and `No passing` were detected correctly, and `Roundabout mandatory` came a close second. The reason these images did better than others is because the entire image consists of the signs only, whereas for other images, they sometime less than 50 % of the image size. This tells that cropping the images before throwing them to the pipeline could be a better option.

Second, the resizing really distored some of the images and could be a reason for not doing good in the prediction. For example, why did no speed limit signs were correctly recognized, even though they are the easiest to recognize? All 3 of them seems very distorted after resizing! More careful resizing techniques after looking at the individual pixel ranges (different color space?) of the images could be other option.

Third, a comparison among images which performed bad in test images to the ones above can be an interesting next task.
