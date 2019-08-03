# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/class_distribution.png "class_distribution"
[image2]: ./data/gray_scale.png "Grayscaling"



[image3]: ./Online_pic/100.jpg "Traffic Sign 1"
[image4]: ./Online_pic/sliperry.jpg "Traffic Sign 2"
[image5]: ./Online_pic/stop_sign.jpg "Traffic Sign 3"
[image6]: ./Online_pic/animal.jpg "Traffic Sign 4"
[image7]: ./Online_pic/yield.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the standard python list information and pandas to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 (from the length of features in the training set)
* The size of the validation set is 4410 (from the length of features in the validation set)
* The size of test set is 12630 (from the length of features in the test set)
* The shape of a traffic sign image is (32, 32) (from the length of second and third dimension in the training set features)
* The number of unique classes/labels in the data set is 43 (read the number of classes from signnames.csv by pandas)

#### 2. Include an exploratory visualization of the dataset.

As shown in the image, the class distribution in trainning, validation, and test set are similar. So it is expected to have similar accuracy after trainning for both valiadation and test set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

All traffic signs are converted to grayscale pictures because for color has few information on the means of traffic signs. The shape of the pictures on the signs are more important.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

Then, all images are normalized by dividing all pixel values with 255. This normalization decrease the magnitude of the data, but can still maintain information such as shapes in the picture.

In addition, the sequecen of the data in the trainning set is shuffled every time before trainning to reduce the influence of data order.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
For this project, the LeNet model architecture by  Yan LeCun is applied as a starting point.

Beside the input and output layer, it has 2 convolutional layers, 2 max pooling layers, 4 RELU layers, 1 flatten layer, and 2 fully connected layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 30x30x6 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 29x29x6 			    	|
| Convolution 2x2	    | 1x1 stride, VALID padding, outputs 28x28x16 	|
| RELU		            |												|
| Max pooling	      	| 1x1 stride,  outputs 27x27x16 			   	|
| Flatten       		| Flatten 27x27x16 to 14400						|
| Fully Connected		| Input 14400 to output 900						|
| RELU					|												|
| Fully Connected		| Input 14400 to output 84						|
| RELU					| 												|
| Output layer			| 84x1 vector									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


To train the model, the Adamoptimizer is applied to reduce the cross entropy. 
The hyparparemeters are:
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:

* validation set accuracy of 0.948 
* test set accuracy of 0.919
The results are in the two cells above the markdown cell titled "STEP3"


The architecture chosen was the LeNet model architecture by Yan LeCun. 
This architecture was chosen for the recognition between pictures of word and pictures of sign are similar: the meanning of the recognized object is represented by the shape. In addition, traffic signs are usually formed by big and sperated shapes, which is the same feature words have.
The original architecture can train the model to have accuracy 0.86 on the validation set, thus the archituecure can recognizing the traffic sign with reasonable accuracy in the first place. 
To increase the accuracy, the filters in the convolutional layers were modified with smaller filter size and strides. 
This modification is applied under the assumption that, as the class of traffic signs is much more than the class of words in the original LeNet example, more data should be passed in the neural network.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 
![alt text][image6] ![alt text][image7]

Other than the U-turn and Stop sign images, all other imges have water mark on it, which introduce noise in these pictures.
All pictures are not strictly square, so they were twisted after resizing into 32x32, which may cause failure of classification.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road 		| Stop											| 
| Yield     			| Priority road 								|
| Wild animals crossing	| Wild animals crossing							|
| Stop	      		    | Bicycles crossing				 				|
| 100km/h				| 80km/h      									|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. The accuracy is much lower than on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.6e-08         		| Stop   										| 
| 4.9e-09     			| Priority road 								|
| .88					| Wild animals crossing							|
| .20	      			| Bicycles crossing      		 				|
| 4.5e-12				| 80km/h           								|


The model is most certain about the Wild animals crossing sign, which is the sign it correctly predicts. The second most certain sign is Bicycles crossing, though the probability is much lower than the first one. The remaining signs have further less probabilities.


