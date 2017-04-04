#**Traffic Sign Recognition Writeup**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./bar_chart.png "Bar Chart Visualization"
[image2]: ./sign_visual.png "Sign Visualization"
[image3]: ./curve_left.jpg "Left Curve"
[image4]: ./no_entry.jpg "No Entry"
[image5]: ./speed_60.jpg "Speed Limit 60"
[image6]: ./stop_sign.jpg "Stop Sign"
[image7]: ./wild_animal_cross.jpg "Wild Animal Crossing"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Please check the associated file report.html or the Traffic_Sign_Classifier.ipynb notebook for the project code.

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python built-in len() funtion and indexing to find the following:

* The size of training set is 34799
* The size of test set is 4410
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the size of data for each unique class. As the graph shows, some classes have less data than the others. This leads to an unbalanced training data for the network.

![alt text][image1]

The image below shows each traffic sign as RGB image with their associated labels. Clearly, some images are darker than the others, which can make it difficult for the network to learn.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

For this project, I did not pre-process any inputs. In the beginning, I tried to augment the data with random translations, rotations, and scales. However, augmentation did not give me a very good result, and it took too much time to train the network.

I pre-processed the training data with normalization and grayscale at first, but I did not achieve validation accuracy >0.93 with this. It took me days to realize that I needed to pre-processed validation data and test data as well. As a result, I started with the original data without any pre-processing and achieved a validation accuracy over 0.94.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data were split into training set, validation set, and test set for us in the beginning of the project.

As mentioned above, I did try to use augmentation, but due to time and computer constraints, I decided to abort those data. Also, AWS did not have enough space to store augmented data.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 32x32x10 	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 28x28x20	|
| Dropout               | Keep Probability 0.9							|
| Max pooling	      	| 2x2 kernel, 2x2 stride,  outputs 14x14x20 	|
| Convolution 5x5		| 1x1 stride, valid padding, outputs 10x10x40	|
| Dropout               | Keep Probability 0.8							|
| Max pooling	      	| 2x2 kernel, 2x2 stride,  outputs 5x5x40 		|
| Fully connected		| Input 1000 -> Output 512        				|
| Dropout               | Keep Probability 0.5							|
| Fully connected		| Input 512 -> Output 43        				|
| Softmax				| Input 43	        							|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fourth, sixth, seventh, and eigth cell of the ipython notebook. 

| Batch Size | Epochs | Optimizer | Learning Rate |
|:----------:|:------:|:---------:|:-------------:|
| 128		 | 30     | ADAM      | 0.001 		  |


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* training set accuracy of 0.990
* validation set accuracy of 0.942
* test set accuracy of 0.936


If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I chose LeNet architecture as my starting point for this model. It is a well-known architecture, and I did a lab to study details of LeNet. However, the vanila LeNet model was not enough to classify the traffic signs. Potential problems were:

* The parameters were not enough. Ex. Too few nodes in the last fully-connected layer.
* Unbalanced data set could contribute to low validation accuracy.
* ReLu activation was not effective and potentially blocked back-propagation.
* Need more convolutional layers.
* The model may have been too complex or too simple.

Firstly, I experimented the model with ReLu and Dropout activation. For this particular architecture, by replacing ReLu with Dropout in the first conv layer and the second conv layer resulted in an increase of 1% ~ 2% of validation accuracy. I also increased the size of output for every layer. However, this actually decreased the validation accuracy due to over-fitting. As a result, I deleted one fully-connected layer and used Dropout activation with keep probability of 0.5, instead of ReLu, for the fully-connected layer. This gave me a huge increase in validation accuracy. In order to further prevent over-fitting and reduce computational costs, I added a conv layer with 1x1 filters to the model as the first layer of the network.

All Dropout layers were important to the model. This is because a Dropout layer is very effective to prevent the network from over-fitting, especially for the fully-connected layer with a large amount of neurons. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3]![alt text][image4] ![alt text][image7]  ![alt text][image5] 
![alt text][image6] 

Each image has high resolution and a decent brightness.
The No-Entry Sign may not be easy to classify because the front has an angle.
The Wild-Animal-Crossing Sign may be difficult to classify because of the 1km sign below it.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the twelfth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No Entry     			| No Entry 										|
| Curve Left			| Speed limit (70km/h)							|
| 60 km/h	      		| Speed limit (60km/h)					 		|
| Wild Animal Crossing	| No entry      								|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This does not compare favorably to the accuracy on the test set of ~94%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

![alt text][image4]
For the first image, the model is sure that this is a no-entry sign (probability of 1.0), and the image does contain a no-entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry   									| 
| .00     				| Speed limit (20km/h) 							|
| .00					| Speed limit (30km/h)							|
| .00	      			| Speed limit (50km/h)					 		|
| .00				    | Speed limit (60km/h)      					|

---
![alt text][image3]
For the second image, the model is sure that this is a Road narrows on the right sign (probability of 0.99), but the image is a curve-left sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Road narrows on the right   					| 
| .00     				| Speed limit (70km/h)							|
| .00					| Dangerous curve to the left					|
| .00	      			| Speed limit (20km/h)					 		|
| .00				    | Double curve      							|

---
![alt text][image7]
For the third image, the model is sure that this is a Speed limit (20km/h) sign (probability of 0.99), but the image is a wild-animal-crossing sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (20km/h)   						| 
| .00     				| No entry										|
| .00					| Vehicles over 3.5 metric tons prohibited		|
| .00	      			| Roundabout mandatory					 		|
| .00				    | Speed limit (30km/h)      					|

---
![alt text][image6] 
For the fourth image, the model is sure that this is a Speed limit (80km/h) sign (probability of 0.99), but the image is a Stop sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (80km/h)   						| 
| .00     				| Stop Sign										|
| .00					| No passing for vehicles over 3.5 metric tons	|
| .00	      			| Speed limit (60km/h)					 		|
| .00				    | Road work    									|

---
![alt text][image5]
For the fifth image, the model is sure that this is a Speed limit (60km/h) sign (probability of 1.0), and the image does contain a Speed limit (60km/h) sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (60km/h)   						| 
| .00     				| Speed limit (80km/h)							|
| .00					| Speed limit (20km/h)							|
| .00	      			| Speed limit (30km/h)					 		|
| .00				    | Speed limit (50km/h)     						|

