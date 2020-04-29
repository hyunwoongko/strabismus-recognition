5# *Strabismus Recognizer* 👀
*This project is to develop strabismus diagnostic software for use in ophthalmology.*
<br><br><br>

## 1. *Project Overeview*
![img](https://www.aao.org/image.axd?id=f0a526af-52f3-4edb-a4a5-a5d9ba8386a7&t=636486437122700000)
*Strabismus is a condition in which the eyes do not properly align with each other when looking at an object. The eye which is focused on an object can alternate. The condition may be present occasionally or constantly. If present during a large part of childhood, it may result in amblyopia or loss of depth perception. If onset is during adulthood, it is more likely to result in double vision.*
<br><br><br>

![img](http://morancore.utah.edu/wp-content/uploads/2017/08/hu_assessment_003.jpg)
*Strabismus is expressed in various types, and according to eye movement, it is classified into esotropia, exotropia, hypertropia and hypotropia. Currently, in ophthalmology, ophthalmologists diagnose the strabismus with the naked eye, but it is very difficult to diagnose strabismus unless a strabismus specialist. However, not all hospitals have strabismus specialists, so many ophthalmologists are currently struggling with strabismus diagnosis. So our goal is to develop software that automatically diagnoses strabismus. We are receiving data from some ophthalmologists and are thinking about using deep learning to solve this problem. We will experiment with various models such as MLP, RNN, GRU, LSTM, CNN, LSTM-CNN, Transformer and so on. and if all models do not perform well, we will construct a new model.*
<br><br><br>

##  2. *Development environment*
* *OS : Windows 10*
* *IDE : IntelliJ 2019.01*
* *Language : Python 3.6*
* *PC Specifications :*
  * *CPU : Intel(R) Core(TM) i7-9700KF @ 3.60Ghz*
  * *RAM : Samsung 16GB*
  * *GPU : Nvidia RTX 2070*
<br><br><br>

##  3. *Experiments*

![image](https://user-images.githubusercontent.com/38183241/70560563-2d915900-1bcc-11ea-8dfd-b1f908dfdd67.png)

*Random cross validation was conducted for the experiment. 
Firstly, we ramdomly select 178 patients' data (90%) 
from a total of 198 patients and proceed model training. 
After that, 20 patients' data (10%) that have not been trained are selected for diagnosis. 
In each experiment, we create and experiment with a total of 10 new models, 
and evaluate the minimum and average accuracy of the 10 models as the 
performance of the model.* 
<br><br>


### *3.1. Traditional Machine Learning Models Experiements*
<br>

|Model|Minimum Acc|Average Acc|
|:---:|:---:|:---:|
|SVM|0.375|0.559|
|NuSVM|0.525|0.6954|
|LinearSVM|0.525|0.6944|
|Decision Tree|0.475|0.706|
|Extra Tree|0.575|0.7144|
|Ada Boost|0.575|0.7204|
|Random Forest|0.625|0.7669|
|Gradient Boosting|0.65|0.7785|
|XGBoost|0.675|0.7769|
|Gaussian Naive Bayes|0.375|0.5215|
|Bernoulli Naive Bayes|0.45|0.5815|
|K Nearest Neighbors|0.6296|0.7074|
|Linear Regression|0.225|0.32|
|Logistic Regression|0.525|0.6425|
|Logistic Regression(CV)|0.55|0.6525|



Experimental results from traditional machine learning models. 
Even with the data added, the performance of the ensemble models 
(Bagging & Boosting) is the highest. In addition, we experimented 
with the XGBoost model, which shows performance similar 
to Gradient Boosting or Random Forest. (About 80% accuracy)
<br><br>


### *3.2. Multi Layer Perceptron Experiements*

#### 3.2.1 Normal MLP
|Model|Size|Params|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|
|MLP_256_5Layer|1.45MB|368,385|0.5|0.675|
|MLP_256_10Layer|2.67MB|699,905|0.375|0.495|
|MLP_256_15Layer|4.06MB|1,031,425|0.35|0.469|
|MLP_512_5Layer|5.41MB|1,392,129|0.5|0.69|
|MLP_512_10Layer|10.52MB|2,710,529|0.35|0.4925|
|MLP_512_15Layer|15.62MB|4,028,929|0.375|0.485|
|MLP_1024_5Layer|20.82MB|5,405,697|0.575|0.675|
|MLP_1024_10Layer|41.03MB|10,663,937|0.4|0.514|
|MLP_1024_15Layer|61.25MB|15,922,177|0.425|0.52|

#### 3.2.1 Residual MLP
|Model|Size|Params|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|
|RES_MLP_256_5Layer|1.45MB|368,385|0.55|0.664|
|RES_MLP_256_10Layer|2.67MB|699,905|0.6|0.712|
|RES_MLP_256_15Layer|4.06MB|1,031,425|0.675|0.7274|
|RES_MLP_512_5Layer|5.41MB|1,392,129|0.625|0.715|
|RES_MLP_512_10Layer|10.52MB|2,710,529|0.6|0.72|
|RES_MLP_512_15Layer|15.62MB|4,028,929|0.475|0.68|
|RES_MLP_1024_5Layer|20.82MB|5,405,697|0.575|0.695|
|RES_MLP_1024_10Layer|41.03MB|10,663,937|0.65|0.725|
|RES_MLP_1024_15Layer|61.25MB|15,922,177|0.55|0.6624|


<br>

### *3.3. 1D Convolutional Neural Networks Experiements*
<br>

|Model|Size|Params|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|
|VGG3|0.15MB|8,161|0.45|0.6625|
|VGG5|0.85MB|133,153|0.55|0.73|
|VGG7|8.8MB|2,105,889|0.525|0.735|
|VGG9|45.24MB|11,552,289|0.7|0.77|
|VGG11|141.53MB|36,728,353|0.5|0.6475|
|VGG13|237.70MB|61,906,465|0.35|0.5075|

|Inception3|14,529|0.24MB|0.425|0.7424|
|Inception5|262,785|1.55MB|0.45|0.745|
|Inception7|||0.75|0.7975|
|Inception9|||0.75|0.82|
|Inception11|||0.725|0.785|

|Inception_resnet5|262,785|1.55MB|0.45|0.737|
|Inception_resnet9|||0|0|
|Inception_resnet13|||0|0|
|Inception_resnet17|||0|0|








<br><br><br>


## 4. *Licence*

    Copyright 2019 CBNU CS Dept, AI/Robot LAB.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
