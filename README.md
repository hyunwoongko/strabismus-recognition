# *Strabismus Recognizer* 👀
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
Firstly, we ramdomly select 158 patients' data (80%) 
from a total of 198 patients and proceed model training. 
After that, 40 patients' data (20%) that have not been trained are selected for diagnosis. 
In each experiment, we create and experiment with a total of 10 new models, 
and evaluate the minimum and average accuracy of the 10 models as the 
performance of the model.* 
<br><br>


### *3.1. Traditional Machine Learning Models Experiements*
<br>

|Model|Minimum Acc|Average Acc|
|:---:|:---:|:---:|
|SVM|0.3750|0.5590|
|NuSVM|0.5250|0.6954|
|LinearSVM|0.5250|0.6944|
|Decision Tree|0.4750|0.7060|
|Extra Tree|0.5750|0.7144|
|Ada Boost|0.5750|0.7204|
|Random Forest|0.6250|0.7669|
|Gradient Boosting|0.6500|0.7785|
|XGBoost|0.675|0.7769|
|Gaussian Naive Bayes|0.3750|0.5215|
|Bernoulli Naive Bayes|0.4500|0.5815|
|K Nearest Neighbors|0.6296|0.7074|

Experimental results from traditional machine learning models. 
Even with the data added, the performance of the ensemble models 
(Bagging & Boosting) is the highest. In addition, we experimented 
with the XGBoost model, which shows performance similar 
to Gradient Boosting or Random Forest. (About 76-77% accuracy)
<br><br>


### *3.2. Multi Layer Perceptron Experiements*

#### 3.2.1 Normal MLP
|Model|Size|Params|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|
|MLP_256_5Layer|1.45MB|368,385|0.55|0.7275|
|MLP_256_10Layer|2.67MB|699,905|0.35|0.4925|
|MLP_256_15Layer|4.06MB|1,031,425|||
|MLP_256_20Layer|5.37MB|1,362,945|||
|MLP_512_5Layer|5.41MB|1,392,129|||
|MLP_512_10Layer|10.52MB|2,710,529|||
|MLP_512_15Layer|15.62MB|4,028,929|||
|MLP_512_20Layer|20.73MB|5,347,329|||
|MLP_1024_5Layer|20.82MB|5,405,697|||
|MLP_1024_10Layer|41.03MB|10,663,937|||
|MLP_1024_15Layer|61.25MB|15,922,177|||
|MLP_1024_20Layer|81.46MB|21,180,417|||

#### 3.2.1 Residual MLP
|Model|Size|Params|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|
|RES_MLP_256_5Layer|1.45MB|368,385|0.55|0.7275|
|RES_MLP_256_10Layer|2.67MB|699,905|||
|RES_MLP_256_15Layer|4.06MB|1,031,425|||
|RES_MLP_256_20Layer|5.37MB|1,362,945|||
|RES_MLP_512_5Layer|5.41MB|1,392,129|||
|RES_MLP_512_10Layer|10.52MB|2,710,529|||
|RES_MLP_512_15Layer|15.62MB|4,028,929|||
|RES_MLP_512_20Layer|20.73MB|5,347,329|||
|RES_MLP_1024_5Layer|20.82MB|5,405,697|||
|RES_MLP_1024_10Layer|41.03MB|10,663,937|||
|RES_MLP_1024_15Layer|61.25MB|15,922,177|||
|RES_MLP_1024_20Layer|81.46MB|21,180,417|||






<br>

### *3.3. 1D Convolutional Neural Networks Experiements*
<br>

|Model|Size|Params|Iteration|Minimum Acc|Average Acc|
|:---:|:---:|:---:|:---:|:---:|:---:|


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
