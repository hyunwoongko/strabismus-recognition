# *Strabismus Recognizer* 👀
*This project is to develop strabismus diagnostic software for use in ophthalmology.*
<br><br><br>

## 1. *Project Overeview*
![img](https://www.aao.org/image.axd?id=f0a526af-52f3-4edb-a4a5-a5d9ba8386a7&t=636486437122700000)
*Strabismus is a condition in which the eyes do not properly align with each other when looking at an object. The eye which is focused on an object can alternate. The condition may be present occasionally or constantly. If present during a large part of childhood, it may result in amblyopia or loss of depth perception. If onset is during adulthood, it is more likely to result in double vision.*
<br><br><br>

![img](http://morancore.utah.edu/wp-content/uploads/2017/08/hu_assessment_003.jpg)
*Strabismus is expressed in various types, and according to eye movement, it is classified into esotropia, exotropia, hypertropia and hypotropia. Currently, in ophthalmology, ophthalmologists diagnose the strabismus with the naked eye, but it is very difficult to diagnose strabismus unless a strabismus specialist. However, not all hospitals have strabismus specialists, so many ophthalmologists are currently struggling with strabismus diagnosis.*
<br><br><br>

![img](https://lilianweng.github.io/lil-log/assets/images/transformer.png)
*So our goal is to develop software that automatically diagnoses strabismus. We are receiving data from some ophthalmologists and are thinking about using deep learning to solve this problem. We will experiment with various models such as MLP, RNN, GRU, LSTM, CNN, LSTM-CNN, Transformer and so on. and if all models do not perform well, we will construct a new model.*
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
Firstly, we ramdomly select 94 patients' data (70%) 
from a total of 135 patients and proceed model training. 
After that, 40 patients' data (30%) that have not been trained are selected for diagnosis. 
In each experiment, we create and experiment with a total of 10 new models, 
and evaluate the minimum and average accuracy of the 10 models as the 
performance of the model.* 
<br><br>


### *3.1. Traditional Machine Learning Models Experiements*
<br>

|Model|Minimum Acc|Average Acc|
|:---:|:---:|:---:|
|SVM|0.4074|0.4703|
|NuSVM|0.4444|0.6407|
|LinearSVM|0.5185|0.6148|
|Decision Tree|0.5185|0.6814|
|Extra Tree|0.6666|0.7333|
|Ada Boost|0.6296|0.7296|
|Random Forest|0.7037|0.8037|
|Gradient Boosting|0.6666|0.7629|
|XGBoost|0.6666|0.7777|
|Gaussian Naive Bayes|0.3703|0.5037|
|Bernoulli Naive Bayes|0.2962|0.5111|
|K Nearest Neighbors|0.6296|0.7074|
|MLP (iter = 20000)|0.4444|0.6962|

<br><br>


### *3.2. 1D Convolutional Neural Networks Experiements*
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
