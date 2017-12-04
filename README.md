## Logistic Regression on Portugese Banking Dataset
# Logistic Regression:
Logistic Regression is a regression model where the dependent variable is categorical. In our project, the dependent variable is binary, hence indicating a Binary Classifier form of logistic regression. In binary classification, the dependent variable can take up only two values, ‘0’ or ‘1’. Such a representation is indicative of tackling problems such as pass/fail, accepted/rejected and so on.

Sigmoid function-
The output vector will only be 0 or 1 i.e. y∈{0,1}. The hypothesis function h(x) must satisfy 0<h(x)<1. In order to map h(x) to the interval (0,1) we use the sigmoid function, also known as the logistic function.

![Image](images/sigmoid.png)

The function g(z) known as the sigmoid function, maps any real number to the (0,1) interval. We use sigmoid function to determine the probability of the output given a particular input.

Cost Function-

The cost function for the logistic regression is as follows-

![Image](images/costfunction.png)

The above two forms of the cost function can be clubbed into one single equation as shown below,

![Image](images/simplifiedcostfunc.png)

Vectorized implementation is as shown below,

![Image](images/vectorImplementationCostFunc.png)

Gradient Descent-

General form of gradient descent,

![Image](images/gdGeneral.png)

Vectorized implementation,

![Image](images/vectorizedGD.png)

Features used for Banking Dataset:

| col 1 is |  left-aligned |
| col 2 is |    centered   |
| col 3 is | right-aligned |



Logistic Regression Accuracy-

![Image](images/sparkLRAccuracyPieChart.png)

Confusion Matrix-

![Image](images/confusionmatrixSpark.png)

Cost vs Number of Iterations -

![Image](images/costvsnumiters.png)
