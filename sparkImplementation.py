
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import itertools
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pyspark import SparkContext
from numpy import *
from numpy.linalg import inv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# scale larger positive and values to between -1,1 depending on the largest
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))


sc = SparkContext(appName="LogisticRegression")
yxinputFile = sc.textFile('bank-full.csv')
yxlines = yxinputFile.map(lambda line: line.split(';'))


(train, test) = yxlines.randomSplit([0.7, 0.3], seed = 42)

input_data_train = np.array(train.collect())
input_data_test = np.array(test.collect())

X_train = input_data_train[:,:-1]
y_train = input_data_train[:,16]

labelencode = LabelEncoder()
X_train[:,1] = labelencode.fit_transform(X_train[:,1])
X_train[:,2] = labelencode.fit_transform(X_train[:,2])
X_train[:,3] = labelencode.fit_transform(X_train[:,3])
X_train[:,4] = labelencode.fit_transform(X_train[:,4])
X_train[:,6] = labelencode.fit_transform(X_train[:,6])
X_train[:,7] = labelencode.fit_transform(X_train[:,7])
X_train[:,8] = labelencode.fit_transform(X_train[:,8])
X_train[:,15] = labelencode.fit_transform(X_train[:,15])
Y_train = labelencode.fit_transform(y_train)

X_train = np.delete(X_train, np.s_[9:11], 1)
X_train = np.array(X_train)
X_train = X_train.astype(np.float)
X_train = min_max_scaler.fit_transform(X_train)
Y_train = np.array(Y_train)

X_test = input_data_test[:,:-1]
y_test = input_data_test[:,16]



X_test[:,1] = labelencode.fit_transform(X_test[:,1])
X_test[:,2] = labelencode.fit_transform(X_test[:,2])
X_test[:,3] = labelencode.fit_transform(X_test[:,3])
X_test[:,4] = labelencode.fit_transform(X_test[:,4])
X_test[:,6] = labelencode.fit_transform(X_test[:,6])
X_test[:,7] = labelencode.fit_transform(X_test[:,7])
X_test[:,8] = labelencode.fit_transform(X_test[:,8])
X_test[:,15] = labelencode.fit_transform(X_test[:,15])
Y_test = labelencode.fit_transform(y_test)

X_test = np.delete(X_test, np.s_[9:11], 1)
X_test = np.array(X_test)
X_test = X_test.astype(np.float)
X_test = min_max_scaler.fit_transform(X_test)
Y_test = np.array(Y_test)




#parallelize and obtain RDD

X_train = sc.parallelize(X_train)
Y_train = sc.parallelize(Y_train)





##The sigmoid function adjusts the cost function hypotheses to adjust the algorithm proportionally for worse estimations
def Sigmoid(z):
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z


def Sigmoid2(z):
	z_matrix = []
	for x in np.nditer(z):
		G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*x))))
		z_matrix.append(G_of_Z)
	npa = np.matrix(z_matrix, dtype=np.float32)
	return npa.T

def Hypothesis(theta, x):
	z = 0
	#print(theta.shape, x.shape)
	theta = np.matrix(theta)
	theta_trans = np.transpose(theta)
	X = np.matrix(x)
	z = np.dot(X,theta_trans)

	return Sigmoid(z)

def Hypothesis2(theta, x):
	z = 0
	x = np.matrix(x)
	theta_trans = np.matrix(theta).T
	z = np.dot(x, theta_trans)
	return Sigmoid2(z)


def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	#print(theta)
	X = np.matrix(X.collect())
	Y = np.matrix(Y.collect())
	# print(X.shape)
	# print(Y.shape)
	Y = Y.T
	#print(theta)
	# print(X.shape)
	#print(X[1])
	#try mapping this
	for i in range(m):
		xi = X[i]
		#print(xi)
		hi = Hypothesis(theta,xi)
		
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	return J

##This function creates the gradient component for each Theta value 
def Cost_Function_Derivative(X,Y,theta,m,alpha):
	sumErrors = 0
	y = np.matrix(Y.collect())
	hi = X.map(lambda a: Hypothesis2(theta,a))
	hi = np.asmatrix(np.array(hi.collect())).T.astype('float')
	X = np.matrix(X.collect())
	hi = np.subtract(hi,y.T)
	error = np.dot(hi.T, X)
	constant = float(alpha)/float(m)
	J = np.multiply(constant,error)
	return error

def Gradient_Descent(X,Y,theta,m,alpha):
	constant = alpha/m
	theta = np.matrix(theta)
	CFDerivative = Cost_Function_Derivative(X,Y,theta,m,alpha)
	CFDerivative = np.multiply(constant,CFDerivative)
	new_theta = np.subtract(theta,CFDerivative)
	return new_theta


def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(X.collect())
	cost = []
	num_iterations = []
	for x in range(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			#print(theta)
			c = Cost_Function(X,Y,theta,m)
			cost.append(np.asscalar(c))
			num_iterations.append(x)
	#print("X_train", m)
	Declare_Winner(theta,cost,num_iterations)

def Declare_Winner(theta,cost,num_iterations):
	score = 0
	winner = ""
	print(num_iterations, cost)
	plt.plot(num_iterations,cost)
	plt.ylabel('Cost')
	plt.xlabel('Number of Iterations')
	plt.show()
	length = len(X_test)
	incorrect_score = 0
	Y_pred = []
	print(theta)
	for i in range(length):
	 	prediction = round(Hypothesis(X_test[i],theta))
	 	Y_pred.append(prediction)
	 	answer = Y_test[i]
	 	if prediction == answer:
	 		score += 1
	 	else:
	 		incorrect_score += 1
	my_score = float(score) / float(length)
	print (my_score*100)
	labels_pie = ['Correct Predictions', 'Incorrect Predictions']
	print(score,incorrect_score)
	values = [score, incorrect_score]
	print("X_test",length)
	colors = ['lightcoral', 'lightskyblue']
	plt.pie(values, labels=labels_pie,colors=colors,autopct='%1.1f%%')
	plt.axis('equal')
	plt.show()
	Y_pred = np.matrix(Y_pred)
	print(Y_test.shape, Y_pred.shape)
	print(Y_pred)
	cnf_matrix = confusion_matrix(Y_test, Y_pred.T)	
	np.set_printoptions(precision=2)
	classes = ['Subscribed', 'Not Subscribed']
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=classes,title='Confusion matrix')
	plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

initial_theta = np.zeros(14)  #[0,0] 
alpha = 0.1
iterations = 1000
Logistic_Regression(X_train,Y_train,alpha,initial_theta,iterations)
sc.stop()