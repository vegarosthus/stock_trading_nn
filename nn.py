import cvxopt as opt
import heapq
import numpy as np
import csv

from plot_stockdata import plot_weight_dev
from scipy.optimize import minimize

N_iter = 1

class NeuralNetwork():
	
	def __init__(self, input_sz, hidden1_sz, hidden2_sz = None, output_sz = 1):

		self.INPUT_SZ = input_sz
		self.OUTPUT_SZ = output_sz
		self.HIDDEN1_SZ = hidden1_sz
		self.HIDDEN2_SZ = hidden2_sz
	
		self.Theta = self.create_weight_matrices()
	
		#Arrays for model verification
		self.costs_training_set = []
		self.costs_cv_set = []
		self.costs_test_set = []
		self.costs_predict_set = []
		
		self.f1_scores_train = []
		self.accuracies_train = []
	
		self.f1_scores_cv = []
		self.accuracies_cv = []
	
		self.f1_scores_test = []
		self.accuracies_test = []
	
		self.accuracies_predict = []
		self.f1_scores_predict = []
	
	def input_data(self, X, y, X_cv, y_cv, X_test, y_test):
	
		self.X = X
		self.y = y
		
		self.X_cv = X_cv
		self.y_cv = y_cv
	
		self.X_test = X_test
		self.y_test = y_test
	
	
	def get_mod_ver_data(self):
		return self.costs_training_set, self.costs_cv_set, self.costs_test_set, self.accuracies_train, self.f1_scores_train, self.accuracies_cv, self.f1_scores_cv, self.accuracies_test, self.f1_scores_test
	
	def set_theta(self, theta):
		self.Theta = theta
	
	def get_theta(self):
		return self.Theta
	
	def create_weight_matrices(self):
		
		theta1 = self.random_init_weights(self.HIDDEN1_SZ, self.INPUT_SZ)
		
		if not self.HIDDEN2_SZ:
			theta2 = self.random_init_weights(self.OUTPUT_SZ, self.HIDDEN1_SZ)
		
			Theta = np.concatenate((np.ravel(theta1, 'F'), np.ravel(theta2, 'F')))

		else:
			theta2 = self.random_init_weights(self.HIDDEN2_SZ, self.HIDDEN1_SZ)
			theta3 = self.random_init_weights(self.OUTPUT_SZ,self.HIDDEN2_SZ)
			Theta = np.concatenate((np.ravel(theta1, 'F'), np.ravel(theta2, 'F'), np.ravel(theta3, 'F')))
		
		return Theta
	
	def F(self, theta = None, X = None, y = None, lambda_reg = None, type = None):
		
		'''
			Envelope for cost function.
		'''
		
		if theta.all() == None:
			theta = self.Theta
		if X.all() == None or y.all() == None or lambda_reg == None:
			return
		return self.cost_function(X, y, lambda_reg, theta, type)[0]

	def dF(self, theta = None, X = None, y = None, lambda_reg = None, type = None):
		
		'''
			Envelope for derivative of cost.
		'''
		
		if theta.all() == None:
			theta = self.Theta
		if X.all() == None or y.all() == None or lambda_reg == None:
			return
		return self.cost_function(X, y, lambda_reg, theta, type)[1]

	def sigmoid(self, array):
		
		'''
			Neuron activation function.
			Returns the elementwise sigmoid map of the input array.
		'''
	
		sigmoid_array = 1/(1+np.exp(-array))
	
		return sigmoid_array
	
	def plot_weights(self, grad, time):
		
		'''
			Function to plot the neuron weights as a color coded map before and after the network is trained.
		'''
		
		Theta1, Theta2, Theta3 = self.parameter_roll_in(grad)
		
		if Theta3 == [None]:
			thetas = [Theta1, Theta2]
		else:
			thetas = [Theta1, Theta2, Theta3]
	
		plot_weight_dev(thetas, time)

	def callbackF(self, theta, type = None):

		global N_iter
		print("Iteration No.: " + str(N_iter))
		
		lambda_reg = 1.2

		J, grad = self.cost_function(self.X, self.y, lambda_reg, theta, type)
		
		if N_iter == 1:
			self.plot_weights(grad, "initial")
		elif N_iter == 200:
			self.plot_weights(grad, "after_training")

		J_train, accuracy_train, f1_score_train = self.test_nn(self.X, self.y, 0, theta, type)
		J_cv, accuracy_cv, f1_score_cv = self.test_nn(self.X_cv, self.y_cv, 0, theta, type)
		J_test, accuracy_test, f1_score_test = self.test_nn(self.X_test, self.y_test, 0, theta)
		
		self.costs_training_set.append(J)
		self.costs_cv_set.append(J_cv)
		self.costs_test_set.append(J_test)
		
		self.accuracies_train.append(accuracy_train)
		self.f1_scores_train.append(f1_score_train)
		
		self.accuracies_cv.append(accuracy_cv)
		self.f1_scores_cv.append(f1_score_cv)
		
		self.accuracies_test.append(accuracy_test)
		self.f1_scores_test.append(f1_score_test)

		N_iter += 1

	def minimize_cost(self, X, y, lambda_reg, theta = None, type = None):

		if theta == None:
			theta = self.Theta

		print("Training network...")
		
		result = minimize(self.F, theta, method = 'BFGS', args=(X, y, lambda_reg, type), callback = self.callbackF, jac = self.dF, options={'disp': True, 'maxiter': 400})
		
		self.Theta = result.x
	
		return result

	def parameter_roll_in(self, theta):
		
		index1 = (self.HIDDEN1_SZ * (self.INPUT_SZ + 1))
		
		if self.HIDDEN2_SZ != None:
			index2 = (self.HIDDEN1_SZ * (self.INPUT_SZ + 1)) + (self.HIDDEN2_SZ * (self.HIDDEN1_SZ + 1))
	
		Theta1 = np.reshape(theta[ : index1], (self.HIDDEN1_SZ, (self.INPUT_SZ + 1)), order='F')
		
		if not self.HIDDEN2_SZ:
			Theta2 = np.reshape(theta[index1 : ], (self.OUTPUT_SZ, (self.HIDDEN1_SZ + 1)), order='F')
		
			return Theta1, Theta2, np.array([None])
		else:
			Theta2 = np.reshape(theta[index1 : index2], (self.HIDDEN2_SZ, (self.HIDDEN1_SZ + 1)), order='F')
			Theta3 = np.reshape(theta[index2 : ], (self.OUTPUT_SZ, (self.HIDDEN2_SZ + 1)), order='F')
			
			return Theta1, Theta2, Theta3

	def forward_propagation(self, input_array, Theta1, Theta2, Theta3 = None):
		
		'''
			Performs forward propagation to derive the activation values of the current network state as a function of the input vector.
		'''
		
		# Perform forward propagation and return single input cost
		a1 = input_array
		z2 = np.dot(Theta1, a1)
		a2 = self.sigmoid(z2)
		
		a2 = np.insert(a2, 0, 1)
		z3 = np.dot(Theta2, a2)
		
		if Theta3.any() == None:
			h_theta = self.sigmoid(z3)
		
			return a1, a2, None, h_theta
		else:
			a3 = self.sigmoid(z3)
			a3 = np.insert(a3,0,1)
			z4 = np.dot(Theta3, a3)
			h_theta = self.sigmoid(z4)

			return a1, a2, a3, h_theta

	def back_propagation(self, Theta1, Theta2, Theta3, Theta1_grad, Theta2_grad, Theta3_grad, h_theta, y, a1, a2, a3 = None):
		
		# Perform back propagation and return parameter gradients
		
		if Theta3.any() == None:
			delta3 = h_theta - y
		
		else:
			delta4 = h_theta - y
			delta3 = np.dot(Theta3.T, delta4) * a3 * (1-a3)
			
			#Delete first element of delta3
			delta3 = np.delete(delta3,0)
	
		delta2 = np.dot(Theta2.T, delta3) * a2 * (1-a2)
			
		#Delete first element of delta2
		delta2 = np.delete(delta2,0)
		
		#Gradients without regularization
		Theta2_grad = Theta2_grad + np.outer(delta3,a2)
		Theta1_grad = Theta1_grad + np.outer(delta2,a1)
		
		if Theta3.all():
			Theta3_grad = Theta3_grad + np.outer(delta4,a3)
			
			return Theta1_grad, Theta2_grad, Theta3_grad
		else:
			return Theta1_grad, Theta2_grad, None



	def cost_function(self, X, y, lambda_reg, theta = None, type = None, predict = False):
		if theta.all() == None:
			theta = self.Theta
		
		Theta1, Theta2, Theta3 = self.parameter_roll_in(theta)

		m = np.shape(X)[0]
		
		J = 0;
		
		Theta1_grad = np.zeros(np.shape(Theta1))
		Theta2_grad = np.zeros(np.shape(Theta2))

		if Theta3.all():
			Theta3_grad = np.zeros(np.shape(Theta3))
		else:
			Theta3_grad = None
		
		X = np.c_[np.ones((m,1)), X[:]]
		
		for i in range(m):

			if type == "verification":
				if y[i] == 0:
					y_logic = np.array([1, 0, 0])
				elif y[i] == 1:
					y_logic = np.array([0, 1, 0])
				elif y[i] == 2:
					y_logic = np.array([0, 0, 1])
			elif type == None:
				y_logic = y[i]

			#Forward propagation
			a1, a2, a3, h_theta = self.forward_propagation(X[i,:], Theta1, Theta2, Theta3)

			#Cost function without regularization
			J = J - np.dot(y_logic, np.log(h_theta)) - np.dot((1-y_logic), np.log(1-h_theta))

			#Back propagation
			Theta1_grad, Theta2_grad, Theta3_grad = self.back_propagation(Theta1, Theta2, Theta3, Theta1_grad, Theta2_grad, Theta3_grad, h_theta, y_logic, a1, a2, a3)


		#Cost function value with regularization of parameters
		if Theta3.all():
			J = (1/m) * J + (lambda_reg/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])) + np.sum(np.square(Theta3[:,1:])))
		else:
			J = (1/m) * J + (lambda_reg/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))

		Theta1_grad = (1/m) * Theta1_grad
		Theta2_grad = (1/m) * Theta2_grad

		if Theta3.all():
			Theta3_grad = (1/m) * Theta3_grad

		#Gradients with regularization
		Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambda_reg/m) * Theta1[:,1:]
		Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambda_reg/m) * Theta2[:,1:]
		
		if Theta3.all():
			Theta3_grad[:,1:] = Theta3_grad[:,1:] + (lambda_reg/m) * Theta3[:,1:]
			grad = np.concatenate((np.ravel(Theta1_grad, 'F'), np.ravel(Theta2_grad, 'F'), np.ravel(Theta3_grad, 'F')))
		else:
			grad = np.concatenate((np.ravel(Theta1_grad, 'F'), np.ravel(Theta2_grad, 'F')))

		return (J, grad)
			
	def predict(self, X, stocks, theta = None):
		
		if theta == None:
			theta = self.Theta
		
		Theta1, Theta2, Theta3 = self.parameter_roll_in(theta)
	
		m = np.shape(X)[0]
		
		X = np.c_[np.ones((m,1)), X[:]]

		y_p = np.zeros((m,1))

		for i in range(m):
			
			#Forward propagation
			a1, a2, a3, h_theta = self.forward_propagation(X[i,:], Theta1, Theta2, Theta3)
			
			y_p[i] = h_theta

		promising_stock_indices = heapq.nlargest(20, range(len(y_p)), y_p.take)
		
		promising_stocks = [(stocks[index], y_p[index]) for index in promising_stock_indices if y_p[index] > 0.7]
		
		return X, stocks, y_p, promising_stocks

	def random_init_weights(self, L_out, L_in):
	
		epsilon = np.sqrt(6)/np.sqrt(L_in+L_out)
		
		W = np.random.random((L_out,L_in+1)) * 2 * epsilon - epsilon
		
		return W
	
	def debug_init_weights(self, L_out, L_in):
		
		sines = np.linspace(-2*np.pi, 2*np.pi, L_out*(L_in+1))
		
		W = np.reshape(sines, (L_out, (L_in + 1)), order='F')/50
		
		return W


	def gradient_check(self, lambda_reg):
	
		INPUTS = 30
		HIDDEN1 = 5000
		HIDDEN2 = 5
		OUTPUTS = 1
		
		m = 2
		
		Theta1 = self.debug_init_weights(HIDDEN1,INPUTS)
		Theta2 = self.debug_init_weights(HIDDEN2,HIDDEN1)
		Theta3 = self.debug_init_weights(1,HIDDEN2)
		
		X = self.debug_init_weights(m, INPUTS-1)
		y = np.ones((m, OUTPUTS))
		
		parameters = np.concatenate((np.ravel(Theta1, 'F'), np.ravel(Theta2, 'F'), np.ravel(Theta3, 'F')))
		
		[J, grad] = self.cost_function(X, y, lambda_reg, parameters)
		num_grad = self.numerical_gradient(X, y, lambda_reg, parameters)
		
		print("Numeric", "Analytical")
		for x, y in zip(num_grad, grad):
			print(x, y)

		difference = (np.linalg.norm(grad) - np.linalg.norm(num_grad)) / (np.linalg.norm(grad) + np.linalg.norm(num_grad))
		
		print(difference)


	def numerical_gradient(self, X, y, lambda_reg, theta):
		
		EPSILON = 0.0001
		grad_approx = np.zeros(np.size(theta))
		
		for i in range(len(theta)):
			
			theta_plus = np.copy(theta)
			theta_minus = np.copy(theta)
			
			theta_plus[i] = theta[i] + EPSILON
			theta_minus[i] = theta[i] - EPSILON
			
			grad_approx[i] = (self.cost_function(X, y, lambda_reg, theta_plus)[0] - self.cost_function(X, y, lambda_reg, theta_minus)[0])/(2*EPSILON)

		return grad_approx

	def test_nn(self, X, y, lambda_reg, theta = None, type = None):

	
		if theta == None:
			theta = self.Theta
		
		Theta1, Theta2, Theta3 = self.parameter_roll_in(theta)
		
		m = np.shape(X)[0]
		
		X = np.c_[np.ones((m,1)), X[:]]
		
		y_test = np.zeros((m,1))
		
		J_test = 0
		
		#Performance data
		false_positives = 0
		false_negatives = 0
		true_positives = 0
		
		f1_score = 0
		
		for i in range(m):
			
			if type == "verification":
				if y[i] == 0:
					y_logic = np.array([1, 0, 0])
				elif y[i] == 1:
					y_logic = np.array([0, 1, 0])
				elif y[i] == 2:
					y_logic = np.array([0, 0, 1])
			else:
				y_logic = y[i]
			
			#Forward propagation
			a1, a2, a3, h_theta = self.forward_propagation(X[i,:], Theta1, Theta2, Theta3)
			
			#Cost function without regularization
			J_test = J_test - np.dot(y_logic, np.log(h_theta)) - np.dot((1-y_logic), np.log(1-h_theta))

			if type == "verification":
			
				index = int(y[i][0])
				h_theta_max = np.max(h_theta)

				if h_theta[index] != h_theta_max:
					y_test[i] = 1
				else:
					y_test[i] = 0

			else:
			
				if h_theta >= 0.5 and y[i] == 0:
					false_positives += 1
					y_test[i] = 1
				elif h_theta < 0.5 and y[i] == 1:
					false_negatives += 1
					y_test[i] = 1
				elif h_theta >= 0.5 and y[i] == 1:
					true_positives += 1
				else:
					y_test[i] = 0

			if type == "test":
				print(h_theta, y[i], y_logic, y_test[i])

			
		#Cost function value with regularization of parameters
		if Theta3.all() != None:
			J_test = (1/m) * J_test + (lambda_reg/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])) + np.sum(np.square(Theta3[:,1:])))
		else:
			J_test = (1/m) * J_test + (lambda_reg/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))

		no_of_errors = np.count_nonzero(y_test)
		no_of_successes = np.size(y_test) - no_of_errors
		
		try:
			precision = true_positives / (true_positives + false_positives)
			recall = true_positives / (true_positives + false_negatives)

			f1_score = 2 * ((precision * recall) / (precision + recall))
		except ZeroDivisionError:
			pass

		accuracy = (m-no_of_errors)/m

		if type == 'test':
			print(accuracy)
			print(f1_score)
			print(J_test)

		return J_test, accuracy, f1_score

	def verify_nn(self, file):

		with open(file) as f:
			read = csv.reader(f, delimiter=",")
			text = list(read)[8:]
		
		X = np.zeros((len(text), 4))
		y = np.zeros((len(text), 1))
		
		for sample, i in zip(text, range(len(text))):
			for sl, sw, pl, pw, cl in [sample]:
				
				X[i,:] = [sl, sw, pl, pw]

				if cl == "Iris-setosa\\":
					y[i] = 0
				elif cl == "Iris-versicolor\\":
					y[i] = 1
				else:
					y[i] = 2


		self.INPUT_SZ = 4
		self.HIDDEN1_SZ = 4
		self.HIDDEN2_SZ = 4
		self.OUTPUT_SZ = 3

		Theta1 = self.random_init_weights(self.HIDDEN1_SZ,self.INPUT_SZ)
		Theta2 = self.random_init_weights(self.HIDDEN2_SZ,self.HIDDEN1_SZ)
		Theta3 = self.random_init_weights(self.OUTPUT_SZ,self.HIDDEN2_SZ)

		parameters = np.concatenate((np.ravel(Theta1, 'F'), np.ravel(Theta2, 'F'), np.ravel(Theta3, 'F')))

		result = self.minimize_cost(X,y,1,parameters,"verification")

		self.test_nn(X,y,1,result.x,"verification")




