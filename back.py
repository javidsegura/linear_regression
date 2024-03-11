
# GET GRADIENT ALPHA PLOT 

""" The objective of this OOP script is to simplify the linear regression
analysis by havign an upper class that computes all the methods, 
independently of the algorithm used to trained the model
(e.g: LSM or Normal Equations). The training is automatically
triggered as soon as the class is instantiated"""

import pandas as pd # Needed to read csv 
import numpy as np # Daata manipulation
import matplotlib.pyplot as plt # Plotting results 
from mpl_toolkits.mplot3d import Axes3D # For multivariate plotting 
import csv, os # I/O functions 

class Linear_regression: # Superclass with methods for trained models 
      def __init__(self, path, names,drop) : # The csv should not contain headers, names parameter will provide the headers
            
            self.x_values, self.y_values = self.__read_data(path,names,drop) # Initiate function to read x and y from csv
            self.x_values, self.y_values, self.theta,self.mean,self.std = self.__set_up(self.x_values,self.y_values) # Once x,y found, add bias to x and create theta's initial guess
            
            
            self.original_x, self.original_y= np.copy(self.x_values), np.copy(self.y_values) # Store original x_values for later use in predictions 
            self.names = names # To be used for dynamic aggreagation in csv

            if isinstance(self,LeastSquaredMethod): # Train model with LSM 
                  self.get_gradient_function() # Start training automatically
            elif isinstance(self,NormalEquation): # Train model with normal equations
                    self.get_gradient_function() # Start training automatically

      def __read_data(self,path,names,drop): # Set up functions are private (should not be accesed from instance) thus the initial double underscore
            
            df = pd.read_csv(path, names=names)
            x_values = df.drop([drop], axis = 1) # Features are everything but target, axis = 1 to refer to columns and not rows 
            y_values = df[drop].values 
            return x_values, y_values
      
      def __set_up(self, x_values, y_values): #Private class
            m = len(y_values) # Get length of dataset, i.e: nº of rows. Could either be features or target (they're supposed to have the same length, as for being part of the training set)
            # Normalizing features
            if self.x_values.shape[1] > 1: # If multivariate, normalize data (assuming large scale difference between features) ; shapre[1] provides the number of columns of the features in without having added the y interception (theta_0)
                  mean,std = x_values.mean(axis = 0),x_values.std(axis = 0)
                  x_values = (x_values - mean) / std
            else:
                  mean= std = None 

            x_values = np.c_[np.ones(m), x_values] # Adding intercept to matrix; Explanation: concatenate an array to x_values full of 1s (as first column). We use 0s instead of 1, because for the former, its matrix multiplication will always yield 0!
            theta = np.zeros(x_values.shape[1]) # Initial guess of weights = 0s

            return x_values, y_values, theta, mean,std 
      def get_stats(self): #1st method. Get coefficient of determination

            if np.array_equal(self.theta, np.zeros_like(self.theta)): # Do not compute if model has not been trained. np.zeros like, creates an array of 0 of the length of self.theta. 
                        print( "The model is not trained!")
            else:
                  h = np.dot(self.x_values,self.theta) #Hypothesis
                  # Calculating R^2

                  ss_res = np.sum((self.y_values - h)**2)
                  ss_tot = np.sum((self.y_values-np.mean(self.y_values))**2)
                  r2 = 1 - (ss_res/ss_tot)
                  return r2
            
      def get_function(self): # 2nd method. Receive the function of the model 
            hypothesis_str = f"{self.theta[0]:.2f}" # y-intersection, first value, restricted to two significant figures
    
            # Add each subsequent term
            for i in range(1, len(self.theta)): # lower limit 1 to not collide with y-intersection
                  if self.theta[i] >= 0: 
                        hypothesis_str += f" + {self.theta[i]:.2f}x_{i}" # concatenation to the original string
                  else:
                        hypothesis_str += f" - {-self.theta[i]:.2f}x_{i}"
            hypothesis = f"h(x) = {hypothesis_str}" # final format
            return hypothesis

      def get_plot(self):
            if np.array_equal(self.theta, np.zeros_like(self.theta)): # Do not plot if model has not been trained
                        print( "The model is not trained!")
            elif len(self.theta) <= 2: # If simple linear regression (2, because at this moment y-intersection is already in the matrix)
                  m = len(self.x_values)
                  h = np.dot(self.x_values, self.theta) # Hypothesis (solving for function)
                  plt.plot(self.x_values[:,1].reshape([m,1]), self.y_values, "rx", label = "Training data") # Get all rows from 1st column and make m x 1 matrix. rx: r = c and marker = x; identifiers for plot being scatterplot
                  plt.plot(self.x_values[:,1].reshape([m,1]), h, label = "Linear regression") # Plot over the whole training the predictions
                  plt.xlabel(f"{self.names[:-1]}")  # Access dynamically to the axis' names
                  plt.ylabel(f"{self.names[-1:]}")
                  hypothesis = self.get_function()
                  plt.title(f"{hypothesis}") # Title is the hypothesis
                  plt.legend(loc = "upper right", bbox_to_anchor=(.37,.92)) # Add legend and adjust position for it to not overlap with the coefficient of determination
                  r2 = self.get_stats() 
                  plt.text(0.05, 0.95, f"$R^2$ = {r2:.2f}", ha = "left", va="center", transform = plt.gca().transAxes) 
                  return plt.show()
            elif len(self.theta) >2: # 3D plot for multivariate linear regression
                  fig = plt.figure()
                  ax = fig.add_subplot(111, projection ="3d")

                  x1_denormalized = self.x_values[:, 1] * self.std[0] + self.mean[0]   # Denormalizing data for proper scaling
                  x2_denormalized = self.x_values[:, 2] * self.std[1] + self.mean[1]

                  ax.scatter(x1_denormalized, x2_denormalized, self.y_values, c = "r", marker="x") 

                  x1_range = np.linspace(x1_denormalized.min(), x1_denormalized.max(),100) #Get line of feature
                  x2_range = np.linspace(x2_denormalized.min(), x2_denormalized.max(),100)
                  x1,x2 = np.meshgrid(x1_range, x2_range) # Scatter plot for 3D

                  # Normalizing surface for correct plotting
                  x1_norm = (x1 - self.mean[0]) / self.std[0]
                  x2_norm = (x2 - self.mean[1]) / self.std[1]
                  y = self.theta[0] + self.theta[1] * x1_norm + self.theta[2] * x2_norm # y is z in the plot (target variable)

                  ax.plot_surface(x1,x2,y, color = "blue", alpha = 0.5) # 3D plot linear regression function. Alpha represents opacity of perfect fit.


                  ax.set_xlabel(f"{self.names[0]}")
                  ax.set_ylabel(f"{self.names[-2]}")
                  ax.set_zlabel(f"{self.names[-1]}")

                  hypothesis = self.get_function()
                  ax.set_title(f"{hypothesis}")
                  return plt.show()

      def get_predict(self,features): # Get an specific prediction
                  if len(features) <= 2 and features[0] == 1: # If simple linear regression ...
                        prediction = np.dot(features,self.theta) 
                  elif len(features[0]) >= 2: # For multivariate the array of features have to be treated differently 
                        for i in features:
                              computation = (i - self.mean)/self.std # Standarize 
                              features = np.append(np.ones(1), computation) # Adjust for hypothesis' matrix computation
                              prediction = np.dot(features, self.theta) # Predicting
                  return prediction 
      def get_predictions(self): # Output automatic random predictions to .csv
            if isinstance(self, LeastSquaredMethod):
                   title = "LeastSquaredMethod"
            elif isinstance(self, NormalEquation):
                   title = "NormalEquation"
            with open(f"/Users/javierdominguezsegura/Programming/Python/Drafts/Scikit/Linear regression/Andrew ng/Multiple linear regression/results/results_{title}.csv", "w") as file:
                  writer = csv.writer(file, delimiter= ",")
                  theta_header = [f"θ_{i}" for i in range(len(self.theta))] 
                  headers = theta_header + [i for i in self.names] 
                  writer.writerow(headers)
            with open(f"/Users/javierdominguezsegura/Programming/Python/Drafts/Scikit/Linear regression/Andrew ng/Multiple linear regression/results/results_{title}.csv", "a") as file:
                  writer = csv.writer(file, delimiter= ",")
                  theta_row = [f"{i:.2f}" for i in self.theta]
                  if len(self.theta) == 2: # Simple linear regression
                         initial_value = float(np.max(self.original_x)) # First value of prediction is last of training 
                         increment = (initial_value - np.min(self.original_x)) / len(self.original_x) # Increment between predictions is range/n
                         for i in range(1500): # Arbitray value of predictions
                                results = list()
                                drop_computation = self.get_predict([1,float((initial_value +increment))]) # First value of prediction is last of training + increment (avoiding overlap)
                                results.append(initial_value) 
                                results.append(drop_computation) 
                                initial_value += increment
                                total_row = theta_row + results
                                writer.writerow(total_row)
                  elif len(self.theta) >2: 
                        initial_value = float((np.max(self.original_x[:,0]))) 
                        lower_limit = int(np.min(self.original_x[:,1])) # Needed for the second feature 
                        upper_limit = int(np.max(self.original_x[:,1]))
                        increment = int((initial_value - min(self.original_x[:,0])) / len(self.original_x))
                        for i in range(1500):
                              for j in range(lower_limit,upper_limit+1): 
                                    results = list()
                                    drop_computation = self.get_predict([[initial_value +increment,j]])
                                    results.append(initial_value) #1st feature
                                    results.append(j) # 2nd feature
                                    results.append(f"${drop_computation:.2f}") # Target of the prior feature array
                                    total_row = theta_row + results # Add all results + parameters (constant)
                                    writer.writerow(total_row)
                              initial_value += increment # Loop again with different values 
            return "Prediction results have been locally stored in a csv file"
      def get_all(self, alpha = 0.3, iterations =500): # General function; optional parameters if LSM wants to be modified
            os.system("clear") # Clean terminal 
            if isinstance(self, LeastSquaredMethod):
                   title = "LeastSquaredMethod"
            elif isinstance(self, NormalEquation):
                   title = "NormalEquation"
            with open(f"/Users/javierdominguezsegura/Programming/Python/Drafts/Scikit/Linear regression/Andrew ng/Multiple linear regression/results/results_{title}.txt","w") as file:
                  file.write("RESULTS OF LINEAR REGRESSION ANALYSIS \n")
                  file.write("-"*60)
                  file.write(f"\n\nFeatures: {self.names[:-1]}, drop: {self.names[-1:]}") 
                  file.write(f"\n{self.get_predictions()}")
                  file.write(f"\nR^2 value is {self.get_stats():.2f}")
                  file.write(f"\n{self.get_gradient_function(alpha, iterations) if isinstance(self, LeastSquaredMethod) else ""}")
                  file.write(f"\n{self.get_gradient_function() if isinstance(self, NormalEquation) else ""}")
                  file.write(f"Hypothesis is {self.get_function()}\n")
                  file.write("-"*60)
            self.get_plot()
            return f"\nSuccesfully executed, results stored in results_{title}.txt/.csv\n "


class LeastSquaredMethod(Linear_regression):
      def __init__(self, path, names ,drop):
            super().__init__(path , names ,drop)
            self.get_gradient_function() # Automatically train the model
      def get_cost_function(self): # Loss function 
            m = len(self.y_values)
            h = np.dot(self.x_values, self.theta)
            J = np.sum((h - self.y_values) **2) / (2 * m)
            return J
      def get_gradient_function(self, alpha = 0.3, iterations = 500): # Training the model; optimization of parameters/weights/coefficients
            J_history =  list()
            m = len(self.y_values)
            theta = self.theta
            for i in range(iterations):
                  h = np.dot(np.array(self.x_values),np.array(self.theta))
                  theta = theta - (alpha/m) * (h-self.y_values).dot(self.x_values)
                  self.theta = theta
                  J_history.append(self.get_cost_function()) # Get perfomance per iteration stored
            outcome = f"Optimal parameters are {self.theta}, with a cost of {J_history[-1]}"
            return  outcome # Return last iteration (optimized) parameters
      def get_alpha_plot(self): # Find most fitted learning rates
            x_values = self.original_x # Values at this point are trained, gotta take them back to original (already normalized)
            y_values = self.original_y

            x_values = np.c_[np.ones(len(x_values)), x_values] # Add bias to x_values
            theta = np.zeros(x_values.shape[1]) # Initial guess of weights = 0s

            def __cost_alpha(x,y,theta):
                  m = len(x)
                  h = np.dot(x, theta)
                  J = np.sum((h -y)**2) / (2 * m)
                  return J
            def __gradient_alpha(x = x_values, y = y_values, theta = theta, iterations = 50 ,alpha = 0): # Get values to orignal
                  J_history =  list()
                  m = len(y_values)
                  for i in range(iterations):
                        h = np.dot(np.array(x_values),np.array(theta))
                        theta = theta - (alpha/m) * (h-y_values).dot(x_values)
                        J_history.append(__cost_alpha(x,y,theta)) # Get perfomance per iteration stored
                  return theta, J_history # Return last iteration (optimized) parameters
            
            theta_3, J_3 = __gradient_alpha(alpha = 0.3) # Try different learning rates 
            theta_1, J_1 = __gradient_alpha(alpha = 0.1)
            theta_03, J_03 = __gradient_alpha(alpha = 0.03)
            theta_01, J_01 = __gradient_alpha(alpha = 0.01)
            theta_003, J_003 = __gradient_alpha(alpha= 0.003)
            theta_001, J_001 = __gradient_alpha(alpha = 0.001)

            plt.plot(J_3, label="0.3") # Plot them all
            plt.plot(J_1, label="0.1")
            plt.plot(J_03, label="0.03")
            plt.plot(J_01, label="0.01")
            plt.plot(J_003, label="0.003")
            plt.plot(J_001, label="0.001")

            plt.title("Testing Different Learning Rates")
            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost J")
            plt.legend(bbox_to_anchor=(1.05, 1.0))
            plt.show()

            return ""

      

class NormalEquation(Linear_regression):
      def __init__(self, path, names,drop):
            super().__init__(path , names ,drop,)
            self.get_gradient_function() # Automatically train the model
      def get_gradient_function(self):
            theta = np.linalg.inv((self.x_values.T.dot(self.x_values))).dot(self.x_values.T.dot(self.y_values)) # Normal equation
            self.theta = theta 
            outcome =  f"Optimal parameters are {self.theta}, with a cost of {self.get_cost_function()}"
            return outcome
      def get_cost_function(self): # Same as for LSM
            h = np.dot(self.x_values, self.theta)
            m = len(self.x_values)
            J = np.sum((h-self.y_values)**2) / (2*m)
            return J