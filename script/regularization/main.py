import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

Implementing L2 (ridge) regularization. Adding bias to the model seeking a decrease in variance. 

I simulate over-fitting by setting cross-validated data of different magnitudes. Really, overfitting
occurs when a model is too complex and provided non-zero parameter to noisy features.

When this is this case you wanna get to equally-accurate hypothesis/model that contain smaller thetas (slopes),
in order to prevent very large unexpected increases in the h(x).

In other words, a model with a slightly added bias and fewer variance is ideal. Regularization does that. 


"""

class LinearRegression():
      def __init__(self, X,y, split = .75) -> None:
            self.y = y
            self.X, self.theta = self.set_up(X, split)

            


      def set_up(self,X, split):
            self.m, self.n = X.shape

            theta = np.zeros(self.n +1) # Adding intercept

            intercept = np.ones((self.m, 1))

            X = np.hstack((intercept, X))

            boundary = int(len(X) * split)


            # You really can make atrri
            self.X_train = X[:boundary]
            self.X_test = X[boundary:]

            self.y_train = y[:boundary]
            self.y_test = y[boundary:]

            return X,theta
      
      def hypothesis(self, mode):

            if mode == "train":
                  X = self.X_train

            elif mode == "test":
                  X = self.X_test

            hypothesis = np.dot(X, self.theta)

            return hypothesis
      
      def gradient_descent(self,iters, alpha = 0.01):

            for i in range(iters):

                  hypothesis = self.hypothesis("train")

                  error = hypothesis - self.y_train # This is a matrix

                  gradient = np.dot(self.X_train.T, error)

                  self.theta -= (alpha/self.m)  * gradient # Scalar times vector

            return self.theta
      
      def regularized_gradient_descent(self, iters, _lambda, alpha = 0.01):

            """
            
            Worst case scenario for setting lambda too low is keeping overfitting present.
            That is, no penalization effect is present.
            
            """

            for i in range(iters):

                  hypothesis = self.hypothesis("train")

                  error = hypothesis - self.y_train # This is a matrix

                  gradient = np.dot(self.X_train.T, error) 
                  
                  penalization = ((self.theta) * (_lambda / self.m))
                  penalization[0] = 0

                  gradient += penalization

                  self.theta -= (alpha/self.m) * gradient # Scalar times vector

            return self.theta

      
      def cost_function(self, mode):

            if mode == "train":
                  y = self.y_train

            elif mode == "test":
                  y = self.y_test

            hypothesis = self.hypothesis(mode)

            error = (hypothesis - y)**2

            cost = np.sum(error) / (2*self.m)

            return cost
      

url = "/Users/javierdominguezsegura/Programming/Python/Algorithms/machine_learning/linear_regression/data/test.csv"

df = pd.read_csv(url)

X = df["feature"].values.reshape(-1,1)
y = df["target"].values


model = LinearRegression(X,y, .2)

fitted_thetas = (model.gradient_descent(1000, .01))

print(fitted_thetas)

def f(X_line, parameters):
      y = parameters[0] + np.dot(parameters[1], X_line)
      return y

plt.scatter(X,y, label = "original")

X_line = np.linspace(X.min(), X.max(), 100)
y_line = f(X_line, fitted_thetas)

cost_non_reg = model.cost_function("test")

print(f"Cost for non-reg {cost_non_reg}\n")

print("-"*40)

# Now REGULARIZING

reg_model = LinearRegression(X,y,.2)

regularized_thetas = reg_model.regularized_gradient_descent(1000, 1, 0.001)

print(regularized_thetas)

X_reg_line = np.linspace(X.min(), X.max(), 100)
y_reg_line = f(X_reg_line, regularized_thetas)

cost_reg = reg_model.cost_function("test")

print(f"Cost for reg {cost_reg}")


plt.plot(X_line, y_line, color='red', label='Non-regularized')
plt.plot(X_reg_line, y_reg_line, color='green', label='Regularized')

plt.legend()
plt.show()



      
