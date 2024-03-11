
""" QUICK DESCRIPTION: Introduce dataset and get linear regression analysis, this includes
plotting, optimized parameters, R^2 value and function for predictions among other features.
Few functions are hardcoded; chaning the dataset, should not affect the functionality of this code. (I have checked this values in
spreadsheet software and results are consisntent)

LIMITATIONS:
    1. Models may fail to be plotted correctly if >3 features 
    2. All datasets have to be locally stored, using scikitlearn's module will not work
    3. Running optimization has not been taken into account (I'm still a beginner) 

PROCEDURE:
1. Get a dataset stored (either in .csv or .txt)
2. Call the model with parameters: 
    - path= directory to csv
    - names = pass an array with the names of all the headers (both features and targets)
    - drop = pass a string with the name of the target (have to be in prior names' array)     
3. Call a method        

 ALL METHODS EXPLAINED: 

get_cost_function : compute loss function
get_gradient_function : optimize for the best parameters 
get_function : display hypothesis of linear regression
get_plot : plot linear regression over training dataset, adjust for a 3D plot for multivariate (3 features) linear regression
get_predict : add specific feature to compute for hypothesis
get_predictions : aggregate a large dataset on the trained data to a csv
get_stats : get coefficient of determination
get_alpha_plot: plot loss function over iterations for different learning rates (only available in LSM)
names = get array of features and target variables
original_x = get initial values of features before being transformed (for the training phase)
std, mean = methods for normalizing features (only available in multivariate linear regression)
theta, x_values, y_values = get access to fundamental matrices for the linaer regression
get_all =  Aggregate the following outputs to a .txt file
      1. Array of features and target
      2. Confirmation of data being aggregated to the csv
      3. Value of coefficient of determination
      4. Paramters selection and loss function display 
      5. Hypothesis display
      5. Get plot 
"""
              

import back  # Importing the main module with the OOP script


# Some examples of models are presented here:

# Simple Linear Regression,LSM
model = back.LeastSquaredMethod(path = "/Users/.../diabetes_dataset.csv",
                                 names = ["bmi","target"], drop="target") # This dataste is the a modified version of scikitlearn's diabetes dataset
print(model.get_all()) 

# Multivariate Linear Regression ,LSM
model2 = back.LeastSquaredMethod(path="/Users/.../houses_oregon.csv", 
                           names=["Size","Bedrooms", "Price"],drop= "Price")
print(model2.get_alpha_plot()) 

# Simple Linear Regression, Normal Equations 
model3 = back.NormalEquation(path="/Users/.../restaurants.csv", 
                           names=["Population","Price"], drop="Price")

print(model3.get_stats()) 
