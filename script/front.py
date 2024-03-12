""" QUICK DESCRIPTION: Introduce any dataset and get inmediate linear regression analysis, this includes
plotting, optimized parameters, R^2 value and function for predictions among other features.
Script proved to be functional (independently of the dataset) due to consistency of results, 
after comparing the results via statsmodel.api (they are showcased in the .txt with the results
when calling 'get_all()' )

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
PROCEDURE:
1. Get a dataset stored (either in .csv or .txt)
2. Call the model with parameters: 
    - path= directory to csv
    - names = pass an array with the names of all the headers (both features and targets)
    - drop = pass a string with the name of the target (have to be in prior names' array)     
3. Call a method    

DYNAMIC USE:
1. Select if you want to split dataset with train/test, if so how much (percentage)
2. Select between two algorithms: LSM or Normal Equations

LIMITATIONS:
    1. Models may fail to be plotted correctly if >3 features 
    2. All datasets have to be locally stored, using scikitlearn's module will not work
     
.... (MORE INFO BACK AT LINE 57)...


"""
              

import back  # Importing the main module with the OOP script


# Some examples of models are presented here:

# Simple Linear Regression,LSM
model = back.LeastSquaredMethod(path = "/Users/.../diabetes_dataset.csv",
                                 names = ["bmi","target"], drop="target", training= .8) # This dataste is the a modified version of scikitlearn's diabetes dataset
print(model.get_plot()) 

# Multivariate Linear Regression ,LSM
model2 = back.LeastSquaredMethod(path="/Users/.../houses_oregon.csv", 
                           names=["Size","Bedrooms", "Price"],drop= "Price",training =1) 
print(model2.get_all()) 

# Simple Linear Regression, Normal Equations 
model3 = back.NormalEquation(path="/Users/.../restaurants.csv", 
                           names=["Population","Price"], drop="Price")

print(model3.get_all()) 





"""
 ALL METHODS EXPLAINED: 

1) get_all =  Aggregate the following outputs to a .txt file
      1. Array of features and target
      2. Confirmation of data being aggregated to the csv
      3. Value of coefficient of determination
      4. Paramters selection and loss function display 
      5. Hypothesis display
      5. Get plot  
2) get_function : display hypothesis of linear regression
3) get_plot : plot linear regression over training dataset, adjust for a 3D plot for multivariate (3 features) linear regression
4) get_predict : add specific feature to compute for hypothesis
5) get_predictions : aggregate a large dataset on the trained data to a csv
6) get_stats : get coefficient of determination, ser and p_values. Additionaly add parameter 'advanced = True', to see full summary from statsmodel.api
7) get_alpha_plot: plot loss function over iterations for different learning rates (only available in LSM)
8) get_homoscedasticity_test: see plotting of error term in order to indentify it the error is not biased 
9) names = get array of features and target variables
10) get_cost_function : compute loss function
11) get_gradient_function : optimize for the best parameters 

  RELEVANT ATTRIBUTES:
original_x = get initial values of features before being transformed (for the training phase)
std, mean = methods for normalizing features (only available in multivariate linear regression)
theta, x_values, y_values = get access to fundamental matrices for the linaer regression"""
