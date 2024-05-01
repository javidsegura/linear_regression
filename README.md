# LINEAR REGRESSION ANALYSIS 
<hr>
<img width="1104" alt="image (9) (1)" src="https://github.com/javidsegura/linear_regression/assets/129964070/000178fd-0527-4c91-a619-dd7f8c73cb46">

 

<h4> QUICK DESCRIPTION : </h4> 
<p> Introduce any dataset and get immediate linear regression analysis. This includes
plotting, optimizing parameters, R^2 value and predictions among other features.
Script proved to be functional (independently of the dataset introduced) due to the consistency of results 
after comparing the results via statsmodel.api (this is showcased in the .txt with the results
when calling 'get_all()' ). They have also been compared with Google spreadsheets, for even 
 more confidence in the prior conclusions. [1] </p>

<h4> LIMITATIONS: </h4><ol> 
    <li>  Models may fail to be plotted correctly if >3 features </li>
    <li>  All datasets have to be locally stored, using scikitlearn's module will not work </li>
    </ol>
    
<h4> DYNAMIC USE: </h4><ol> 
    <li>  Select if you want to split the dataset with train/test, and if so how much (percentage) </li>
    <li>  Select between two algorithms: LSM or Normal Equations </li>
    </ol>

<h4> PROCEDURE: </h4><ol> <li>  Get a dataset stored (either in .csv or .txt) </li>
 <li> Go to "front.py" </li>
<li> Call the model with parameters: </li>
    <ul>
    <li> path= directory to csv </li>
     <li> names = pass an array with the names of all the headers (both features and targets) </li>
    <li> drop = pass a string with the name of the target (have to be contained in prior names' array)  </li>   
    </ul>
<li> Call any method </li>
    </ol>
<p> <i> There is a complete explanation of all the methods at "front.py", starting in line 22 </i></p>

<hr>
[1] Validation of the script is shown here: 
<img width="500" alt="image (6) (1)" src="https://github.com/javidsegura/linear_regression/assets/129964070/c8ffb2bc-4747-422d-9246-b6cc6861c211">



<p> Image shows how the results from the script are consistent with that from spreadsheet software. You can also refer to the results in .txt output file when calling 
the get_all() method. </p>

<hr>
<p> Footnote: this is my first machine-learning program! </p>
<p> <i> JDS, 03/11/2024 </i></p>

