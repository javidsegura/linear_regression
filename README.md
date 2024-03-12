# LINEAR REGRESSION ANALYSIS 
<hr>
<img src="https://lh3.googleusercontent.com/drive-viewer/AKGpihaxzyZw4QmmEZU3w2aKtZJ-q4rL_qyoJmbMll7Vgyglfjoi4WUOVP-I3lgkPbOv1l-rJt0Twju_sHbMxH208sBF4ppk=s1600" width = "500"/>


 

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
<img src="https://lh3.googleusercontent.com/drive-viewer/AKGpihYt0HfmFcoMSvHtqkSnDRPNLFL_ajuPCR3WFClTwTVvRGCiaZ8oBjp-ttcRT4l6rVsCVFdGBvlEoF53dOdW4cS4VvPRzQ=s1600" width = "400", height "500" /> 

<p> Image shows how the results from the script are consistent with that from spreadsheet software. You can also refer to the results in .txt output file when calling 
the get_all() method. </p>

<hr>
<p> Footnote: this is my first machine-learning program! </p>
<p> <i> JDS, 03/11/2024 </i></p>

