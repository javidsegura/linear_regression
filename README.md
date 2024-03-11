# LINEAR REGRESSION ANALYSIS 


<p> <b> QUICK DESCRIPTION : </b>  Introduce any dataset and get linear regression analysis, this includes plotting, optimized parameters, R^2 value, and function for predictions among other features. Few functions are hardcoded; changing the dataset, should not affect the functionality of this code. (I have checked these values in
spreadsheet software and results are consistent)[1] </p>

<p> <b> LIMITATIONS: </b>
    1. Models may fail to be plotted correctly if >3 features 
    2. All datasets have to be locally stored, using scikitlearn's module will not work
    3. Running optimization has not been taken into account 
</p>

<p> <b> PROCEDURE </b>:
1. Get a dataset stored (either in .csv or .txt)
2. Call the model with parameters: 
    - path= directory to csv
    - names = pass an array with the names of all the headers (both features and targets)
    - drop = pass a string with the name of the target (have to be in prior names' array)     
3. Call a method    </p>

<p> <i> There is a complete explanation of all the methods at "front.py", starting in line 22 </i></p>

[1] Validation of the script is explained here: 
<img src="https://lh3.googleusercontent.com/drive-viewer/AKGpihYt0HfmFcoMSvHtqkSnDRPNLFL_ajuPCR3WFClTwTVvRGCiaZ8oBjp-ttcRT4l6rVsCVFdGBvlEoF53dOdW4cS4VvPRzQ=s1600" width = "200", height "300" /> 
<p> Image shows how the results from the script are consistent with that from spreadsheets </p>
