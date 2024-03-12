# LINEAR REGRESSION ANALYSIS 
<hr>
<img src="https://lh3.googleusercontent.com/drive-viewer/AKGpihbEkddvBoCgCt6TuF7hhMteuzHQozb_4epmoMq2zHfBSykBuXRUTSVqy6SIZuCOg4dodN8l-m_JUB5xL4TJn7I8xxpLQg=s1600" width = "500", height = "600"/> 

<img src="https://lh3.googleusercontent.com/drive-viewer/AKGpihaxzyZw4QmmEZU3w2aKtZJ-q4rL_qyoJmbMll7Vgyglfjoi4WUOVP-I3lgkPbOv1l-rJt0Twju_sHbMxH208sBF4ppk=s1600" width = "500", height = "600"/> </a>


 

<h4> QUICK DESCRIPTION : </h4> 
<p> Introduce any dataset and get inmediate linear regression analysis, this includes
plotting, optimized parameters, R^2 value and function for predictions among other features.
Script proved to be functional (independently of the dataset) due to the consistency of results, 
after comparing the results via statsmodel.api (they are showcased in the .txt with the results
when calling 'get_all()' ). They have also been compared with Google spreadsheets, for even 
 more confidence in these conclusions. [1] </p>

<h4> LIMITATIONS: </h4><ol> 
    <li>  Models may fail to be plotted correctly if >3 features </li>
    <li>  All datasets have to be locally stored, using scikitlearn's module will not work </li>
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

