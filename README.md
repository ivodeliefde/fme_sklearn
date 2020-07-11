# fme_sklearn
Custom transformer that integrates scikit-learn models in FME. 

# Install 
After you clone this repository you have to install the following libraries to your FME Python installation: 
- Pandas
- Scikit-learn

Do this with the following commands:
`fme python -m pip install pandas scikit-learn --user`
If there are multiple versions of FME installed, make sure to pip install using the correct FME.

Then run `install_transformers.bat`, open FME and look for the transformers called *sklearn_train* & *sklearn_predict*. 

For a manual install copy .fmx files and past them in the following folder: 
`C:\Users\<username>\Documents\FME\Transformers`. Copy the .py files and past them in the folder 
`C:\Users\<username>\Documents\FME\Plugins\Python\python37`. Alternatively, you could add the git folder to the default paths under FME options. 



