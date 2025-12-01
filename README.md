

# AID ICU Pred
This is a repository for the AID ICU prediction study.
The study aims to develop prediction models based on data collected in the AID ICU trial (DOI: 10.1007/s00134-023-07024-9)

------------------------------------------------------------------------

The repository contains:
- The aid_icu_eda.R and aid_icu_eda_90d.R files, used for the initial exploratory data analyses and for creation of graphs
- The aid_icu_pred.R and aid_icu_pred_90d.R files, documenting the development and validation of the prediction models
- The r folder, containing the implementation of performance metrics used in the study not included in the tidymodels packages

Further, the enviroment used to run the study can be recreated using the aid_icu_pred.Rproj together with the `renv` package and the renv.lock + renv folder