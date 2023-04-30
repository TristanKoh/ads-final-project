# Applied Data Science Final Project: Group 5
Team members: Ahmed Lassoued, Daniel Lim, Tristan Koh, Vasu Namdeo, Zen Goh

Github link and other code for this project can be found [here](https://github.com/TristanKoh/ads-final-project).

The full file directory and csv / submission files can be found [here](https://drive.google.com/file/d/122VR8vbsGGJQbWpaijXQKDM6gLAqgDWj/view?usp=sharing)

## Statement of contribution
Ahmed Lassoued - Worked on ARIMA, ETS and clustering (not included in files because did not manage to get it to work in the end).
Daniel Lim - Worked on EDA, and helped with ML data structuring.
Tristan Koh - ML data structuring and ML modelling.
Vasu Namdeo - Led, coordinated and designed the presentation, did some of the EDA.
Zen Goh - Coordinated the compilation of the final notebook from different separate notebooks, and annotated the code chunks.

## Directory structure

### Datasets
The PPT of our presentation and PDF of our notebook is including in the root of this directory.

[code](/code/) contains the final notebook with EDA and models, and ```func.py``` contains the helper functions for our notebook.

[datasets](/datasets/) contains ```.csv``` files from [Kaggle](https://www.kaggle.com/competitions/ysc4224-2023-final-project/data). It also contains ```long_train_6_mths_with_lags.csv``` long form data restructured from ```train.csv``` that is used for ML modelling.

[submissions](/submissions/) contains the ```.csv``` files of the top 5 models and ML models that we submitted to Kaggle. [best_model_params](/submissions/ml_best_models_params/lgb_model_fc.txt) also contains the trained hyperparamters of the Light GBM model.

### Dependencies
Install dependencies using ```pip install -r requirements.txt``` in your command line. If you use new packages not already included in [requirements.txt](/requirements.txt), make sure to update [requirements.txt](/requirements.txt).