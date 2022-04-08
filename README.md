# Introduction
Exploratory notebooks and utils related to the [Feedback Prize - Evaluating Student Writing](https://kaggle.com/c/feedback-prize-2021) Kaggle competition.

We scored a cool top 44% on the private leaderboard, with our team "Wagon Bar" :) Our final inference notebook is available [here](https://www.kaggle.com/code/valentinlaurent2/two-longformers-inference)

A lot of our work (notebooks and datasets) is still currently private on Kaggle, but we plan to add it here soon.

# Our work
During this 3 months-long competition, we:
1. Implemented a baseline using Naive Bayes
2. Did a lot of EDA
2. Used mostly Longformers (BERT-style transformer optimized for long inputs) for advanced modelling

It is tough to compete with teams that have GPUs to train and fine-tune massive models, so re-using their work is a interesting way to learn while staying on top of the leaderboard. For our final inference model, we started from [this high-score public notebook](https://www.kaggle.com/code/abhishek/two-longformers-are-better-than-1).

The idea is to use 2 trained Longformers stacked together. Each model is trained on 5 folds, so we end up with 10 models, and we average the predictions.

We took advantage of the fact that we had models weight for each fold to cross-validate locally a lot of post-processing ideas, without having to train models ourselves. Luckily we found one that worked, so we used it to boost our final score. It is the function `clean_rebuttals`and the end of [our final inference notebook](https://github.com/Valentin-Laurent/evalstudent/blob/master/notebooks/two-longformers-inference.ipynb).

Please note that this notebook will not work locally, as it needs access to the competition data and the models weight. However, you can make it run on Kaggle [here](https://www.kaggle.com/code/valentinlaurent2/two-longformers-inference/data)!


## Exploratory Data Analysis 
You can find most of our findings in the notebook arthur/findings

## Modeling phase 
Before using the models from the competition best notebooks we tried several approaches, especially stacking a LSTM head after a Longformer. Details of the model achitectures are in the notebooks arthur/training_v2 and v3

# Set up
To run our code locally, you can clone the project:
```bash
git clone git@github.com:Valentin-Laurent/evalstudent.git
```

We do not provide a requirement.txt file, so you may need to install new Python libraries to make it work.

Some notebooks are using functions defined in `utils.py`and `metrics.py`. For these to run properly, you can install the package (in development mode) with:

```bash
cd evaluating-student-writing
pip install -e .
```
