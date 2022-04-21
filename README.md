# Introduction
Exploratory notebooks and utils related to the [Feedback Prize - Evaluating Student Writing](https://kaggle.com/c/feedback-prize-2021) Kaggle competition.

As team "Wagon Bar", we scored a nice top 44% on the private leaderboard :)

Some of our work (notebooks and datasets) is still currently private on Kaggle, but we hope to add it here soon.

# Our work
During this 3 months-long competition, we:
1. Implemented a [baseline using Naive Bayes](https://github.com/Valentin-Laurent/evalstudent/blob/master/notebooks/valentin/bayesline-submission.ipynb)
2. Did a lot of EDA
2. Used mostly Longformers (BERT-style transformer optimized for long inputs) for advanced modelling

## Exploratory Data Analysis
You can find some of our findings in the notebook arthur/findings and [valentin/First exploration](https://github.com/Valentin-Laurent/evalstudent/blob/master/notebooks/valentin/First%20exploration.ipynb)

## Modeling phase 
Before using the models from the competition's best notebooks we tried several approaches, for example stacking a LSTM head after a Longformer. Details of the model achitectures are in the notebooks arthur/training_v2 and v3.

It is tough to compete with teams that have GPUs to train and fine-tune massive models, so re-using their work is a interesting way to learn while staying on top of the leaderboard. For our final inference notebook, we started from [this high-score public notebook](https://www.kaggle.com/code/abhishek/two-longformers-are-better-than-1). The idea is to use 2 Longformers stacked together. Each Longformer is trained on 5 different folds, so we end up with 10 models, and we average the predictions.

The author of the notebook made public the weights of these 10 trained models. We took advantage of this to cross-validate locally a lot of post-processing ideas, without having to train models ourselves. Luckily we found one idea that was indeed improving our CV score: the function `clean_rebuttals` at the end of [our final inference notebook](https://github.com/Valentin-Laurent/evalstudent/blob/master/notebooks/two-longformers-inference.ipynb).

Please note that this notebook will not work locally, as it needs access to the competition data and the models weight. However, you can make it run on Kaggle [here](https://www.kaggle.com/code/valentinlaurent2/two-longformers-inference/data)!

# Set up
To run our code locally, you can clone this repository:
```bash
git clone git@github.com:Valentin-Laurent/evalstudent.git
```

We do not provide a requirement.txt file, so you may need to install new Python libraries to make it work.

Some notebooks are using functions defined in `utils.py`and `metrics.py`. For these to run properly, you can install the package (in development mode) with:

```bash
cd evalstudent
pip install -e .
```
