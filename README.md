# Introduction
Exploratory notebooks and utils related to the [Feedback Prize - Evaluating Student Writing](https://kaggle.com/c/feedback-prize-2021) Kaggle competition.

We scored a decent top 44% on the private leaderboard, with our team "Wagon Bar" :) Our final inference notebook is available [here](https://www.kaggle.com/code/valentinlaurent2/two-longformers-inference)

A lot of our work (notebooks and datasets) is still currently private on Kaggle, but we plan to add it here soon.

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
