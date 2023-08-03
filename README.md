FootballPredictor
==============================

A toy project to learn about the world of football predictions.

This project is just intended as an exploration of these concepts for now. An OO implementation will follow when time allows!

# To run:

```
python3.7 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 
```

# To Do:

## Code useability:

* speed optimisation
* unit testing
* doc strings
* linting
* functional -> object orientated
* git integration


## Data Science:

* calculate player position distribution within each match
* Either
    * train a regression model to predict net home goals & compare
    * train a multi class bayes to classify W/D/L
* implement ranking algorithm (listwise method)



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── helpers.py     <- Utility functions
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
