# Titanic Survival Prediction

## Overview
This repository contains two environments, one in **Python** and one in **R**, for predicting passenger survival in the Titanic dataset.  
Each version is built with Docker so the grader can reproduce the results without installing packages manually.

---

## Data
Download `train.csv` and `test.csv` from the [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data) and place them in `src/data/`.  
These files are required to run both the Python and R Docker environments but are not included in the repository.

---

## Running the Code

### Python Version
1. From the project root, build the image:
   ```bash
   docker build -t titanic-py -f src/titanic/Dockerfile .
   ```
2. Run the container:
   ```bash
   docker run --rm titanic-py
   ```
   This will train a logistic regression model and save predictions to `src/data/predictions.csv`.

---

### R Version
1. From the project root, build the image:
   ```bash
   docker build -t titanic-r -f src/r_titanic/Dockerfile .
   ```
2. Run the container:
   ```bash
   docker run --rm -v ${PWD}/src/data:/app/src/data titanic-r
   ```
   This will train a logistic regression model in R and save predictions to `src/data/predictions_r.csv`.
