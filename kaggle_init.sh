#!/bin/sh

kaggle competitions download -c titanic -p ./input -f train.csv
kaggle competitions download -c titanic -p ./input -f test.csv
kaggle competitions download -c titanic -p ./input -f gender_submission.csv
