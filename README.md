# CoronaClassifier

After pulling down any changes, run `make clean` to clean up all temporary files.

## Install Dependencies

`make install_requirements`

## Preprocessing

Preprocess the source data to generate the training and testing data sets.

The data sets are generated in the `preprocessed_data/train_set` and `preprocessed_data/test_set` directories, with
an filename extension of `.pt`.

`make preprocess`

> Take Note!
> Sometimes this step cannot exit automatically, even though all the data sets have been generated successfully.
> Be sure to press `Ctrl + C` to exit manually.

## Training

`make train`