# KNN Classification

### data_loader.py

Data Loader is used to prepare the data before using KNN.

* Read the CSV file

* Randomly select some data if `sample_size` is set because KNN must calculate the distance between data points

* Transform `pixels` from string to array of numbers

* Use the specified column as the label (`gender`)

* Split the data into train/test sets

* Standardize the pixel values to mean = 0 and std = 1


### main.py

Main is used to run the KNN model and save the results.

* Create the `outputs` folder if it doesn't already exist

* Train and evaluate the model for each value of `k`

* Store the results for the best `k` value to use for the confusion matrix and `predictions.csv`

* Save a graph comparing the accuracy of each `k` value

* Save a table comparing the actual labels with the predicted labels for the best `k` only


### evaluate.py

Evaluate is used to calculate the accuracy and create graphs.

* `calculate_accuracy`

Calculate the model accuracy.

* `plot_k_curve`

Plot a graph comparing the accuracy for each value of `k`.

* `plot_confusion_matrix`

Plot the confusion matrix of the best-performing model.


### KNN.py

KNN is used to predict the label of each test point.

* Transform Tensor to TensorFlow

* Loop to predict each test point one by one

* Calculate distance

* Find the index of the `k` nearest points

* Get the labels of the `k` nearest neighbors

* Find the most common label among the neighbors (majority vote)