
 
import tensorflow as tf
import numpy as np
from collections import Counter
 
 
def knn_predict(X_train, y_train, X_test, k):
    
    """
    paramitor:
        X_train (ndarray): data for train 80% (2400 picture)ข้อมูลฟีเจอร์ชุด train (ที่ผ่านการ standardize แล้ว)
        y_train (ndarray): label's train (gender 0,1)
        X_test (ndarray): data for test test 20% (600 picture) for prediction
        k (int): number of neighbor for answer
    return:
        predictions (ndarray): array's class predic for all poin in x_test 
    """

    # trancefome --> Tensor to TensorFlow
    X_train_tf = tf.constant(X_train, dtype=tf.float32)
    X_test_tf = tf.constant(X_test, dtype=tf.float32)
 
    predictions = []
 
    # loop for Predict each test point one by one
    for i in range(X_test_tf.shape[0]):
        test_point = X_test_tf[i]
 
        # calculate distance 
        # formula : sqrt( (x1-x1')^2 + (x2-x2')^2 + ... )
        distances = tf.sqrt(
            tf.reduce_sum(tf.square(X_train_tf - test_point), axis=1)
        )
 
        # Find the index of the k nearest points.
        # tf.math.top_k Find the maximum value and use "-distances" to flip value 
        _, nearest_index = tf.math.top_k(-distances, k=k)
        nearest_index = nearest_index.numpy()
 
        # Get the labels of the k nearest neighbors.
        nearest_labels = y_train[nearest_index]
 
        # Find the most common label among the neighbors (majority vote).
        most_common_label = Counter(nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
 
    return np.array(predictions)
 