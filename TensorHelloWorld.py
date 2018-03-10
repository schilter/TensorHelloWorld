import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def main():

    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format

    california_housing_dataframe = pd.read_csv(
        "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

# We'll randomize the data, just to be sure not to get any pathological ordering
# effects that might harm the performance of Stochastic Gradient Descent.
# Additionally, we'll scale median_house_value to be in units of thousands,
# so it can be learned a little more easily with learning rates in a range that
# we usually use.
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index))

    california_housing_dataframe["median_house_value"] /= 1000.0

    ###
    # In this exercise, we'll try to predict median_house_value, which will
    # be our label (sometimes also called a target). We'll use total_rooms as
    # our input feature. NOTE: Our data is at the city block level,
    # so this feature represents the total number of rooms in that block.

    # Define the input feature: total_rooms.
    my_feature = california_housing_dataframe[["total_rooms"]]

    # Configure a numeric feature column for total_rooms.
    feature_columns = [tf.feature_column.numeric_column("total_rooms")]
    # NOTE: The shape of our total_rooms data is a one-dimensional array
    #(a list of the total number of rooms for each block).
    # This is the default shape for numeric_column, so we don't have to
    # pass it as an argument.

    # Define the label.
    targets = california_housing_dataframe["median_house_value"]

    # Use gradient descent as the optimizer for training the model.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
# no clue on that split thing any thoughts???
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    # Configure the linear regression model with our feature columns (definition
    # not the data) and optimizer.
    # Set a learning rate of 0.0000001 for Gradient Descent.
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    linear_regressor = linear_regressor.train(
        input_fn=lambda: my_input_fn(my_feature, targets),
        steps=100
    )

    # Training done .. yajiii

    # Create an input function for predictions.
    # Note: Since we're making just one prediction for each example, we don't
    # need to repeat or shuffle the data here.
    def prediction_input_fn(my_feature, targets): return lambda: my_input_fn(
        my_feature, targets, num_epochs=1, shuffle=False)

    # use of the lambada ... not a nice one I think google ...
    #>>> def make_incrementor (n): return lambda x: x + n
    #>>>
    #>>> f = make_incrementor(2)
    #>>> g = make_incrementor(6)
    #>>>
    #>>> print f(42), g(42)
    # 44 48
    #>>>
    #>>> print make_incrementor(22)(33)
    # 55

    # Call predict() on the linear_regressor to make predictions.
    predictions = linear_regressor.predict(input_fn=prediction_input_fn(my_feature, targets))

    # Format predictions as a NumPy array, so we can calculate error metrics.
    predictions = np.array([item['predictions'][0] for item in predictions])

    # Print Mean Squared Error and Root Mean Squared Error.
    mean_squared_error = metrics.mean_squared_error(predictions, targets)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
    print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error

# Is this a good model? How would you judge how large this error is?
# Mean Squared Error (MSE) can be hard to interpret, so we often look at
# Root Mean Squared Error (RMSE) instead. A nice property of RMSE is that
# it can be interpreted on the same scale as the original targets.
# Let's compare the RMSE to the difference of the min and max of our targets:

    min_house_value = california_housing_dataframe["median_house_value"].min()
    max_house_value = california_housing_dataframe["median_house_value"].max()
    min_max_difference = max_house_value - min_house_value

    print "Min.  House Value: %0.3f" % min_house_value
    print "Max.  House Value: %0.3f" % max_house_value
    print "Difference between Min. and Max.: %0.3f" % min_max_difference
    print "Root Mean Squared Error: %0.3f" % root_mean_squared_error

    calibration_data = pd.DataFrame()
    calibration_data["predictions"] = pd.Series(predictions)
    calibration_data["targets"] = pd.Series(targets)
    print calibration_data.describe()

    sample = california_housing_dataframe.sample(n=300)

    # Get the min and max total_rooms values.
    x_0 = sample["total_rooms"].min()
    x_1 = sample["total_rooms"].max()

    # Retrieve the final weight and bias generated during training.
    weight = linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
    bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

    # Get the predicted median_house_values for the min and max total_rooms values.
    y_0 = weight * x_0 + bias
    y_1 = weight * x_1 + bias

    # Plot our regression line from (x_0, y_0) to (x_1, y_1).
    plt.plot([x_0, x_1], [y_0, y_1], c='r')

    # Label the graph axes.
    plt.ylabel("median_house_value")
    plt.xlabel("total_rooms")

    # Plot a scatter plot from our data sample.
    plt.scatter(sample["total_rooms"], sample["median_house_value"])

    # Display graph.
    plt.show()


if __name__ == '__main__':
    main()
