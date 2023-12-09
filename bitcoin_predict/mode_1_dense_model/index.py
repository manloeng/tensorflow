import tensorflow as tf
from tensorflow.keras import layers

from prep_data import prep_data_model_2
from bitcoin_predict.utils.model import compile_model, fit_model
from helper_functions.evaluation.metrics import make_preds, evaluate_preds

HORIZON = 1  # predict next 1 day
WINDOW_SIZE = 7  # use the past week of Bitcoin data to make the prediction


def train_model_1():
    train_windows, test_windows, train_labels, test_labels = prep_data_model_2(WINDOW_SIZE, HORIZON)

    # Set random seed for as reproducible results as possible
    tf.random.set_seed(42)

    # 1. Construct model
    model_1 = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(HORIZON, activation="linear")  # linear activation is the same as having no activation
    ], name="model_1_dense")  # name our model so we can save it

    # 2. Compile
    compile_model(model_1)

    # 3. Fit the model
    fit_model(model_1, train_windows, train_labels, test_windows, test_labels)
    print(model_1.evaluate(test_windows, test_labels))
    print("complete!")


def evaluate():
    train_windows, test_windows, train_labels, test_labels = prep_data_model_2(WINDOW_SIZE, HORIZON)
    model_1 = tf.keras.models.load_model("model_experiments/model_1_dense/")
    # print(model_1.evaluate(test_windows, test_labels))

    model_1_preds = make_preds(model_1, test_windows)
    # len(model_1_preds), print(model_1_preds[:10])

    model_1_results = evaluate_preds(
        y_true=tf.squeeze(test_labels),
        y_pred=model_1_preds
    )
    print(model_1_results)


# train_model_1()

evaluate()
