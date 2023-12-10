import tensorflow as tf

BATCH_SIZE = 1024

# Batch and prefetch
def increase_dataset_performance(train_dataset, test_dataset):
    """
    For more detail: https://www.tensorflow.org/guide/data_performance
    """
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset
