import tensorflow as tf
import matplotlib.pyplot as plt


# NOTE: This code uses the TensorFlow API to get the MNIST dataset. It is not the most efficient way
# to that as it requires that you have the "tensorflow" module installed in order to run the code.

def load_mnist(print_data=False):
    """
    :param print_data: bool - If True, prints the shape of the train and test data as well as it type.
    :return: A tuple of numpy ndarray - ((train_data, train_labels), (test_data, test_labels))
    """
    mnist = tf.keras.datasets.mnist
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = tf.keras.utils.normalize(train_data, axis=1)
    test_data = tf.keras.utils.normalize(test_data, axis=1)

    if print_data:
        print('MNIST data loaded.')
        print('train_data shape:', train_data.shape, type(train_data))
        print('train_labels shape:', train_labels.shape, type(train_labels))
        print('test data shape:', test_data.shape, type(test_data))
        print('test labels shape:', test_labels.shape, type(test_labels))

    return (train_data, train_labels), (test_data, test_labels)


def show_train_img(index, print_array=False):
    """
    :param index: an integer - The position of the image to show.
    :param print_array: bool - If True, prints the array that represents the image.
    :return: None.
    """
    train_img, label_img = load_mnist()[0]
    print('Showing the number:', label_img[index])
    plt.imshow(train_img[index], cmap=plt.cm.binary)
    plt.show()
    if print_array:
        print('\n', train_img[index])
