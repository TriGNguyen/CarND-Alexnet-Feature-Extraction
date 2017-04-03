import pickle
import tensorflow as tf
import time
from sklearn.cross_validation import train_test_split
from alexnet import AlexNet

# wget https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p
TRAINING_FILE = 'train.p'
NUM_CLASSES = 43
LEARNING_RATE = 0.001
BATCH_SIZE = 512
EPOCHS = 3

# TODO: Load traffic signs data.
with open(TRAINING_FILE, mode='rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
x_train, x_dev, y_train, y_dev = train_test_split(data['features'], data['labels'], test_size=0.1)

# TODO: Define placeholders and resize operation.
x_variables = \
    tf.placeholder(\
        tf.float32, \
        shape=( \
            None, \
            x_train.shape[1], \
            x_train.shape[2], \
            x_train.shape[3]))
resized_x_variables = tf.image.resize_images(x_variables, (227, 227))
y_variables = tf.placeholder(tf.int32, shape=(None))
one_hot_y = tf.one_hot(y_variables, NUM_CLASSES)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_x_variables, feature_extract=True)

# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], NUM_CLASSES)  # use this shape for the weight matrix
weights_8 = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
bias_8 = tf.Variable(tf.zeros(NUM_CLASSES))
fully_connected_8 = tf.matmul(fc7, weights_8) + bias_8

logits = fully_connected_8

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
training_operation = optimizer.minimize(loss_operation)

prediction_operation = tf.argmax(tf.nn.softmax(logits), 1)
correct_prediction_operation = tf.equal(prediction_operation, tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction_operation, tf.float32))

def evaluate(X_data, y_data, batch_size, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = \
            sess.run(accuracy_operation, \
                     feed_dict={x_variables: batch_x, y_variables: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()
    
    for i in range(EPOCHS):
        start_time = time.time()

        for offset in range(0, len(x_train), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, \
                     feed_dict={x_variables: batch_x, y_variables: batch_y})

        train_accuracy = evaluate(x_train, y_train, BATCH_SIZE, sess)
        validation_accuracy = evaluate(x_dev, y_dev, BATCH_SIZE, sess)
        print("Trained epoch %d in %d seconds" % (i + 1, time.time() - start_time))
        print("Train accuracy = {:.3f}".format(train_accuracy))
        print("Validation accuracy = {:.3f}".format(validation_accuracy))
