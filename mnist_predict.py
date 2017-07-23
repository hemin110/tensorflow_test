import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node = 784
output_node = 10

LAYER_NODE = 500
batch_size = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.001
TRAINING_STEP = 3000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor , avg_class , weight1 ,biases1 , weight2 , biases2 ):
    #avg_calss
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor , weight1)+biases1)
        return tf.matmul(layer1 , weight2)+biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor , avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1 , avg_class.average(weight2))+avg_class.average(biases2)

#train the model
def train(mnist):
    x = tf.placeholder(tf.float32 , shape=[None , input_node] , name = 'x-input')
    y_ = tf.placeholder(tf.float32 , shape=[None , output_node] , name = 'y-intput')
    weight1 = tf.Variable(tf.truncated_normal([input_node , LAYER_NODE] , stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1 , shape=[LAYER_NODE]))

    weight2 = tf.Variable(tf.truncated_normal([LAYER_NODE , output_node] , stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1 , shape=[output_node]))

    #not use average
    y = inference(x , None , weight1=weight1 , biases1=biases1 , weight2=weight2 , biases2=biases2)
    global_step = tf.Variable(0,trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY , global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x , variable_averages , weight1 , biases1 , weight2 , biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y , tf.arg_max(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weight1)+regularizer(weight2)

    loss = cross_entropy_mean+regularization

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_example/batch_size,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss , global_step = global_step)
