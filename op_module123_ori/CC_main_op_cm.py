# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py
# quoc_trinh

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import pylab as pl

import os
import sys
import datetime
import cPickle as cp

# get current file_name as [0] of array
file_name =  os.path.splitext(os.path.basename(sys.argv[0]))[0]
print(" File Name:")
print(file_name)
print("")
# FLAG to know that whether this is traning process or not.
FLAG = 'train'
N_HIDDEN_CONFIG = 6
N_CLASS_CONFIG = 18
X_AFTER_RESHAPE = 12
Y_AFTER_RESHAPE = 10

OPP_DATA_PATH = '../data/oppChallenge_gestures_2.data'

save_path_name =  file_name + "/model.ckpt"

print(datetime.datetime.now())
# Write to file: time to start, type, time to end
f = open(file_name + '/time.txt', 'a+')
f.write("------------- \n")
f.write("This is time \n")
f.write("Started at \n")
f.write(str(datetime.datetime.now())+'\n')

if __name__ == "__main__":
    #########################################
    def load_dataset(filename):

        f = file(filename, 'rb')
        data = cp.load(f)
        f.close()

        X_train, y_train = data[0] 
        print("-----------X_train-----------")
        print(X_train)    
        print(len(X_train))
        print(len(X_train[0]))
        print((X_train.shape))
        print("-----------y_train-----------")
        print(y_train)    
        print(len(y_train))
        print((y_train.shape))
        X_test, y_test = data[1]
        print("-----------y_test-----------")
        print(y_test)    
        print(len(y_test))
        print((y_train.shape))
        # exit()
        print(" ..from file {}".format(filename))
        print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        # The targets are casted to int8 for GPU compatibility.
        y_train = y_train.astype(np.uint8)
        y_test = y_test.astype(np.uint8)

        return X_train, y_train, X_test, y_test

    print("Loading data...")
    X_train_op, y_train_op, X_test_op, y_test_op = load_dataset(OPP_DATA_PATH)
    ##########################################

    # -----------------------------
    # step1: load and prepare data
    # -----------------------------
    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "../data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    # Preparing data set:
    TRAIN = "train/"
    TEST = "test/"

    # Load "X" (the neural network's training and testing inputs)
    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'rb')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                    ]]
            )
            file.close()

        """Examples
        --------
        >> > x = np.arange(4).reshape((2, 2))
        >> > x
        array([[0, 1],
               [2, 3]])

        >> > np.transpose(x)
        array([[0, 2],
               [1, 3]])

        >> > x = np.ones((1, 2, 3))
        >> > np.transpose(x, (1, 0, 2)).shape
        (2, 1, 3)
        """

        return np.transpose(np.array(X_signals), (1, 2, 0))

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
        ]
    # X_train = load_X(X_train_signals_paths)  # [7352, 128, 9]
    # X_test = load_X(X_test_signals_paths)    # [7352, 128, 9]
    X_train = X_train_op #  (557963, 113)
    X_test = X_test_op

    X_train = np.reshape(X_train, [-1, X_AFTER_RESHAPE, Y_AFTER_RESHAPE])
    X_test = np.reshape(X_test, [-1, X_AFTER_RESHAPE, Y_AFTER_RESHAPE])

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    def one_hot(label):
        """convert label from dense to one hot
          argument:
            label: ndarray dense label ,shape: [sample_num,1]
          return:
            one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
        """
        label_num = len(label)
        new_label = label.reshape(label_num)  # shape : [sample_num]
        # because max is 5, and we will create 6 columns
        n_values = np.max(new_label) + 1
        return np.eye(n_values)[np.array(new_label, dtype=np.int32)]

    # Load "y" (the neural network's training and testing outputs)
    def load_y(y_path):
        file = open(y_path, 'rb')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
                ]],
            dtype=np.int32
        )
        file.close()
        # Subtract 1 to each output class for friendly 0-based indexing
        return y_ - 1

    # y_train = one_hot(load_y(y_train_path))
    # y_test = one_hot(load_y(y_test_path))
    y_train = one_hot(y_train_op)
    y_test = one_hot(y_test_op)


    print("---------y_train----------")
    # print(y_train)
    print(len(y_train))  # 7352
    print(len(y_train[0]))  # 6

    # -----------------------------------
    # step2: define parameters for model
    # -----------------------------------
    class Config(object):
        """
        define a class to store parameters,
        the input should be feature mat of training and testing
        """

        def __init__(self, X_train, X_test):
            # Input data
            self.train_count = len(X_train)  # 7352 training series
            self.test_data_count = len(X_test)  # 2947 testing series
            self.n_steps = len(X_train[0])  # 128 time_steps per series

            # Training
            self.learning_rate = 0.0025
            self.lambda_loss_amount = 0.0015
            self.training_epochs = 1
            self.batch_size = 1000

            # LSTM structure
            self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
            self.n_hidden = N_HIDDEN_CONFIG  # nb of neurons inside the neural network
            self.n_classes = N_CLASS_CONFIG  # Final output classes
            self.W = {
                'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),  # [9, 32]
                'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))  # [32, 6]
            }
            self.biases = {
                'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),  # [32]
                'output': tf.Variable(tf.random_normal([self.n_classes]))  # [6]
            }

    config = Config(X_train, X_test)
    # print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("features shape, labels shape, each features mean, each features standard deviation")
    # print(X_test.shape, y_test.shape,
    #       np.mean(X_test), np.std(X_test))
    # print("the dataset is therefore properly normalised, as expected.")
    #
    #
    # ------------------------------------------------------
    # step3: Let's get serious and build the neural network
    # ------------------------------------------------------
    # [none, 128, 9]
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs], name = 'X')
    # [none, 6]
    Y = tf.placeholder(tf.float32, [None, config.n_classes], name = 'Y')

    print("-------X Y----------")
    print(X)
    X = tf.reshape(X, shape=[-1, X_AFTER_RESHAPE, Y_AFTER_RESHAPE])
    print(X)

    print(Y)
    Y = tf.reshape(Y, shape=[-1, N_CLASS_CONFIG])
    print(Y)

    # Weight Initialization
    def weight_variable(shape):
        # tra ve 1 gia tri random theo thuat toan truncated_ normal
        initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

    def bias_varibale(shape):
        initial = tf.constant(0.1, shape=shape, name='Bias')
        return tf.Variable(initial)

    # Convolution and Pooling
    def conv2d(x, W):
        # Must have `strides[0] = strides[3] = 1 `.
        # For the most common case of the same horizontal and vertices strides, `strides = [1, stride, stride, 1] `.
        return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME', name='conv_2d')

    def max_pool_2x2(x):
        return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1], padding='SAME', name='max_pool')

    def LSTM_Network(feature_mat, config):
        """model a LSTM Network,
          it stacks 2 LSTM layers, each layer has n_hidden=32 cells
           and 1 output layer, it is a full connet layer
          argument:
            feature_mat: ndarray feature matrix, shape=[batch_size,time_steps,n_inputs]
            config: class containing config of network
          return:
                  : matrix  output shape [batch_size,n_classes]
        """

        feature_mat_image = tf.reshape(feature_mat, shape=[-1, X_AFTER_RESHAPE, Y_AFTER_RESHAPE, 1])
        with tf.name_scope("CNN"):
            with tf.name_scope("CNN-layer-1"):
                W_conv1 = weight_variable([3, 3, 1, 2])
                b_conv1 = bias_varibale([2])

                tf.histogram_summary("W_conv1", W_conv1)
                tf.histogram_summary("b_conv1", b_conv1)
        
                print("----feature_mat_image-----")
                print(feature_mat_image.get_shape())

                h_conv1 = tf.nn.relu(conv2d(feature_mat_image, W_conv1) + b_conv1)
                # h_pool1 = max_pool_2x2(h_conv1)
                h_pool1 = h_conv1

      
            with tf.name_scope("CNN-layer-2"):
                # Second Convolutional Layer
                W_conv2 = weight_variable([3, 3, 2, 4])
                b_conv2 = weight_variable([4])
                tf.histogram_summary("W_conv2", W_conv2)
                tf.histogram_summary("b_conv2", b_conv2)
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                h_pool2 = max_pool_2x2(h_conv2)
                # h_pool2 = h_conv2

            with tf.name_scope("CNN-layer-3"):
                # Second Convolutional Layer
                W_conv3 = weight_variable([3, 3, 4, 1])
                b_conv3 = weight_variable([1])
                tf.histogram_summary("W_conv3", W_conv3)
                tf.histogram_summary("b_conv3", b_conv3)
                h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
                # h_pool2 = max_pool_2x2(h_conv2)
                h_pool3 = h_conv3

                h_pool3 = tf.reshape(h_pool3, shape=[-1, X_AFTER_RESHAPE, Y_AFTER_RESHAPE])
                feature_mat = h_pool3
                print("----feature_mat-----")
                print(feature_mat)

    
        with tf.name_scope("LSTM-layers"):
            # Exchange dim 1 and dim 0
            # Ban dau: [0,1,2] = [batch_size, 128, 9] => [batch_size, 32, 36]
            feature_mat = tf.transpose(feature_mat, [1, 0, 2])
            # New feature_mat's shape: [time_steps, batch_size, n_inputs] [128, batch_size, 9]
            print("----feature_mat-----")
            print(feature_mat)
            # exit()

            # Temporarily crush the feature_mat's dimensions
            feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])  # 9
            # New feature_mat's shape: [time_steps*batch_size, n_inputs]  # 128 * batch_size, 9

            tf.histogram_summary("self_W_hidden", config.W['hidden'])
            tf.histogram_summary("self_b_hidden", config.biases['hidden'])

            # Linear activation, reshaping inputs to the LSTM's number of hidden:
            hidden = tf.nn.relu(tf.matmul(
                feature_mat, config.W['hidden']
            ) + config.biases['hidden'])
            # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden] [128*batch_size, 32]

            print("--n_steps--")
            print(config.n_steps)
            print("--n_steps--")
            print(hidden)
            # exit()

            # Split the series because the rnn cell needs time_steps features, each of shape:
            hidden = tf.split(0, config.n_steps, hidden)  # (0, 128, [128*batch_size, 32])
            # New hidden's shape: a list of length "time_step" containing tensors of shape [batch_size, n_hidden]

            # Define LSTM cell of first hidden layer:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)

            # Stack two LSTM layers, both layers has the same shape
            lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

            # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
            outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
            # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

            print("------------------list-------------------")
            # print(outputs)
            # Get last time step's output feature for a "many to one" style classifier,
            # as in the image describing RNNs at the top of this page
            lstm_last_output = outputs[-1]  # Chi lay phan tu cuoi cung voi shape: [?, 32]

            print("------------------last outputs-------------------")
            print (lstm_last_output)

            tf.histogram_summary("self_W_output", config.W['output'])
            tf.histogram_summary("self_b_output", config.biases['output'])
            # Linear activation
            return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']

    pred_Y = LSTM_Network(X, config)   # shape[?,6]
    print("------------------pred_Y-------------------")
    print(pred_Y)

    with tf.name_scope('summaries'):
        # Loss,train_step,evaluation
        l2 = config.lambda_loss_amount * \
            sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
        # Softmax loss and L2
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
        train_step = tf.train.AdamOptimizer(
            learning_rate=config.learning_rate).minimize(cost)

        correct_prediction = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        # create a summary for our cost and accuracy
        tf.scalar_summary("accuracy", accuracy)
        tf.scalar_summary("cost", cost)
        # merge all summaries into a single "operation" which we can execute in a session 
    summary_op = tf.merge_all_summaries()

    # --------------------------------------------
    # step4: Hooray, now train the neural network
    # --------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.

    # Initializing the variables
    init = tf.initialize_all_variables()
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    best_accuracy = 0.0
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    # def convert_to_one_hot(aa):
    #     best = aa[0]
    #     for i in range(len(aa)):
    #         best = max(best, aa[i]) 

    #     result = []
    #     for i in range(len(aa)):   
    #         if (best == aa[i]):
    #             result.append(1)
    #         else:
    #             result.append(0)
    #     return result
    def convert_to_one_hot(aa):
        best = aa[0]
        for i in range(len(aa)):
            best = max(best, aa[i]) 

        # result = []
        for i in range(len(aa)):   
            if (best == aa[i]):
                return i
        #     else:
        #         result.append(0)
        # return result

if (FLAG == 'train') : # If it is the training mode
    with tf.Session() as sess:
        # tf.initialize_all_variables().run()
        sess.run(init)  # .run()
        f.write("---Save model \n")
        writer = tf.train.SummaryWriter(file_name + "/oppboard", graph=tf.get_default_graph())

        # Start training for each batch and loop epochs
        for i in range(config.training_epochs):
            X_train, y_train = shuffle(X_train, y_train, random_state=i*42)

            for start, end in zip(range(0, config.train_count, config.batch_size),  # (0, 7352, 1500)
                                range(config.batch_size, config.train_count + 1,
                                        config.batch_size)):  # (1500, 7353, 1500)
                # print(start)
                # print(end)

                sess.run(train_step, feed_dict={X: X_train[start:end],
                                            Y: y_train[start:end]})
            # Test completely at every epoch: calculate accuracy
            _summary_str, pred_out, accuracy_out, loss_out = sess.run([summary_op, pred_Y, accuracy, cost], feed_dict={
                X: X_test, Y: y_test})
            # if i == 1:
            #     cm = confusion_matrix(y_test, pred_out)
            #     pl.matshow(cm)
            #     pl.title('Confusion matrix of the classifier')
            #     pl.colorbar()
            #     pl.show()
                
            writer.add_summary(_summary_str, i)

            print("traing iter: {},".format(i) + \
                " test accuracy : {},".format(accuracy_out) + \
                " loss : {}".format(loss_out))
            best_accuracy = max(best_accuracy, accuracy_out)

            # Save the model in this session
            save_path = saver.save(sess, file_name + "/model.ckpt")
            print("Model saved in file: %s" % save_path)

        print("")
        print("final loss: {}").format(loss_out)
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("")
        # Write all output to file
        f.write("final loss:" + str(format(loss_out)) +" \n")        
        f.write("final test accuracy:" + str(format(accuracy_out)) +" \n")
        f.write("best epoch's test accuracy:" + str(format(best_accuracy)) + " \n")
        # Results

        _summary_str, pred_out, accuracy_out, loss_out = sess.run([summary_op, pred_Y, accuracy, cost], feed_dict={
            X: X_test, Y: y_test})

        print ('pred_out')
        print (pred_out)

        out_out = []
        for i in range(len(pred_out)):
            out_out.append(convert_to_one_hot(pred_out[i]))


        print ('out_out')
        # print(out_out)
        print (type(out_out))
        # output = tf.nn.softmax(pred_out).eval()
        # print ('output')
        # print (output)
        y_test_list = y_test.tolist()
        out_test = []
        for i in range(len(y_test_list)):
            out_test.append(convert_to_one_hot(y_test_list[i]))

        print(out_test[0])
        # print ('y_test')
        # print (y_test.tolist()[0])
        # print (type(y_test.tolist()))
        cm = confusion_matrix(out_test, out_out)
        pl.matshow(cm)
        pl.title('Confusion matrix of the classifier')
        pl.colorbar()
        pl.show()

else :
    # Running a new session
    print("Starting 2nd session...")
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        f.write("---Restore model \n")

        # Restore model weights from previously saved model
        saver.restore(sess, file_name+ "/model.ckpt")
        print("Model restored from file: %s" % save_path_name)
        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
            X: X_test, Y: y_test})

        # print("traing iter: {}," + \
        #       " test accuracy : {},".format(accuracy_out) + \
        #       " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

        print("")
        print("final loss: {}").format(loss_out)
        print("final test accuracy: {}".format(accuracy_out))
        print("best epoch's test accuracy: {}".format(best_accuracy))
        print("")
        # Write all output to file
        f.write("final loss:" + str(format(loss_out)) +" \n")
        f.write("final test accuracy:" + str(format(accuracy_out)) +" \n")
        f.write("best epoch's test accuracy:" + str(format(best_accuracy)) + " \n")

    # #------------------------------------------------------------------

f.write("Ended at \n")
f.write(str(datetime.datetime.now())+'\n')
f.write("------------- \n")
f.close()