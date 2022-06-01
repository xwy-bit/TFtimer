from numpy import dtype
import tensorflow.compat.v1 as tf
Timer_ops = tf.load_op_library('/home/xwy/asc/exp/tensorflow/timer/v02/libtimer.so')
tf.disable_v2_behavior()
g = tf.Graph()
N = 20000
with g.as_default():
    a = tf.constant([1])
    # a , time0,time1 = Timer_ops.timer(a)
    w2 = tf.random_normal([100, 100], mean=.0, stddev=4,seed = 102)
    w0 = tf.random_normal([100, 100], mean=.0, stddev=4,seed = 100)
    w1 = tf.random_normal([100, 100], mean=.0, stddev=4,seed = 101)
    w3 = tf.random_normal([100, 100], mean=.0, stddev=4,seed = 102)
    w3_ = tf.cast(w3,dtype=tf.int32)
    w3_ ,time0,time1 = Timer_ops.timer(w3_)
    # w3_ = tf.cast(w3,dtype = tf.float32)
    w01 = tf.matmul(w0,w1)
    for i in range(N):
        w01 = tf.matmul(w01,w0)
    w23 = tf.matmul(w2,w3)
    # for i in range(N):
    #     w23 = tf.matmul(w23,w3)
    w_all = tf.matmul(w23,w01)
    w_all = tf.cast(w_all,dtype = tf.int32)
    w_all,time2,time3 = Timer_ops.timer(w_all)
    time_list = [time0,time1,time2,time3]
with tf.Session(graph = g) as sess:
    model = tf.global_variables_initializer()
    sess.run(model)
    _,time_list_out = sess.run([w_all,time_list])
    time_sec = time_list_out[2]-time_list_out[0]
    time_mil = time_list_out[3]-time_list_out[1]
    time_cost = time_sec+time_mil*1.0/1e6
    print("time-->",time_cost)