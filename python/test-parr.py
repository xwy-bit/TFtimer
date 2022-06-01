from ctypes import cast

from numpy import dtype
import tensorflow.compat.v1 as tf
Timer_ops = tf.load_op_library('/home/xwy/asc/exp/tensorflow/timer/v02/libtimer.so')
tf.disable_v2_behavior()
def sort_time(a1,a2,a3,a4,b1,b2,b3,b4,sum1,sum2):
    pass

g = tf.Graph()
with g.as_default() :
    coef = tf.constant([1])
    # coef , time0 ,time1 = Timer_ops.timer(coef)
    w0 = tf.random_normal([10, 10], mean=.0, stddev=4,seed = 100)
    w1 = tf.random_normal([10, 10], mean=.0, stddev=4,seed = 101)
    w2 = tf.random_normal([10, 10], mean=.0, stddev=4,seed = 102)
    w3 = tf.random_normal([10, 10], mean=.0, stddev=4,seed = 102)
    w3_ = tf.cast(w3,dtype=tf.int32)
    w3_ ,time0,time1 = Timer_ops.timer(w3_)
    w01 = tf.matmul(w0,w1)
    for i in range(100):
        w01 = tf.matmul(w01,w0)
    w23 = tf.matmul(w2,w3)
    for i in range(100):
        w23 = tf.matmul(w23,w2)    
    w_all = tf.matmul(w01,w23)
    sum_reduce = tf.reduce_sum(w_all,[0,1])
    sum_reduce = tf.cast(sum_reduce,tf.int32)
    sum_reduce , time2 ,time3 = Timer_ops.timer(sum_reduce)
    time_list = [time0,time1,time2,time3]


with tf.Session(graph = g) as sess:
    model = tf.global_variables_initializer()
    sess.run(model)
    sum_reduce_out,time_list_out,_ = sess.run([sum_reduce,time_list,w3_])
    print("result==>",sum_reduce_out)
    time_sec = time_list_out[2]-time_list_out[0]
    time_mil = time_list_out[3]-time_list_out[1]
    time_cost = time_sec+time_mil*1.0/1e6
    print("time-->",time_cost)

