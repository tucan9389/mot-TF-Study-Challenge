#-*- coding: utf-8 -*-
#! /usr/bin/env python

import tensorflow as tf
from data_manager import DataManager
from layers import conv_layer, max_pool_2x2, full_layer
import os

class Evaluation:
    def __init__(self, path):
        print("Evaluation")
        self.path = path
        self.path_logs = os.path.join(path, "logs")
        self.data_manager = DataManager(path + 'data/mnist/', batch_size=1)

    def create_graph(self):
        # TODO: get_Variable 써보기?

        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        conv1 = conv_layer(x, shape=[5, 5, 1, 32])
        conv1_pool = max_pool_2x2(conv1)

        conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
        conv2_pool = max_pool_2x2(conv2)

        conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])
        full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

        keep_prob = tf.placeholder(tf.float32)
        full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

        y_conv = full_layer(full1_drop, 10)

        return x, keep_prob, y_conv

    def create_optimizor(self, y_conv):
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return y_, train_step, accuracy

    def run_evaluation(self):

        # ------------ train.py의 run_training()와 동일 ------------------
        # 그래프 구성하기
        x, keep_prob, y_conv = self.create_graph()

        # 최적화 그래프 만들기(loss function, optimizer, accuracy evaluator)
        y_, train_step, accuracy = self.create_optimizor(y_conv)
        # --------------------------------------------------------------


        # Saver 만들기
        saver = tf.train.Saver()

        # 세션 열기
        with tf.Session() as sess:

            # 저장된 model-XXXX checkpoint로 모델 불러오기
            saver.restore(sess, os.path.join(self.path_logs, "model-1000"))

            # 테스트를 위한 데이터 불러오기
            X, Y = self.data_manager.get_test_batch_data()  # self.data_manager.get_test_data()

            # 테스트 수행
            test_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y, keep_prob: 1.0})

        print("Accuracy: {}".format(test_accuracy))
        # Accuracy: 0.09037 ????


eval = Evaluation("./")
eval.run_evaluation()