#-*- coding: utf-8 -*-
#! /usr/bin/env python

import tensorflow as tf
from data_manager import DataManager
from layers import conv_layer, max_pool_2x2, full_layer
import os


MINIBATCH_SIZE = 50
STEPS = 1000

class Trainer:
    def __init__(self, path):
        self.path = path
        self.path_logs = os.path.join(path, "logs")
        self.data_manager = DataManager(path+'data/mnist/', batch_size=MINIBATCH_SIZE)
        self.saver = None

    # TODO
    def save_tfgraph_pb(self):
        print("save_tfgraph_pb")

    # eval.py의 코드와 겹침
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

    def run_training(self):

        # 그래프 구성하기
        x, keep_prob, y_conv = self.create_graph()

        # 최적화 그래프 만들기(loss function, optimizer, accuracy evaluator)
        y_, train_step, accuracy = self.create_optimizor(y_conv)


        # 세션 열기
        with tf.Session() as sess:
            # 전역 변수 초기화
            sess.run(tf.global_variables_initializer())

            # STEPS 수만큼 학습
            for i in range(STEPS):
                batch_X, batch_y = self.data_manager.next_batch()
                step = i+1

                # 학습 accuracy 계산해보기
                if i % 100 == 100-1:
                    train_accuracy = sess.run(accuracy, feed_dict={x: batch_X, y_: batch_y,
                                                                   keep_prob: 1.0})
                    print("step {}, training accuracy {}".format(step, train_accuracy))

                # batch_size 만큼의 1회 학습 실행
                sess.run(train_step, feed_dict={x: batch_X, y_: batch_y, keep_prob: 0.5})

                # checkpoint 저장하기
                if i % 1000 == 1000 - 1:
                    self.save_ckpt(sess, model_path=os.path.join(self.path_logs, "model"), step=step)


            # 저장된 model-XXXX checkpoint로 모델 불러오기
            self.restore_ckpt(sess, os.path.join(self.path_logs, "model-1000"))

            # 테스트를 위한 데이터 불러오기
            X, Y = self.data_manager.get_test_batch_data(test_batch_size=50)#self.data_manager.get_test_data()

            # 테스트 수행
            test_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y, keep_prob: 1.0})


            print("test accuracy: {}".format(test_accuracy))

    def save_ckpt(self, sess, model_path, step):
        print("save_ckpt")
        if (self.saver == None):
            self.saver = tf.train.Saver(max_to_keep=7, keep_checkpoint_every_n_hours=0.1)

        self.saver.save(sess, model_path, global_step=step)

    def restore_ckpt(self, sess, model_path):
        if (self.saver != None):
            self.saver.restore(sess, model_path)

trainer = Trainer("./")
trainer.run_training()




