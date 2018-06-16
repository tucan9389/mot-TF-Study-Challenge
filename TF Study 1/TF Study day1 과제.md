#### 참고코드를 이용하여 자신만의 CNN을 위한 MNIST 입력처리 코드를 깃헙에 업로드 하기

- `data_loader.py` : 데이터를 빠르게 불러와 메모리에 로드
- `input_ops.py` : 사용 모델 인풋 형식에 맞게 변환 (여러 데이터를 쉽게 바꿔끼고 섞을 수 있어야한다.)
- `preprocess_data.py` : 사용 모델에 필요한 인풋 전처리 

(* 과제는 MoT 깃헙에 강재욱님께 권한을 받아서 코드 업로드를 해주셔야합니다. 과제는 다음주에 다같이 리뷰합니다.)

#### 참고코드

1. 강재욱 github :  data loader (tf.gfile)<br>
   [https://github.com/jwkanggist/tf-cnn-model/blob/master/mnist_data_loader.py](https://github.com/jwkanggist/tf-cnn-model/blob/master/mnist_data_loader.py)
2. 이준호 github :  data_manager (tf.data)<br>
   [https://github.com/motlabs/mot-dev/blob/180506_tfdata_jhlee/lab11_tfdata_example/data_manager%20(mnist).ipynb](https://github.com/motlabs/mot-dev/blob/180506_tfdata_jhlee/lab11_tfdata_example/data_manager%20(mnist).ipynb)
3. Learning Tensorflow github : queue, multithread, tf.python_io.Tfrecord<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/08__queues_threads](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/08__queues_threads)

