[TF Pattern Study 수업방향 Docs](https://docs.google.com/document/d/1k_aXsx4CB0U1DP89p7Q0RqLOvCqnBnNLX3bFoK1rWf0/edit?usp=sharing)

### 참고코드를 이용하여 자신만의 추상화, 라이브러리를 통해 간소화 된 CNN 코드를 깃헙에 업로드하기

- 데이터셋 선택 자유(MNIST?, CIFAR?)
- 모델 선택 자유(MoT Labs 이전 발표자료 참고)

#### 과제 제출 코드

- `train.py` : training 코드
- `eval.py` : evaluation 코드

(* 과제는 MoT 깃헙에 강재욱님께 권한을 받아서 코드 업로드를 해주셔야합니다. 과제는 다음주에 다같이 리뷰합니다.)

#### 참고코드

1. 강재욱님 github :  cnn ([run_lenet5_save_pb_ckpt_with_tfboard.py](https://github.com/jwkanggist/tensorflowlite/blob/master/run_lenet5_save_pb_ckpt_with_tfboard.py)로 실행) <br>
   [https://github.com/jwkanggist/tensorflowlite](https://github.com/jwkanggist/tensorflowlite)
2. Learning Tensorflow github : cnn<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/04__convolutional_neural_networks](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/04__convolutional_neural_networks)
3. Learning Tensorflow github : abrtraction<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/07__abstractions](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/07__abstractions)
4. Learning Tensorflow github : model exporting<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/blob/master/10__serving/Chapter10.ipynb](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/blob/master/10__serving/Chapter10.ipynb)
5. 정찬희님 keras 코드 ([TF vs Keras _Jungchanhee.ipynb](참고코드/TF%20vs%20Keras%20_Jungchanhee.ipynb))

### 과제설명

- `train.py`
  1. data_manager.py의 DataManager로 데이터를 준비

학습을 끝내고 `test_accuracy = sess.run(accuracy, feed_dict={x: X, y_: Y, keep_prob: 1.0})` 를 실행시켜서 테스트 데이터로 accurary를 확인해보면 학습이 되지 않은것과 동일한 결과(0.1)가 나옵니다 ㅠ 학습시 Variable이 저장이 안 된 걸까요..

- `eval.py` : evaluation 코드

// 원래 데이터셋은 하나이고, 메모리위에서 train/test 데이터셋으로 나눠서 들고 있었는데, 파일을 `train.py`와 `eval.py`로 나누면서 같은 인터프리트 실행 안에서 같은 train/test셋을 공유하게끔 하지 못하고 있습니다. DataManager는 싱글톤으로 만드는 방법을 찾아봐야겠습니다.



### 질문
Q. 그래프 설계 방법은 어떻게 알 수 있을까?<br>
Q. 그래프 아키텍처를 적용시킬때 어떤 레퍼런스를 보고 적용시켜야되는지..
☞ 예를들면, MNIST 데이터셋으로 손글씨 분류를 한다했을때, **어떤 레이어들을 연결시키는게 좋을지 어떻게 알 수 있을까요?**



Q. Ch4. 합성곱 신경망 MNIST 예제(p.82)에서 드롭아웃 인풋이 1024인 이유?
Q. 비슷한 맥락에서 CNN 피처맵 갯수가 32, 64개인 이유?

