#### 참고코드를 이용하여 자신만의 추상화, 라이브러리를 통해 간소화 된 CNN 코드를 깃헙에 업로드하기

- train.py : training 코드
- eval.py : evaluation 코드

(* 과제는 MoT 깃헙에 강재욱님께 권한을 받아서 코드 업로드를 해주셔야합니다. 과제는 다음주에 다같이 리뷰합니다.)

• 참고코드

1. 강재욱님 github :  cnn (`run_lenet5_save_pb_ckpt_with_tfboard.py`로 실행) <br>
   [https://github.com/jwkanggist/tensorflowlite](https://github.com/jwkanggist/tensorflowlite)
2. Learning Tensorflow github : cnn<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/04__convolutional_neural_networks](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/04__convolutional_neural_networks)
3. Learning Tensorflow github : abrtraction<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/07__abstractions](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/tree/master/07__abstractions)
4. Learning Tensorflow github : model exporting<br>
   [https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/blob/master/10__serving/Chapter10.ipynb](https://github.com/Hezi-Resheff/Oreilly-Learning-TensorFlow/blob/master/10__serving/Chapter10.ipynb)
5. 정찬희님 keras 코드 ([TF vs Keras _Jungchanhee.ipynb](참고코드/TF%20vs%20Keras%20_Jungchanhee.ipynb))

### 과제설명

- `data_manager.py`를 실행하면 됩니다.
- `data/mnist/` 경로 안에 0~9까지 폴더가 있고, 각각 폴더 안에 `28x28` 손글씨 이미지가 들어있습니다.
- `data_loader.py`: 이미지 파일들의 경로와 레이블을 가져옵니다.
- `input_ops.py`: 학습/테스트 데이터로 분리합니다. (`0.8:0.2`)
- `preprocess_data.py`: 단순 경로 데이터를 실제 데이터(`28x28x1`)로 만들어줍니다.
- `data_manager.py`에서는 batch 크기씩 나누어 입력 데이터를 제공합니다.



### 질문
Q. 
