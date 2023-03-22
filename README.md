# AlexNet 구현 프로젝트

* [사용한 모델](#Used-model)
* [파라메터](#Model-parameter)

* [기능 구현](#Implementation)
* [실험 결과](#Result)
 	* [1차 Test](#1st-Test)
 
---
### Used model
 - AlexNet
---
### Model parameter
 - 6,7번 레이어를 둘로 나누지 않고 5번 레이어의 tensor를 concatenate한 뒤 단일 레이어로 연산.
 - Layer 1 : 11x11 Conv layer(x2), stride=4, no padding, 3->48ch, batch norm, maxpool
 - Layer 2: 5x5 Conv Layer(x2), stride=1, 2 zero-padding, 48->128ch, batch norm, maxpool
 - Layer 3: 3x3 Conv Layer(x2), stride=1, 1 zero-padding, 128->192ch, no batch, no maxpool
 - Layer 4: 3x3 Conv Layer(x2), stride=1, 1 zero-padding, 192->192ch, no batch, no maxpool
 - Layer 5: 3x3 Conv Layer(x2), stride=1, 1 zero-padding,, 192->128ch, no batch, maxpool
 - Layer 6: 9216(6*6*128)x4096 Linear Layer
 - Layer 7: 4096x2 Linear Layer, no Activation
 - Activation : ReLU(1~6 Layer)
 - MaxPool : 3x3 maxpool with 2 stride
 - Dropout : p=0.5
 - Adam Optimizer
 - 특징 : Loss function으로 CrossEntropyLoss를 사용하였기 때문에, Training시 결과값에 Softmax를 적용하지 않음.


---
### Implementation
 - 1차. Trainer, Model, DataLoader, Logger 프로토타입 구현(2023-03-22)



---
### Result
#### 1st Test
 - 전체 데이터셋을 train과 test로 나누지 않고 학습이 진행되는지 확인하는 용도
 - batch size 64, epoch 20까지 진행하며 정상적인 학습 확인 단계
 - learning rate 0.001, weight decay 0.005
 	- test 결과 값이 [1,0]과 같이 한 class로 통일되는 현상 발견
	- learning rate가 0.0001일 때 정상적인 학습이 진행되는 것을 확인. 학습 시작


 -