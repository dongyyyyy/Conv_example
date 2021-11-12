# Conv_example

Convolution Neural Network

Forward

and 

BackWard 구현 ( Using MatMul Calcul)

구현 내용
 - 2D Convolution Layer
 - 2D Maxpool
 - Global Average Pool
 - ReLU
 - Fully Connected Layer

### our_results.txt  => 구현한 Convolutional Layer로 학습을 수행했을 때의 결과
### tensor_results.txt => 구현한 Convolutional Layer와 동일한 Architecture로 Pytorch로 구현했을 때의 결과

## Problem
#### - im2col 후 MatMul연산을 수행하는데, Pytorch CPU 버전과 비교했을 때 속도 차이가 심각하게 많이 나고 있는 상황이다.
#### - 속도를 빠르게 할 수 있는 방법에 대해 고민할 필요성이 존재함
#### - Backpropagation에서 Conv전 구간에서 reshape부분이 im2col로 구현되어 있지 않음(수정 필요)


