# 빠른 논문 정리

## 2021-01-29 (A1-A8)

### A1 (Convolutional) 2017-03

> geometric matching을 위한 CNN의 가장 간단한 paper

이미지를 각각 shared weights VGG16으로 feature extraction하고 conv 2회에 fc를 붙여 matching한 다음 L2 normalization한 결과를 6D affine warp parameters로 간주한다. 그리고 나서 이걸 affine regression해서 얻은 affine transformation으로 warp한 source와 원본 target을 가지고 다시 feature extraction과 matching을 돌린 다음 그 결과를 thin plate spline regression해서 새로 geometric transformation parameters를 얻은 다음 이 최종 결과를 GT geometric transformation parameters와 비교해서 backpropagation한다.

  * weights shared in feature extraction
  * fully supervised
  * PCK 57%

### A2 (Recurrent) 2018-10

> self-supervised geometric matching with recurrency

일단 이미지를 각각 CAT-FCSS, VGGNet, ResNet 등으로 feature extraction하고 그 결과들의 cosine simillarity를 구하여 correlation map을 만든 다음에 이것을 geometric matching을 위해 UNet에 넣고 fc를 통과시켜서 geometric transformation 결과를 얻는다. 그리고 이 과정을 반복하기 위해 그냥 backpropagation하는 것이 아니라 target 쪽에서 그 다음 iteration부터는 correlation map을 feature extraction 단계에 집어넣고 그 결과에 직전 iteration의 결과를 더한다. 이렇게 해서 recurrent하게 geometry estimation을 할 수 있다.

UNet을 사용하는 것을 통해 affine이나 TPS를 사용하지 않는다.

  * weights not shared in feature extraction
  * self-supervised
  * PCK 77%

### A3 (Spatial) 2015-06

> geometric matching의 세 단계 모델을 제시하는 가장 기초적인 paper

먼저 localization network로 임의의 CNN에 이미지를 넣고 돌린 다음 fc를 통해 spatial transformation parameters를 뽑아낸다. 그걸 grid generator에 넣으면 해당 spatial transformation을 기존 이미지의 사이즈에 적용했을 때 각 pixel들이 어디로 가야 하는지를 알려 주는 새로운 grid 대응 map이 나온다. 그러면 그걸 가지고 sampler가 필요한 pixels만 sampling을 하면 sparse affine transformation이 적용되고 필요한 부분만 crop되어 그 결과가 정돈된 attended part를 추출할 수 있다.

CNN을 개선하고 싶을 때 임의의 위치에 넣을 수 있어 parallel computation을 방해하지 않고, mixed images disentangling에 사용할 수 있다.

  * self-supervised
  * PCK 84%

### A4 (Learning) 2019-03

> CycleGAN의 cycle-consistency를 time과 같은 frame을 기준으로 도입하여 correspondence를 확인할 때 더 robust하도록 만든 paper

새로운 frame의 patch와 기존 frame image를 각각 ResNet50을 이용하여 1/8 * 1/8 크기로 feature extraction하고 이걸 product해서 affine matrix를 얻는다. 그리고 이걸 conv 2번과 fc를 돌려서 6D affine transformation parameters로 정리한다. 그리고 그 결과를 새로운 기존 frame image의 feature extraction 결과에 적용해서 affine transformation한다. 그러면 그게 기존 frame에서 해당 patch의 위치를 찾은 다음 해당 patch와 같게 affine transformation한 것이다.

  * soft-supervised
  * PCK varies

### A5 (Neighbourhood) 2018-10

> initial suggestion of neighbour consensus

textureless images의 경우 그냥 벽과 같이 별 내용이 없는 부분은 도저히 matching하기 어렵다. 따라서 해당 사진에서 액자가 어디로 움직였는지 등을 가지고 matching 결과를 estimation할 수 있다. 당연히 이를 위해서는 dense feature extraction이 필요하고 이걸 가지고 source 2D 그리고 target 2D 해서 4D의 feature match map을 만든다. 그걸 normalization하면 4D correlation map이 된다. 일단 그 4D map에 image A 기준 match ratio와 image B 기준 match ratio를 곱해서 soft mutual nearest neighbourhood filtering을 적용한다. 이건 correlation map을 세제곱하고 상수로 나누는 효과가 있어서 correlation map 안의 scalar value들이 보다 0과 1로 나뉜다. 그리고 나서 그걸 neighbour consensus network에 넣어서 1 4D map - N1 4D maps - N2 * N1 4D maps - N2 4D maps 해서 CNN을 돌린 다음 fc로 다시 1 4D map 으로 만든다. 이걸 할 때 map의 transpose도 network에 넣은 다음 그걸 다시 transpose한 값도 따로 구한 다음 둘을 더한다. 이렇게 하면 A에서 B로 가는 transformation과 B에서 A로 가는 transformation은 서로 역함수 관계여야 한다는 점을 잘 지킬 수 있게 된다. 그리고 거기에 다시 soft mutual nearest neighbourhood filtering을 적용한 다음 softmax한다. 이 결과는 여전히 4D map인데 재미있는 점은 모든 scalar value들이 0보다 크고 image별로 합을 구하면 1이라는 점이다. 따라서 이걸 image A의 해당 pixel이 image B의 해당 pixel과 연관이 있는지에 대한 4D 확률로 볼 수 있다. 그래서 image A에서 특정 patch를 잡은 다음 4D map에 넣어서 argmax하면 image B의 related patch를 찾을 수 있다.

어떤 transformation이 적용되었는지는 따로 찾아줘야 한다.

  * self-supervised
  * PCK 78%

### A6 (Semantic) 2019-04

> 형태는 target으로 보내고 style은 source로 보내는 양방향 동기화를 제안한 paper

일단 deep features 기법을 이용해서 feature extraction을 한다. 그러면 source 쪽에서는 attribute transfered feature가 나오고 target 쪽에서는 warped target feature가 나온다. 먼저 semantic matching의 측면에서 이 둘의 correlation map을 구한 것을 encoder-decoder에 넣고 6D affine 또는 thin spline transformation field를 구한다. 그리고 attribute transfer의 측면에서 이 둘을 target 쪽에 confidence를 적용하여 blending한 것을 CNN에 넣고 돌리면 target의 feature가 extract되어 source 쪽에 transfer된 image가 나온다. 그리고 아까 구한 affine transformation map을 이 image에 적용하면 target의 style에 source의 형태를 적용한 것이 나온다. 그런데 이렇게만 하면 원본과 너무 멀어질 수 있으므로 attribute transfer network 쪽에는 binary random variable을 도입하여 직전의 결과를 그대로 내놓는 skip connection을 높은 확률로 적용한다.

  * weights shared in feature extraction
  * self-supervised
  * PCK 87%

### A7 (Universal) 2016-06

> semantic matching과 geometric matching을 한꺼번에 할 수 있는 model

미리 images 자체에 label이 붙어 있고 images 안의 keypoint들에 annotation이 다 붙어 있는 dataset을 집어넣고 먼저 dense하게 FCNN을 돌린다. 그리고 나서 각각의 keypoint들에 대해서 정해진 kernel size k에 맞게 k * k patch를 잘라내서 각각 [A3]을 돌린다. 그렇게 해서 얻은 새로운 k * k patch들의 모음은 기존 이미지와 비교하면 patch size가 k배인데, stride k로 convolution해서 배율을 맞춘다. 그리고 나서 images 간에 같은 keypoint끼리의 거리는 최소화하고, 다른 keypoint끼리의 거리는 최소한 m의 margin이 있도록 loss function을 설정하여 backpropagation한다. 그러면 label과 함께 학습했기 때문에 semantic estimation이 가능하면서도, 모르는 image에 keypoint들을 설정해서 target image와 함께 주고 돌리면 target image에서 해당 keypoint와 semantic의 관점에서 같은 keypoint들의 위치를 찾아준다.

  * fully supervised
  * PCK very high

### A8 (DGC-Net) 2018-10

> robust geometric correspondence network in strong geometric transformations

현실에서는 아주 다른 각도에서 보는 homography나 약간 다른 위치에서 보는 affine 그리고 이와 비슷하게 thin plate spline 변환보다 훨씬 심하게 사진이 다를 수 있기 때문에 이런 상황에서도 여전히 correspondence map을 구성하려면 deep 그리고 dense한 network를 구성해야 할 것이다. 먼저 source 그리고 target image를 Siamese VGG16에 넣는다. 그렇게 해서 얻은 source 그리고 target feature vector를 가지고 먼저 coarse layer에서 correlation map을 구한다. 이것은 두 feature vector의 scalar product로 구할 수 있다. 그리고 이것을 5 layer의 CNN으로 구성된 correspondence map decoder에 넣는다. 그러면 그게 그 layer의 correspondence map이 된다. 그리고 나서는 먼저 source 그리고 target feature vector와 위의 layer에서 얻은 correspondence map을 bilinear upsampler에 넣어서 2 * 2 upsample한다. 그리고 source feature vector를 correspondence map대로 warp한다. 그리고 그 결과와 target feature vector 그리고 correspondence map을 모두 더한다. 그리고 그것을 해당 layer의 새로운 correspondence map으로 사용한다. 그렇게 해서 전체 5 layer pyramid 구조를 만든다. 마지막 layer에 사용한 두 feature vector와 마지막 layer에서 나온 correspondence map을 concatenate하여 4 layer CNN으로 이루어진 matchability decoder에 넣으면 matchability map을 얻는다. loss로는 matchability map에서 BLE loss를 얻고 GT correspondence map과 model에서 얻은 correspondence map의 L1 distance를 match에 관한 binary mask를 곱해서 원소별로 다 더한 다음 confidence를 곱하는 일을 layer마다 해서 모두 더한 loss를 얻은 다음 둘을 가중합한 것을 사용한다. 마지막 layer에서 더 풍부한 convolution 결과를 활용하기 위해 correspondence map decoder에 layer 3부터 dilation을 주면 좋다.

모델이 복잡하지만 pyramid 구조의 이점을 살리면서 CNN도 있고 warp도 하고 matchability map도 loss로 활용하면서 correspondence map도 loss로 활용하기 때문에 dense geometric correspondence network에서 train에 활용할 수 있는 거의 모든 것을 다 활용하는 모델이라고 할 수 있다.

  * self-supervised
  * PCK varies

### A9 (GLU-Net) 2019-12

> geometric, semantic 그리고 optical flow 문제를 단일 weights로 해결하는 universal model

일단 VGG16으로 feature extraction을 두 번 한다. 그리고 resolution을 줄인 원본 image로도 두 번 한다. 줄인 image로 두 번 해서 나온 결과는 줄인 image의 feature vector이다. 이걸 가지고 일단 global correlation map을 만든다. 이때 cyclic consistency를 적용한다. 그리고 나서 이것을 correspondence map decoder에 집어넣고 거기서 원래 feature vector를 뺀다. 그리고 이걸 scale 2 deconvolution한다. 이걸로 줄인 이미지로 한 번 feature extraction한 결과 중에서 source feature vector를 warp하고, 그 결과와 target feature vector로 구한 local correspondence map을 correspondence map decoder에 집어넣을 때 scale 2 deconvolution한 것을 같이 넣어서 돌린다. 그리고 그 결과에 또 deconvolution 결과를 집어넣는다. 그리고 거기에다가 decoder의 결과를 refine network에 집어넣은 것을 더한다. 앞의 layer들도 비슷하다. 이렇게 text로 정리한 것을 다시 보는 것보다 paper의 figure 3을 직접 보는 것이 훨씬 빠를 것이다.

중요한 점은 resolution을 줄인 것을 dense하게 feature extraction한 것을 가지고 global correlation map을 구할 수 있고, 그리고 그 아래 layer들에서 위의 layer에서 계산한 map을 이용해서 local correlation map을 구할 수 있기에 global 그리고 local correlation을 모두 고려할 수 있다는 것이다.

모델이 굉장히 복잡하지만 image를 crop하는 일이 없기 때문에 더 이상 어차피 sparse하게 작업할 것이라고 중요한 정보가 어디 있는지도 모르는데 마구 crop하는 일이 없어서 정보의 손실을 막을 수 있다. resolution을 바꾸는 것은 dense한 처리를 위해서는 필수적이고 dense한 image 처리는 warp하기 전에 image를 가지고 correspondence map을 만드는 것을 의미한다.

  * self-supervised
  * PCK varies

#### Quick Notes

  * Geometric Matching: Image A에 나오는 것을 다른 각도에서 찍은 사진이 image B인데, 같은 물체를 sparse하게 찾아내서 image A를 warp하여 image B처럼 geometric transformation을 해라.
  * Semantic Matching: Image A에 나오는 것과 비슷한 것을 다른 배경에서 찍은 사진이 image B인데, 비슷한 물체를 sparse하게 찾아내서 image A를 warp하여 image B처럼 geometric transformation을 해라.
  * Optical Flow: Video와 같이 image들의 시계열 data가 주어지면 이것을 frame별로 쪼갠 다음 frame들을 dense하게 서로 비교해서 frame마다 image 속의 물체들이 어떻게 움직였는지를 frame A를 warp하여 frame B처럼 geometric transformation을 해서 보여라.

## 2021-02-01 (A11-A12)

### A11 (GOCor) 2020-09

> 별도의 dense convolution 없이도 global weights를 고려해서 estimation을 진행하는 모델

correspondence map에 penalization weights를 곱해서 맞는 cell은 강하게 하고 틀린 cell은 약하게 한 다음 이것의 LSE와 correspondence map을 4D convolution한 결과의 LSE 그리고 correspondence map의 제곱을 적절히 가중합한 것을 loss로 사용하여 정확도와 smoothness를 올리고 regularization을 강화한다.

모듈로 끼워넣을 수 있고 간단한데 성능이 좋아서 최근에서야 나온 이유에 의문이 생기게 된다.

  * Fully supervised
  * PCK very high

### A12 (Correspondence) 2020-03

> dense annotation 없이도 dense correlation map을 만들어서 sparse estimation을 보다 global하게 진행하는 모델

source와 target을 각각 feature extraction한 다음 3 * 3 크기로 벡터들을 뽑아서 9개의 cosine distance를 구한다. 그리고 이것을 길이가 9인 벡터로 만들어서 W * H * 9 크기의 행렬을 만든다. 그리고 이것을 2번 2D convolution하여 3개의 행렬을 만들어서 모두 concatenation한다. 그리고 나서 source와 target의 feature map에 대해 cosine correlation을 구하고 source와 target의 feature map을 앞에서 말한 대로 조작한 것에 대해 cosine correlation을 구한 다음 이 둘의 transpose matrix까지 해서 넷을 각각 non-isotropic 4D convolution layers인 adaptive neighbourhood consensus filter에 넣어서 refine하고 그 결과를 sum한다. 그리고 나서 soft mutual NN filtering과 softmax를 진행한다.

  * fully supervised
  * PCK very high

## 2021-02-02 (A13-A14)

### A13 (Semantic) 2020-06

> semantic correspondence를 사용해서 semantic matching 문제를 optimal transport 문제로 바꾸는 논문

source와 target을 각각 feature extraction한 다음 가장 좋은 layers를 모아서 features를 구성하고 cosine correlation을 사용해서 correlation map을 만든 다음 M = 1 - C로 M을 만든다. feature extraction에서의 가장 마지막 layer를 GAP 돌리고 FC에 넣고 staircase function으로 정리하고 softmax에 넣고 argmax로 가장 잘 들어맞는 class를 찾는다. 그리고 방금 찾은 class를 가지고 feature extraction에서의 가장 마지막 layer와 FC의 weights를 곱해서 W * H CAM을 만든다. 이걸 row와 column을 기준으로 평균을 내고 이것을 사용해서 T의 후보들을 구한 다음 T * M이 최소인 T star를 구한다. 이때 negative entropic regularizer를 사용한다. 그리고 미리 설정한 offset에 비해서 각 annotated pixels가 얼마나 움직였는지를 Gaussian mask로 변환해 둔다. 이것과 T star를 Hough space matching confidence 공식에 집어넣고 돌리면 matching confidence를 얻을 수 있는데 여기에 argmax를 사용하면 어떤 pixel pair가 가장 유망한지 알 수 있고 따라서 어떤 annotated source pixel에 대해서 가장 잘 match되는 target pixel을 찾을 수 있다.

Hough space에 대해 처음 들어봐서 굉장히 복잡하고 어려웠는데 이걸 도입하는 것이 computation 관점에서 매우 유리할 것 같긴 하다는 생각이 들었다.

  * self-supervised
  * PCK varies
