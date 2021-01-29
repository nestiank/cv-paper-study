# 빠른 논문 정리

## 2021-01-29 (A1-A8)

### A1 (Convolutional) 2017-03

> geometric matching을 위한 CNN의 가장 간단한 논문

이미지를 각각 shared weights VGG16으로 feature extraction하고 conv 2회에 fc를 붙여 matching한 다음 L2 normalization한 결과를 6D affine warp parameters로 간주한다. 그리고 나서 이걸 affine regression해서 얻은 affine transformation으로 warp한 source와 원본 target을 가지고 다시 feature extraction과 matching을 돌린 다음 그 결과를 thin plate spline regression해서 새로 geometric transformation parameters를 얻은 다음 이 최종 결과를 GT geometric transformation parameters와 비교해서 backpropagation한다.

  * weights shared in feature extraction
  * fully supervised
  * PCK 57%

### A2 (Recurrent) 2018-10

> self-supervised geometric matching with recurrency

일단 이미지를 각각 CAT-FCSS, VGGNet, ResNet 등으로 feature extraction하고 그 결과들의 cosine simillarity를 구하여 correlation map을 만든 다음에 이것을 geometric matching을 위해 UNet에 넣고 fc를 통과시켜서 geometric transformation 결과를 얻는다. 그리고 이 과정을 반복하기 위해 그냥 backpropagation하는 것이 아니라 target 쪽에서 그 다음 iteration부터는 correlation map을 feature extraction 단계에 집어넣고 그 결과에 직전 iteration의 결과를 더한다. 이렇게 해서 recurrent하게 geometry estimation을 할 수 있다.

UNet을 사용하는 것을 통해 affine이나 TPS를 사용하지 않는다.

  * weights not shared in feature extraction
  * fully self-supervised
  * PCK 77%

### A3 (Spatial) 2015-06

> geometric matching의 세 단계 모델을 제시하는 가장 기초적인 논문

먼저 localization network로 임의의 CNN에 이미지를 넣고 돌린 다음 fc를 통해 spatial transformation parameters를 뽑아낸다. 그걸 grid generator에 넣으면 해당 spatial transformation을 기존 이미지의 사이즈에 적용했을 때 각 pixel들이 어디로 가야 하는지를 알려 주는 새로운 grid 대응 map이 나온다. 그러면 그걸 가지고 sampler가 필요한 pixels만 sampling을 하면 sparse affine transformation이 적용되고 필요한 부분만 crop되어 그 결과가 정돈된 attended part를 추출할 수 있다.

CNN을 개선하고 싶을 때 임의의 위치에 넣을 수 있어 parallel computation을 방해하지 않고, mixed images disentangling에 사용할 수 있다.

  * fully self-supervised
  * PCK 84%

### A4 (Learning) 2019-03

> CycleGAN의 cycle-consistency를 time과 같은 frame을 기준으로 도입하여 correspondence를 확인할 때 더 robust하도록 만든 논문

새로운 frame의 patch와 기존 frame image를 각각 ResNet50을 이용하여 1/8 * 1/8 크기로 feature extraction하고 이걸 product해서 affine matrix를 얻는다. 그리고 이걸 conv 2번과 fc를 돌려서 6D affine transformation parameters로 정리한다. 그리고 그 결과를 새로운 기존 frame image의 feature extraction 결과에 적용해서 affine transformation한다. 그러면 그게 기존 frame에서 해당 patch의 위치를 찾은 다음 해당 patch와 같게 affine transformation한 것이다.

  * soft-supervised
  * PCK varies

### A5 (Neighbourhood) 2018-10

> initial suggestion of neighbour consensus

textureless images의 경우 그냥 벽과 같이 별 내용이 없는 부분은 도저히 matching하기 어렵다. 따라서 해당 사진에서 액자가 어디로 움직였는지 등을 가지고 matching 결과를 estimation할 수 있다. 당연히 이를 위해서는 dense feature extraction이 필요하고 이걸 가지고 source 2D 그리고 target 2D 해서 4D의 feature match map을 만든다. 그걸 normalization하면 4D correlation map이 된다. 일단 그 4D map에 image A 기준 match ratio와 image B 기준 match ratio를 곱해서 soft mutual nearest neighbourhood filtering을 적용한다. 이건 correlation map을 세제곱하고 상수로 나누는 효과가 있어서 correlation map 안의 scalar value들이 보다 0과 1로 나뉜다. 그리고 나서 그걸 neighbour consensus network에 넣어서 1 4D map - N1 4D maps - N2 * N1 4D maps - N2 4D maps 해서 CNN을 돌린 다음 fc로 다시 1 4D map 으로 만든다. 이걸 할 때 map의 transpose도 network에 넣은 다음 그걸 다시 transpose한 값도 따로 구한 다음 둘을 더한다. 이렇게 하면 A에서 B로 가는 transformation과 B에서 A로 가는 transformation은 서로 역함수 관계여야 한다는 점을 잘 지킬 수 있게 된다. 그리고 거기에 다시 soft mutual nearest neighbourhood filtering을 적용한 다음 softmax한다. 이 결과는 여전히 4D map인데 재미있는 점은 모든 scalar value들이 0보다 크고 image별로 합을 구하면 1이라는 점이다. 따라서 이걸 image A의 해당 pixel이 image B의 해당 pixel과 연관이 있는지에 대한 4D 확률로 볼 수 있다. 그래서 image A에서 특정 patch를 잡은 다음 4D map에 넣어서 argmax하면 image B의 related patch를 찾을 수 있다.

어떤 transformation이 적용되었는지는 따로 찾아줘야 한다.

  * fully self-supervised
  * PCK 78%

### A6 (Semantic) 2019-04

> 형태는 target으로 보내고 style은 source로 보내는 양방향 동기화를 제안한 논문

일단 deep features 기법을 이용해서 feature extraction을 한다. 그러면 source 쪽에서는 attribute transfered feature가 나오고 target 쪽에서는 warped target feature가 나온다. 먼저 semantic matching의 측면에서 이 둘의 correlation map을 구한 것을 encoder-decoder에 넣고 6D affine 또는 thin spline transformation field를 구한다. 그리고 attribute transfer의 측면에서 이 둘을 target 쪽에 confidence를 적용하여 blending한 것을 CNN에 넣고 돌리면 target의 feature가 extract되어 source 쪽에 transfer된 image가 나온다. 그리고 아까 구한 affine transformation map을 이 image에 적용하면 target의 style에 source의 형태를 적용한 것이 나온다. 그런데 이렇게만 하면 원본과 너무 멀어질 수 있으므로 attribute transfer network 쪽에는 binary random variable을 도입하여 직전의 결과를 그대로 내놓는 skip connection을 높은 확률로 적용한다.

  * weights shared in feature extraction
  * PCK 87%

### A7 (Universal) 2016-06

> semantic matching과 geometric matching을 한꺼번에 할 수 있는 모델

미리 images 자체에 label이 붙어 있고 images 안의 keypoint들에 annotation이 다 붙어 있는 dataset을 집어넣고 먼저 dense하게 FCNN을 돌린다. 그리고 나서 각각의 keypoint들에 대해서 정해진 kernel size k에 맞게 k * k patch를 잘라내서 각각 [A3]을 돌린다. 그렇게 해서 얻은 새로운 k * k patch들의 모음은 기존 이미지와 비교하면 patch size가 k배인데, stride k로 convolution해서 배율을 맞춘다. 그리고 나서 images 간에 같은 keypoint끼리의 거리는 최소화하고, 다른 keypoint끼리의 거리는 최소한 m의 margin이 있도록 loss function을 설정하여 backpropagation한다. 그러면 label과 함께 학습했기 때문에 semantic estimation이 가능하면서도, 모르는 image에 keypoint들을 설정해서 target image와 함께 주고 돌리면 target image에서 해당 keypoint와 semantic의 관점에서 같은 keypoint들의 위치를 찾아준다.

  * PCK very high

### A8 (DGC-Net) 2018-10

>


