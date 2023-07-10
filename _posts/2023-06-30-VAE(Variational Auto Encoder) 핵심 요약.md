---
title: "VAE(Variational AutoEncoder) 핵심 요약"

excerpt: "Vaiaional Autoencoder 모델 알고리즘 요약!"

date: 2023-06-30 16:50:00 +0900
lastmod: 2023-06-30 16:50:00 +0900 # sitemap.xml에서 사용됨

author_profile: false # 왼쪽부분 프로필을 띄울건지

categories:
  - Model

tags:
  - ML
  - VAE

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---

> 막연하게만 이해하고 있던 것들을 정리를 통해 구체화 해보자! VAE 부터 시작!

# Variantional AutoEcoder에 대해 알아보자!

## VAE와 AutoEncoder 차이가 뭐야?

---

![AE](https://namu-tree-kim.github.io/assets/images/ae.jpg "AE"){: .align-center}

![VAE](https://namu-tree-kim.github.io/assets/images/vae.jpg "VAE"){: .align-center}
위 그림과 같이 AutoEncoder와 Variational AutoEncoder의 큰 구조는 같다.

1. Input
2. Encoder
3. Latent Varible
4. Decoder
5. Reconstruction

하지만 두 모델의 접근 관점은 신기하게도 정반대이다.

### AutoEncoder

---

![AE](https://namu-tree-kim.github.io/assets/images/ae.jpg "AE"){: .align-center}

AutoEncoder의 목표는 Manifold learning, Input 데이터를 차원 축소하는 것이다.

즉, AE 모델의 시작점은 Encoder이다. 차원 축소를 어떻게 하면 더 효율적으로 할 수 있을까?

Unsupervised Learning으로 해당 모델을 학습하는 것보다 Supervised Learning으로 학습하는 것이 더 효율적이므로 Auto Encoder 모델에서는 전체 모델을 Supervised 형태로 만들며 효율적인 차원 축소한다!

모델을 Supervised 형태로 만든 다는 것은 Encoder 뒷단에 Decoder를 붙여 Reconstruction 값과 Input X 값을 Loss 평가하는 형태로 만들면서 위 그림과 같은 형태가 나오게되었다.

다시 말해 Input 데이터가 모델을 Supervised하는 값이되고 그 데이터를 잘 복원 하는 Latent Variable을 학습하는 방향으로 효율적인 차원 축소가 이뤄진다.

### Variational AutoEncoder

---

![VAE](https://namu-tree-kim.github.io/assets/images/vae.jpg "VAE"){: .align-center}

그렇다면 VAE는 어떨까? VAE의 목표는 데이터 생성이다.

즉, VAE 모델의 시작점은 Decoder이다. Decoder에서 Latent Variable Z를 기반으로 데이터 생성을 하기 위해 Z latent variable을 생성하는 Encoder를 앞에 붙이면서 위 그림과 같은 형태가 나오게 되었다.

Latent Variable Z는 원하는 데이터를 뽑기 위한 리모컨 같은 역할을 한다. 다시 말해, Z를 control하여 원하는 데이터를 뽑을 수 있다.

자 그럼 좀 더 디테일하게 하나씩 알아가보자

### AutoEncoder를 그대로 생성에 쓰면?

---

위 설명만 봤을 땐 그러면 AutoEncoder를 그대로 데이터 생성에 사용하면 되는거 아냐?란 생각을 해볼 수 있다.

하지만 AutoEncoder 모델을 데이터 생성에 사용하면 발생하는 문제가 존재한다!

![ae_latent_space](https://namu-tree-kim.github.io/assets/images/ae_latent_space.jpg "ae_latent_space"){: .align-center}

1. Latent Space의 Distribution이 정의되어 있지 않다.

   Latent Variable 데이터가 기준없이 흩어져 있다.

   (위 그래프를 보면 (0,0)원점이 기준 대칭이 아니고 데이터가 기준 없이 퍼져있다)

2. 같은 확률로 데이터를 일정하게 샘플링 할 수 없다.

   어떤 데이터는 밀집해있고 어떤 데이터는 넓게 퍼져 있다.

3. 데이터 품질이 떨어진다.
   데이터 Z 사이 거리가 멀고 빈 공간이 많다.

4. 연속적으로 Latent Variable가 만들어지지 않는다.

   (-2, 2)와 (-2.1, 2.1)이 비슷하단 것을 ensure할 수 있는 mechanism이 존재하지 않는다.

5. Interpret, Manipulate Latent Feature하기 어렵다.

위 문제를 해결하기 위해 Encoder와 Loss Function을 수정한 것이 VAE이다!

### VAE는 뭐가 달라?

---

![va_vae_encode](https://namu-tree-kim.github.io/assets/images/va_vae_encode.jpg "va_vae_encode"){: .align-center}

**1. Encoding 차이점**

VAE는 Latent Space 포인트 주변 Multivariate Normal Distribution에 매핑 vs AE는 Latent Space 한 포인트에 직접 매핑

즉, VAE 인코더는 입력 데이터를 Latent Space의 다변수 정규 분포를 정의하는 2개의 벡터 평균, 분산 값으로 인코딩된다. (분산은 항상 양수이므로 분산에 로그 취한 후 매핑한다)

이전 AE는 latent space를 연속적으로 만들 필요가 없었지만 이렇게되면 평균 주변 지역에서 랜덤한 포인트를 샘플링 하므로 Decoder는 Reconstruction Error를 줄이기 위해 자연스럽게 (-2, 2)와 (-2.1, 2.1)이 비슷한 데이터를 reconstruct하도록 만들어진다.

**2. Loss Function 차이점**


$D_{KL} [N(\mu,\sigma) \|\| N(0,1)]$

KL Term이 Reconstruction에러에 더해진다.

의미하고 모델상 생기는 강점은

2.1 Latent Space에서 포인트를 선택할 때 사용되는 분포가 표준 정규 분포를 가지게 된다.

2.2 인코딩된 분포들이 표준 정규분포에 가깝게 강제되었으므로 포인트 군집(Cluster) 사이에 큰 간격이 생길 가능성이 낮아진다. Latent Space를 대칭적이고 효율적으로 사용가능해진다(대칭적이고 space낭비 없이 Points가 찍힐 것)

![vae_latent_space](https://namu-tree-kim.github.io/assets/images/vae_latent_space.jpg "vae_latent_space"){: .align-center}

왼쪽은 VAE의 Latent Space(Class 별로 색을 다르게 표현)
오른쪽은 왼쪽 값을 (0,1)사이의 값으로 변환한 그래프

### 알아야할 추가 테크닉들

---

**1. Variational Inference**

   학습시에 p(Z|X)를 알 수 없으므로 우리가 다루기 쉬운 q()분포를 가정하여 q(Z|X)를 p(Z|X)에 근사한다.
   (수식 나중에 추가 예정)

**2. Reparameterization Trick**

   학습시에 Monte Carlo Method를 적용하게 되고 Z 샘플링시에 평균, 분산 값이 랜덤이면 back propagation 적용이 불가능하다.

   그래서 $Z = \mu + \sigma^2 \epsilon, \epsilon ~ N(0,1)$ 형태로 표현하여 $\mu와 \sigma$ 에 대해 미분 가능한 형태로 만든 것이 Reparameterization Trick이다.
