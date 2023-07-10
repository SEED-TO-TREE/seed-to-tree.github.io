---
layout: single
title: "Vaiaional Autoencoder for Collaborative Filtering"
excerpt: "Vaiaional Autoencoder for Collaborative Filtering 논문 정리!"

date: 2023-07-10 16:50:00 +0900
lastmod: 2023-07-10 16:50:00 +0900 # sitemap.xml에서 사용됨

author_profile: false # 왼쪽부분 프로필을 띄울건지

categories:
  - Recommender System

tags:
  - ML
  - VAE

# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---

Variational Autoencoder를 추천 시스템에 적용할 수 있을까?

이번 포스트에서 정리하려는 논문은 제목 그대로 literally 
Variational Autoencoder를 Collaborative Filteirng에 적용한 논문이다 
WWW '18년도에 나온 논문이고 이후 이 논문을 기반으로 VAE for CF모델 Line의 논문이 있는데 차례로 정리할 예정이다.

먼저 이 논문의 핵심 Contributions을 기준으로 두고 논문을 읽어나가면 이해가 쉬우므로 Contributions을 확인해보자.

## Contributions
---
**1. VAE를 CF에 적용한 것**

**2. Multinomial Distribution을 Likelihood에 사용한 것**

  실험적으로 Collaborative Filtering에 Multinomial이 Likelihood로 적절함을 보여주었다.  
  (하지만 이는 데이터가 달라지면 달라질 수 있다.)

**3. 기존 VAE ELBO에 Partial Regularize한 것**

  말은 어려워 보이지만 VAE ELBO의 Regularize Term에 $\beta$ 를 넣어 Prior를 따라야하는 Regularize를 줄이고 추천 성능을 좀 더 높이기 이해 ELBO Term을 수정한 것일 뿐이다. 
  
  이 또한 실험을 통해 더 성능이 좋음을 Empirical하게 보여준다.  
  (이는 데이터가 달라져도 적용 가능한 General한 방법이다.)


## Methods
---

![vae_cf1](https://namu-tree-kim.github.io/assets/images/vae_cf1.PNG "vae_cf1"){: .align-center}

### **모델의 흐름**

$z_{u} \to f_{\theta}(z_{u}) \to x_{u}$


**1. $z_{u} \sim N(0, I_{K})$**

각 유저에 대해 Standard Gaussian Prior에서 K차원의 latent representation $z_{\mu}$를 Sampling 하는 것이 모델의 첫 단계이다. 

그런데 단순히 수식만 보면 Prior에서 바로 뽑는 것처럼 생각되지만 VAE의 경우 Encoder를 거쳐서 $\mu$ 와  $\sigma$ 값을 뽑은 후 해당 값을 기반으로 $Z_u$ 을 Sampling한다. 

여기서 Standard Gaussian Prior에서 샘플링한다는 것은 추후 Loss Function을 보면 알 수 있는데 $q(z\|x)$가 $p(z)$ prior를 따라가도록하는 KL Divergence term이 있다.  

그래서 Standard Gaussian Prior를 닮아가도록 Optimized된 $q(z\|x)$에서 reparameterization을 적용 후 Sampling하는 것을 $z_{u} \sim N(0, I_{K})$ 같이 표현한 것으로 이해했다.

**2. $\pi(z_u) \propto exp(\{f_{\theta}(z_{u})\})$**

$f_{\theta}$는 decoder network(Multilayer Perceptron)를 의미한다. exp가 붙는 이유는 Varaince 값은 항상 양수이다.

때문에 log variance로 표현하여 encoder의 neural network가 $(- \inf, \inf)$ 범위로 Mapping 가능하게 하도록한다. 

그래서 exp로 원래 variance 값으로 돌려주는 것이다.

**3. $x_u \sim Mult(N_u, \pi(z_u))$**

$N_u$는 User u가 전체 item을 클릭한 수의 총합 즉 User u의 전체 클릭 수를 의미한다. 

$\pi(z_u)$의 Output은 Softmax를 거쳐서 User u가 모든 Item i 를 클릭할 확률을 의미한다.


### **Multinomial Likelihood Distribution**
![vae_cf_multinomial](https://namu-tree-kim.github.io/assets/images/vae_cf_multinomial.PNG "vae_cf_multinomial"){: .align-center}

위 처럼 multinomial likelihood를 사용한다. multinomial likelihood를 사용하면 좋은 이유는 추천 Ranking Loss와 맥락이 잘 맞다. 왜냐하면 $\pi(z_u)$ 의 총 합은 항상 1이 되어야하고 모든 items는 제한된 확률 분포 공간안에서 높은 확률을 차지하기 위해 경쟁해야한다.(Must compete for this limited budget) 즉 더 많이 클릭한 items에 더 높은 확률을 부여할 것이다.  이것은 top-N ranking loss와 같은 평가 방식이기 때문에 multinomial likelihood를 사용하는 것이 성능이 더 좋다.

만약 $f_{\theta}$를 Linear 한 함수 그리고 Likelihood를 Gaussian Distribution으로 설정하면 모델은 Collaborative Filtering에 가장 많이 사용된 방법인 Matrix Factorization과 같은 방법이된다.



### **Partial Regularizing EBLO**

![vae_cf_elbo1](https://namu-tree-kim.github.io/assets/images/vae_cf_elbo1.PNG "vae_cf_elbo1"){: .align-center}
![vae_cf_elbo2](https://namu-tree-kim.github.io/assets/images/vae_cf_elbo2.PNG "vae_cf_elbo2"){: .align-center}

기존 ELBO식과 달리 KL Term에 $\beta$가 붙는다. 이는 $q(z\|x)$가 prior $p(z)$를 따르게하는 regularizer를 완화하여 추천 성능을 높이는데 초점을 맞춘다.
Collaborative Filtering의 핵심은 추천 정확도를 높이는 것이지 기존 Historical Data를 비슷하게 잘 생성하는 것(ancestral sampling)이 아니기 때문이다.

위 $\beta$를 설정할 때 grid search를 하기엔 computationally 비효율적이기 때문에 Annealing 방법을 사용한다. 
여기서 Annealing이란 == We linearly anneal the KL term slowly over a large number of gradient updates to θ,ϕ and **record the best β when its performance reaches the peak**.