# Statistical Learning

## What is Statistical Learning?
* 어떠한 독립 변수 (X1, X2..Xp)와 종속 변수의 관계를 다음 식으로 표현한다.
  > `Y = f(X) + ε`    
  > *  `Y`는 종속변수에 해당
  > * `X`는 독립변수에 해당
  > * `f(X)`는 종속변수와 독립변수 사이의 관계
  > * `ε`는 랜덤 에러로, X와 독립적이며 평균은 0이다.
* `Statistical Learning`은 이 `F`를 찾는 **일련의 통계학적 과정**을 뜻한다.

## Why Estimate F?
* 이를 `Statistical Learning` 으로 알아야 하는 이유는 크게 `예측`과 `유추`로 나뉜다.

### Prediction (예측)
* `Y = f(X) + ε`가 있다고 할 때, 저 안의 변수 간 관계를 알아내는 건 쉽지 않다.
* 그렇기 때문에 `^Y = f^(X) + ε` 형태로 어림짐작을 한 뒤 데아터를 삽입 시 나오는 결과를 예측
* `^f(X)`의 정확도는 `Irreducible Error`와 `Reducible Error`로 나뉘며,
  * `Reducible Error`의 경우에는 올바른 테크닉을 통해 줄일 수 있지만,
  * `Irreducible Error`의 경우에는 예측때부터 생긴 에러이므로 줄일 수 없다.
  * 이러한 형태의 에러에는 미처 측정되지 않은 **변수**나 **분산**에서 비롯되는 경우가 많다.
> 실제로 어떤 예측값과 실제값의 기댓값을 측정할 때,    
> >E(Y-^Y)^2    
> = E[(f(X) + ε - ^f(X))]^2    
> = [f(X)-^f(X)]^2 (*Reducible*) + Var(ε) (*irreducible*)

### Inference
* 우리는 `독립변수`가 변화함에 따라 `종속변수`가 어떻게 변하는지를 알고 싶다.
* 이 시점에서 `^f(x)`는 그 안의 원리를 정확히 알아야 한다.
* 다음의 질문을 통해서 이를 알아보아야 한다.
> 1. 어떤 predictors가 response가 연관이 있는가?
> 2. response와 each predictor 사이에 어떠한 연관이 있는가?
> 3. Y와 predictor 사이의 관계가 linear하게 요약될 수 있는가? 아니면 더 복잡한가?
* 모델링은 어느 하나만 시행하기 위해 쓰는 것은 아니다; 둘 다 활용할 수 있다.

## How Do We Estimate f?
* 이 통계학적 배움의 주된 몬적은 훈련 데이터를 통해 훈련함으로써 알려지지 않은 f를 찾기 위함이다.
* 이를 알아내는 모든 통계적인 방법은 크게 `모수적 추정 (parametic)`과 `비모수적 추정 (non-parametic)`이 있다.

### 모수적 추정 (Parametic)
* 먼저 어떤 모델에 대해 가정을 하고, 그다음에 이에 맞게 데이터를 학습시킨다.
* 가장 주된 방식은 `Least Squares` 방식이다.
* 이러한 추정의 단점은 앞서 가정한 모델이 정확하지 않을 수 있다는 점이다.
* 만약 잘못된 추정을 했을 시에는 `Overfitting` 같은 통계적 문제가 발생할 수 있다.

### 비모수적 추정 (Non-Parametic)
* 섣불리 모델을 가정하지 않고 **더 폭넓은 가정**을 통해서 접근하는 방식이다.
* 단, 모델이 **사전적으로 가정되지 않으므로** 엄청나게 **많은 샘플**이 필요하다는 단점이 있다.
* 대표적인 사례는 `thin-plate spline` 형식으로, 과거 모델을 가정하지 않고 관찰된 데이터가 맞춰 진행
* 이 `spline`을 맞추기 위해서 `Data Analyst`는 여기에 맞는 `smoothiness`를 사전에 설정해야 한다.

## 예측 정확도와 모델 설명도 간의 관계
* 많은 통계적 방법 중에서, 그 **유동성과 정확도** 간에는 어느 정도의 희생이 있다.
* 통계적 목적 (설명성 or 정확성)에 따라 그 수치 간의 타협을 보아야 한다.    
  ex) *Subset Selection Lasso, Least Squares, GAMT, Bagging/Boosting, SVM*...   
* 하지만 대부분의 방법은 다양한 상황 속에서 **더 정확하면서도 덜 유동적인 방법**을 찾을 것이다.

## Supervised Versus Unsupervised
* `Supervised` 모델의 경우에는 어떠한 변수가 종속변수에 영향을 미치는지를 연구한다.
* `Unsupervised` 모델의 경우에는 **종속변수 값이 없다**. 대신 **변수 간의 관계**에 집중한다.
  * ex) Cluster Analysis: 변수 간의 집단을 구성하여 파악한다,
* 단, 이 두 개를 나누는 기준도 **명확하지는 않다**. (**한 문제에 두 개의 경우**가 섞여있을 수 있다)

## Regression Versus Classification
* 변수에는 `정량적 변수`와 `정성적 변수`가 존재한다.
* 만약 `정량적 변수` 사이의 관계를 주로 본다면 `regression (회귀)`, `정성적`이면 `classification`이라고 한다.
* 하지만 회귀와 분류 사이의 기준또한 다른 것과 마찬가지로 명확하지는 않다. (`KNN`이나 `Boosting`)

## Assessing Model Accuracy (모델 정확도 측정)
* 어떤 모델도 다른 모델보다 항상 우수하거나 열등한 경우는 없다.
* 즉, 조사하고자 하는 데이터셋에 **잘 맞는 모델**이 무엇인지를 파악하는 방법이 필요하다.

### Measuring The Quality of Fit
* 모델에 따라 예측한 결과가 실제 데이터셋에 맞는 정도를 측정한다.
* 가장 많이 쓰이는 value는 `MSE (Mean-Squared Error)`이다.
> *MSE* = 1/n * 𝛴(Yi - ^f(Xi))^2    
> **MSE**는 **실제값과 예측값의 차이의 제곱**을 **샘플 수**로 나는 것이다.
* 실제로 우리가 신경쓰는 건 **우리의 예측**이 **모델의 실제값과 얼마나 차이나냐는 것**에 해당한다.
* 문제는 **훈련 상에서 낮은 MSE**가 **실제 테스트에서의 낮은 MSE를 보장하지는 않는다**는 점이다.
* `flexibility`가 커지면, 그 curve는 데이터에 `좀 더 가깝게 fit`하는 경우가 많다.
* `flexibility`의 레벨에 따라서, 이 데이터에 대한 다양한 fit을 찾을 수 있다.
* 그 `flexibility`의 정도는 `degrees of freedom`을 통해 구한다.
* 만약 유연도가 낮아지기 시작하면,`Training MSE`는 낮아지더라도 `Test MSE`가 낮아지진 않는다.
* 만약 Training MSE가 Test MSE에 비해 지나치게 낮으면 `overfitting`을 의심해야 한다.

### The Bias-Variance Trade-off
> *MSE*     
> = 1/n * 𝛴(Yi - ^f(Xi))^2    
> = Var(^F(x0)) + [Bias(^f(X0))]^2 + Var(ε)

* 어떤 모델을 썼을 때 예상되는 MSE는 위의 식대로 구할 수 있다.
* 즉, **어떤 모델을 쓸 때 `variance`와 `bias`가 낮은 것**을 써야 한다.
* 여기서 `Variance`는 다른 Training Set이 제공될 때 ^F(Xi)가 바뀌는 정도,
* `bias`는 실질적인 문제로 인해 생기는 오류에 해당한다. (*즉, 실제 모델과 예측 모델의 차이*)
* 실제 문제에서는 **True F를 찾을 수 없**으므로 반드시 **bias-variance trade-off**를 기억해야 한다!

## Classification Setting
* 이전의 여러 고려사항은 `regression`에 적용되었으나, `classification`에도 이를 적용할 수 있다.
* 단, 종속변수가 categorical이므로 이에 따룬 약간의 조절은 필요하다.
* 주로 쓰이게 되는 classification의 정확도의 척도는 `Error Rate`로 다음과 같다:
> *Error Rate*    
> = 1/n * 𝛴(n, i=1) * I(Yi = ^Yi)     
> = 즉, Yes/No중 **얼마나 많이 맞췄는가 (같으면 0, 틀리면 1)**에 대한 비율을 구하는 것

### Bayes Classifier
* Predictor Vector `x0`에 대해서 다음과 같은 확률을 구할 수 있다.    
> Pr(Y=j | x = x0)    
> = x0 이라는 predictor Vector가 클래스 j에 들어갈 확률    
> 이는 `조건부 확률`로, **어떤 상황이 주어졌을 때의 확률**에 대해 설명.
* 예를 들어, 클래스 1과 2가 있을 때 특정 Predictor Vector가 있다고 가정한다.    
그 Vector가 `Pr(Y=1 | X=x0) > 0.5`면 이는 `x0`은 **클래스 1**으로 분류한다.
* `Bayes Error Rate`로 `Bayes Classifier`상에서 이론상 최소 오차를 제공한다.
> 어떤 Bayes Classifier에서는 항상 `Pr(Y=j | X = x0)`이 최대인 클래스를 선택,    
> 즉 X = `x0`에서 `BER`은 `1 - max_j(Pr(Y = j | X = x0))` 이다.    
> `OBER`은 `1 - E(max_j(Pr(Y = j | X = x0)))` 이다.

### KNN (K-Nearest Neighbors)
* 실제 분류 문제의 경우 이론적인 모수의 분포 및 조건부 확률을 알기는 쉽지 않다.
* 그러므로 이를 estimate하는데, 대표적인 방법 중 하나는 `KNN`으로
> 1. x0과 최대한 가까운 K개의 포인트의 집합 𝑁₀을 구한 다음
> 2. 이를 갖고 대략적인 조건부 확률을 다음의 식을 통해 어림한다
> > Pr(Y = j | X = X₀) = 1/K * 𝛴(i는 𝑁₀에 포함) * I(Yi = j)
* K 값의 default는 5로, 이 값은 너무 커서도 작아서도 안된다.
