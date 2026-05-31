# Algorithm Development Notes

이 문서는 공동교육과정 배정 AI의 발전 과정을 기록한다. 핵심 목표는 수요가 낮거나 과목 접근성이 낮은 학교의 학생들이 주변 학교와 함께 수업을 들을 수 있도록 배정하고, 그 결과 최저 SAI와 평균 SAI를 함께 끌어올리는 것이다.

## 1. 알고리즘 변천사

### 1.1 Notebook 분석에서 파이프라인으로 전환

초기 분석은 Colab notebook에서 진행되었다. NEIS 수집, 공동교육과정 정리, 과목 표준화, SAI 계산, 배정 추천이 한 파일 안에 섞여 있어 실행이 오래 걸리고 중간 결과 검증이 어려웠다.

이를 `coursemap_pipeline/` 코드베이스로 분리했다.

- `scripts/`: 단계별 실행 파일
- `src/coursemap/`: 여러 단계에서 실제로 공유되는 로직
- `Makefile`: 변경된 입력만 다시 실행
- `build/`: 중간/최종 산출물
- `config/`: 과목 override, ignore, blacklist

원칙은 명확하다. 조인 실패, 필수 feature 누락, 표준화 실패는 조용히 무시하지 않고 명시적으로 실패하거나 blacklist/review 파일에 남긴다.

### 1.2 공동교육과정 배정 문제 정의

초기에는 공동교육과정을 별도 bonus처럼 보려는 흐름이 있었다. 이후 문제 정의를 바꿨다.

보통 수업과 공동 수업을 구분해서 점수를 주는 것이 아니라, 학생이 접근 가능한 전체 과목 공급을 하나의 offering set으로 합친 뒤 그 결과를 평가한다.

현재 후보 action은 다음 형태다.

```text
(hub school, subject, domain)
```

한 action은 특정 거점 학교가 특정 과목을 열었을 때 반경 내 학교들이 그 과목에 접근 가능해지는 것을 의미한다.

### 1.3 Greedy Baseline

첫 배정 알고리즘은 greedy였다.

- 후보: `(hub school, domain)`
- 점수: 약한 학교의 부족한 계열을 많이 덮는 후보를 우선 선택
- 장점: 빠르고 baseline으로 해석이 쉬움
- 문제: 평균은 올라가도 최저 SAI 학교가 그대로 남을 수 있음

이후 후보를 `(hub, subject, domain)`으로 세분화했다. 계열만 배정하면 실제로 어떤 과목을 추가했는지 SAI가 설명되지 않기 때문이다.

### 1.4 RL Baseline: Step 11

`scripts/11_train_rl_assignments.py`는 PyTorch policy-gradient baseline이다.

- action: 후보 공동수업 하나 선택
- episode: 예산만큼 순차 선택
- reward: 최종 선택 묶음의 incremental SAI 기반 fairness reward
- 비교 대상: greedy baseline

Step 11은 단순한 RL baseline으로 유지한다. Actor-Critic 구조나 state-value head는 Step 12에 둔다.

현재 Step 11 reward는 `IncrementalAssignmentSimulator.score_selected()`를 사용한다.

### 1.5 Actor-Critic: Step 12

`scripts/12_train_actor_critic_assignments.py`와 `src/coursemap/actor_critic_assignments.py`는 Step 12 실험이다.

Step 12는 Step 11보다 더 강한 정책 구조를 가진다.

- actor: 현재 선택 가능한 후보의 logit 출력
- critic: 현재 state의 value 추정
- step reward: 선택 후 reward 증가량
- advantage: return - value

state feature에는 다음 정보가 포함된다.

- 예산 진행률
- 커버된 부족 pair 비율
- 현재 reward score
- 선택된 후보들의 평균 거리
- hub 다양성
- domain 사용 분포

중요한 정리: Step 12에는 한때 fast proxy reward, 후보 pruning, feature cache를 넣었지만, 현재 요구사항에 맞게 제거했다. 현재 Step 12도 reward는 incremental SAI만 사용한다.

### 1.6 현재 알고리즘 구조

현재 알고리즘 계층은 다음과 같다.

```text
Greedy baseline
-> Step 11 policy-gradient RL
-> Step 12 Actor-Critic
```

세 알고리즘은 같은 후보 공간 `(hub, subject, domain)`과 같은 incremental SAI objective를 사용한다. 차이는 후보를 고르는 정책이다.

## 2. 목적함수 변천사

### 2.1 초기 목적함수: 넓은 coverage

초기 greedy 목적함수는 많은 약한 학교-계열 pair를 덮는 후보를 선호했다.

```text
score ~= covered shortage pairs + distance weight - duplicate penalty
```

이 목적함수는 빠르고 직관적이었지만 문제가 있었다.

- 학교별 최저 SAI를 직접 보지 않음
- 평균 개선에 유리한 후보가 선택될 수 있음
- 최하위 학교가 방치될 수 있음
- 계열 단위라 실제 과목 다양성 개선을 설명하기 어려움

### 2.2 SAI 재설계

SAI는 데이터베이스 컬럼에 얽매인 고정 계산식이 아니라, Python에서 직접 계산하는 최적화 objective로 재설계했다.

현재 SAI는 학교별 subject set과 6개 계열 count에서 계산된다.

```text
SAI =
  0.55 * 계열폭점수
+ 0.35 * 계열균형성
+ 0.10 * 과목다양성
```

중요한 변경은 과목다양성 scaling이다. 기존 min/max scaling은 전체 학교 분포에 의존해서 한 학교의 배정을 평가하려면 전체 학교를 다시 계산해야 했다. 현재는 고정 target subject count를 사용한다.

```text
과목다양성 = min(과목수 / TARGET_SUBJECT_COUNT, 1) * 100
```

이 덕분에 학교 하나의 과목이 추가될 때 해당 학교 SAI만 다시 계산할 수 있다.

### 2.3 Fairness-oriented reward

목표는 평균 SAI만 올리는 것이 아니라 낮은 학교를 확실히 끌어올리는 것이다.

현재 reward는 다음 항목을 결합한다.

```text
 weak minimum SAI improvement
+ weak mean SAI improvement
+ weak lower-quartile SAI improvement
+ weak bottom-3 SAI improvement
+ weak minimum delta
+ weak mean delta
+ all-school mean delta
+ weak-school improvement ratio
- distance penalty
```

최저값과 평균을 동시에 넣은 이유는 명확하다.

- 최저값만 보면 일부 학교만 몰아줄 수 있음
- 평균만 보면 최하위 학교가 방치될 수 있음
- q25/bottom-3는 하위권 전체가 같이 올라가는지 보기 위함

### 2.4 Incremental SAI objective

최적화 알고리즘이 빠르게 objective를 호출할 수 있도록 incremental state를 만들었다.

관련 코드:

- `src/coursemap/sai.py`
  - `IncrementalSaiState`
  - `score_school()`
  - `sai_from_counts()`
- `src/coursemap/assignments.py`
  - `IncrementalAssignmentSimulator`
  - `score_selected()`
  - `simulate()`

학습 reward는 이제 `simulate()`를 타지 않는다. `simulate()`는 최종 CSV/그래프 출력용이다.

학습 중 reward는 다음 경로를 사용한다.

```text
selected assignments
-> simulator.score_selected(selected)
-> changed schools only update
-> dict 기반 after/delta 반환
-> reward 계산
```

즉, reward에서 pandas merge/filter/sort를 하지 않는다.

## 3. 성능 진단

incremental SAI 적용 후 reward 자체는 빨라졌다.

측정 결과:

```text
score_selected() 100회: 약 0.08초
sample_episode() 1회: 약 1.74초
```

따라서 현재 큰 병목은 SAI 계산이 아니라 후보 선택 루프다.

남은 주요 병목:

- `available_candidates()`가 매 step마다 모든 후보를 순회
- `materialize_selection()`이 반복 호출됨
- `feature_matrix()`가 매 step마다 후보 feature를 다시 numpy/torch tensor로 생성
- 후보 수가 많을 때 한 episode에서 이 과정이 예산 수만큼 반복됨

즉, 다음 성능 개선은 SAI가 아니라 RL 후보 처리 구조에 집중해야 한다.

## 4. 현재 평가 산출물

추천 알고리즘은 greedy와 비교된다.

출력 지표:

- 전체 학교 평균/표준편차/최소/중앙 delta
- 약한 학교 평균/표준편차/최소/q25/bottom-3
- improved school count
- RL 또는 Actor-Critic minus greedy school-level delta
- SAI before/after dot plot

dot plot에는 약한 학교의 after mean과 after min 수평선을 표시한다. 목표는 이 수평선, 특히 최저선이 올라가는 것이다.

## 5. 다음 개선 방향

우선순위는 다음과 같다.

1. `feature_matrix()` 후보 feature precompute/cache
2. `available_candidates()`의 covered-pair 계산을 episode state로 유지
3. 후보 수를 줄이는 별도 실험은 Step 12 성능 실험으로 분리
4. hub capacity, 학교별 참여 상한, 과목 중복 제한 추가
5. fairness-greedy baseline 추가
6. public facility accessibility를 후보 feature에 반영

중요한 원칙:

- SAI reward는 proxy가 아니라 incremental SAI를 사용한다.
- Step 11은 단순 RL baseline으로 유지한다.
- Step 12는 Actor-Critic 실험장으로 유지한다.
- 최적화 성능 개선과 objective 변경은 구분해서 기록한다.
