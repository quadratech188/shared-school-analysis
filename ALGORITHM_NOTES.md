# Algorithm Development Notes

## 1. Initial Greedy Assignment

The first assignment algorithm generated joint-course candidates as
`(hub school, domain)` pairs and selected them greedily. The score favored
candidates that covered many weak school-domain shortage pairs within the
distance radius.

This was useful as a baseline, but it had a clear weakness: it could improve
average SAI while leaving the lowest-SAI schools unchanged.

## 2. SAI Redesign

SAI was redesigned around combined course offerings. Regular school courses and
new joint-course assignments are converted into the same internal offering
schema, then scored as school-level subject sets and six domain counts.

Joint courses are therefore not a separate bonus bucket. They affect SAI only
by changing the actual combined subject supply used to compute subject
diversity, domain breadth, and domain balance.

The current SAI uses a fixed target subject count for subject diversity. This
removes the old full-distribution min/max scaling, so optimization can update
one school at a time without changing every other school's scale.

## 3. Fairness-Oriented Objective

The main objective shifted from broad coverage to educational fairness. The
algorithm now treats the weak-school tail as a first-class target.

Important tail metrics:

- weak-school minimum SAI
- weak-school mean SAI
- weak-school lower-quartile SAI
- weak-school bottom-3 average SAI
- number of weak schools improved

This prevents assignments that raise the average while ignoring the lowest
schools.

## 4. Reinforcement Learning Policy

The assignment problem is modeled as sequential decision-making.

- State: current candidate set, already covered school-domain pairs, weak-school
  SAI profile, distance coverage, and duplicate assignment pressure.
- Action: choose one joint-course assignment candidate, currently
  `(hub school, subject, domain)`.
- Episode: select up to the assignment budget.
- Reward: compute post-assignment SAI and score the result.

Step 11 keeps the simpler PyTorch policy-gradient network and compares its
result against the greedy baseline.

Step 12 is the larger Actor-Critic experiment. Its policy input combines
candidate features with the current assignment state,
including budget progress, covered shortage ratio, current reward score, average
selected distance, hub diversity, and domain usage. This lets the same candidate
be evaluated differently early and late in an episode.

Rewards are assigned at each step from the incremental improvement in the
current assignment score. The critic learns the expected return for the current
state, reducing the variance of policy-gradient updates.

For runtime, Step 11 and Step 12 use the incremental SAI simulator. It clones a
baseline `IncrementalSaiState`, applies proposed joint offerings, and computes
SAI from in-memory subject sets and domain counts. The reward is the real
incremental SAI objective, not a separate proxy.

## 5. Current Reward Design

The reward is designed to raise both the bottom line and the average:

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

The minimum and mean terms are both intentional. A good assignment should push
up the lowest visible horizontal line while also raising the weak-school
average.

## 6. Evaluation Outputs

The algorithm is evaluated against the greedy baseline with:

- all-school mean, standard deviation, minimum, and median delta
- weak-school mean, standard deviation, minimum, lower quartile, and bottom-3
  score
- number of improved schools
- RL-minus-greedy school-level delta
- before/after dot plot

The RL plot includes horizontal lines for weak-school after-mean and after-min
SAI so that fairness gains are visually explicit.

## 7. Next Improvements

Planned algorithm improvements:

- add hub capacity and school participation constraints
- add a fairness-greedy baseline
- include geocoded public hub accessibility in candidate features
- compare multiple reward weight settings with the same evaluation table
