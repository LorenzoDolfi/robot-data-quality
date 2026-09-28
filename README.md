# When Does Data Quality Matter in Robot Imitation Learning?

**Four controlled experiments on curation, defect detection, contamination, and what public datasets actually contain**

Lorenzo Dolfi · September 2026

---

## Summary

Robot learning is widely described as data-constrained, and a growing industry sells curated training data on the premise that removing low-quality demonstrations improves learned policies. I tested that premise directly.

Across four experiments on the PushT benchmark using Diffusion Policy in LeRobot, I found:

1. **Heuristic quality filtering underperformed random selection.** A scorer built from completion, efficiency, idle time, and motion smoothness selected subsets that trained *worse* policies than size-matched random subsets, because its efficiency terms silently removed the longest and hardest demonstrations.
2. **Defect detection itself is solvable.** A physics-based consistency check — comparing commanded actions against the robot's observed motion — identified 62 of 62 synthetically corrupted episodes across four corruption types, with no false positives, and flagged only 1.5% of genuinely clean episodes.
3. **Removing detected defects did not improve policies at moderate contamination.** At a 30% corruption rate, removing exactly the corrupted episodes performed no better than removing the same number at random (30.0 ± 0.8 vs. 31.3 ± 1.1 percent success).
4. **Contamination does hurt, but only past a threshold.** Holding dataset size fixed and varying only the corrupted share produced a clean monotonic curve: 20.9% success at 0% contamination, 17.2% at 25%, 10.1% at 50%, and 2.1% at 100%.
5. **Public datasets are close to clean by this measure.** Scanning three public LeRobot datasets found a 0% defect rate in two and 10% in the third, consisting entirely of small timing offsets consistent with normal human teleoperation.

Taken together: defect removal is technically feasible and measurably valuable, but only in a contamination regime that published robotics datasets do not appear to occupy.

A separate methodological finding may be the most broadly useful one. An apparent 9-point performance advantage, significant-looking across three random seeds, **disappeared entirely when evaluation was repeated with 500 episodes instead of 100**. The effect was an artifact of evaluation noise, not of the intervention.

---

## 1. Setup

| Component | Choice |
|---|---|
| Benchmark | PushT (gym-pusht), 206 human demonstrations |
| Policy | Diffusion Policy (ResNet-18 vision encoder, DDPM, horizon 16) |
| Framework | LeRobot |
| Training | 30,000 steps, batch size 64, identical across all conditions |
| Evaluation | Simulated rollouts, fixed evaluation seed, 100 or 500 episodes as noted |
| Hardware | Single RTX 5080 laptop GPU |

Every comparison holds training steps and evaluation protocol constant. Where subsets differ in size, that difference is stated explicitly, because with a fixed step budget a smaller subset receives more passes over each episode.

---

## 2. Experiment 1: Heuristic quality filtering

**Design.** Each of the 206 episodes received a quality score combining maximum achieved reward, final reward, episode length, idle fraction, and second-difference action magnitude (a smoothness proxy). Three training sets were compared: the 124 highest-scoring episodes, 124 episodes drawn at random, and all 206.

**Result** (30,000 steps, 100-episode evaluation):

| Training set | Episodes | Success | Avg. max reward |
|---|---|---|---|
| Curated (top 60%) | 124 | 24.0% | 0.709 |
| Random (60%) | 124 | 27.0% | 0.794 |
| Full | 206 | 41.0% | 0.894 |

**Interpretation.** Curation did not beat random selection. Subset composition analysis explains why: the curated set contained 13,919 frames against random's 15,262, with a mean episode length of 112 versus 123 and a markedly narrower length distribution (standard deviation 27.1 versus 35.5). Coverage of the action and start-state spaces was comparable across subsets, so the scorer did not remove whole regions of the task; it removed the *long* instances within them.

Episode length in PushT correlates with task difficulty. Hard initial configurations require more pushing and repositioning. By rewarding efficiency, the scorer systematically discarded the hardest demonstrations — precisely the ones the policy fails on at test time.

**Lesson.** Any quality criterion correlated with task difficulty will preferentially remove the most informative data. On already-clean data, quantity and diversity dominate polish.

---

## 3. Experiment 2: Defect detection and removal

### 3.1 A physics-based detector

Rather than scoring how a demonstration *looks*, this detector tests whether the recorded commands are consistent with the recorded motion. In PushT, the action is a target position for the agent, so the agent's velocity should correlate with the displacement between its position and the commanded target. Four statistics are computed per episode:

- **Consistency**: correlation between agent velocity and (action − position) at the dataset's characteristic lag.
- **Lag**: the time offset at which that correlation peaks, which exposes clock drift between streams.
- **Action jitter**: high-frequency action energy relative to the robot's actual displacement.
- **Freeze fraction**: the longest stretch where the action is constant while the robot continues to move.

Episodes are flagged using robust z-scores (median and MAD) against the dataset's own distribution, so the method adapts to each dataset's baseline rather than relying on absolute thresholds. A later revision requires an episode to be both a relative outlier and absolutely poorly tracked, which prevents uniformly noisy datasets from being flagged wholesale.

Notably, none of these defects are addressable by signal filtering. A low-pass filter cannot recover a mislabeled outcome, a duplicated episode, a time-shifted stream, or a swapped file, and smoothing action labels degrades exactly the sharp contact events that imitation learning needs.

### 3.2 Detection performance

A corrupted copy of PushT was produced by modifying the action labels of a random 30% of episodes (62 of 206), leaving frame counts, video, and metadata intact. Four corruption types were applied in equal proportion: additive noise at 8% of action range, frozen actions across 40% of an episode, a six-step temporal shift, and actions swapped in from a different episode.

| Corruption type | Injected | Detected |
|---|---|---|
| Noise | 16 | 16 |
| Freeze | 16 | 16 |
| Shift | 15 | 15 |
| Swap | 15 | 15 |
| **Total** | **62** | **62** |

Precision 1.00, recall 1.00. On the uncorrupted dataset, the detector flagged 3 of 206 episodes (1.5%), all with consistency above 0.98 and no time offset — false alarms produced by the dataset's unusual homogeneity rather than by real defects.

### 3.3 Does removal help?

Three training sets were compared at 30,000 steps across three seeds: all 206 episodes (62 corrupted), the 144 episodes surviving detection (0 corrupted), and 144 episodes with the same number removed at random (approximately 43 corrupted remaining).

| Training set | 100-episode evaluation | 500-episode evaluation |
|---|---|---|
| All 206 (30% corrupted) | 22.7 ± 3.3 | 27.7 ± 0.6 |
| Random removal (144) | 25.7 ± 4.2 | 31.3 ± 1.1 |
| Detector removal (144) | **31.0 ± 5.7** | 30.0 ± 0.8 |

**The methodological finding.** Under 100-episode evaluation, detector removal led random removal by 5.3 points on average and won in all three seeds — an apparently consistent effect. Re-evaluating the *same checkpoints* with 500 episodes eliminated it. Per-condition standard deviations fell from roughly ±4 to ±1, and the ordering reversed.

The effect being measured was smaller than the noise floor of the measurement. Consistency across seeds did not protect against this, because the dominant variance came from the evaluation itself rather than from training. Robot learning results reported on 50–100 rollouts should be treated with corresponding caution.

---

## 4. Experiment 3: Dose-response

Experiment 2 compared 0% contamination against 30%. To locate the threshold where contamination begins to matter, dataset size was fixed at 100 episodes and only the corrupted share was varied, across three seeds with 500-episode evaluations.

| Corrupted share | Success | Avg. max reward |
|---|---|---|
| 0% | 20.9 ± 1.9 | 0.733 ± 0.020 |
| 25% | 17.2 ± 1.1 | 0.669 ± 0.018 |
| 50% | 10.1 ± 1.2 | 0.519 ± 0.007 |
| 100% | 2.1 ± 0.1 | 0.363 ± 0.006 |

The relationship is monotonic and tightly estimated, with every seed reproducing the ordering. Success falls about 18% relative at 25% contamination, 52% at 50%, and collapses to near-chance at 100%. Average reward declines from the first increment, indicating that policy quality degrades before the binary success metric registers it.

This also explains Experiment 2's null result: its comparison spanned 0% to 30% contamination, a region where the expected gap is only a few points — small enough to be obscured by differences in subset size and composition.

---

## 5. Experiment 4: How contaminated is public data?

The preceding results make the business-relevant question empirical: what contamination rate do real datasets exhibit? The detector was applied to three public LeRobot datasets, downloading only tabular data (no video). The method first selects the control model that better explains each dataset — actions as position targets or as velocity commands — then flags outliers within it.

| Dataset | Episodes | Model fit | Flagged |
|---|---|---|---|
| lerobot/pusht | 206 | 0.99 | 0.0% |
| lerobot/aloha_sim_insertion_human | 45 | 0.44 | 0.0% (unreliable) |
| lerobot/aloha_static_coffee | 50 | 0.80 | 10.0% |

The aloha_sim result should be disregarded: its characteristic lag saturated the search boundary and its median consistency was 0.45, indicating that neither control model describes that system. The 10% flagged in aloha_static_coffee consisted entirely of three-to-four-step timing offsets, well within the range of normal human teleoperation variability.

Against the dose-response curve, these rates sit in the regime where contamination costs little or nothing.

---

## 6. Conclusions

1. **Quality filtering on clean data is counterproductive** when the quality criterion correlates with task difficulty.
2. **Command–motion consistency is an effective, physically grounded defect detector**, and detects failure modes that signal processing cannot address.
3. **Contamination degrades policies in a dose-dependent manner**, with meaningful damage beginning near 25% and severe damage by 50%.
4. **Well-maintained public datasets do not appear to reach that regime**, at least for defects of this class.
5. **Evaluation budgets in robot learning are frequently too small** to support the claims made from them. A 9-point effect, reproduced across three seeds, proved to be measurement noise.

The practical implication for data curation as a product: removal-for-performance has a narrow addressable regime. The more defensible problems in public robot data are likely *inconsistency across sources* — differing action spaces, reference frames, control rates, and licensing terms — rather than corruption within any single source.

---

## 7. Limitations

- One benchmark, one policy class, one action dimensionality. PushT is two-dimensional and unusually homogeneous; conclusions may not transfer to high-dimensional manipulation.
- Training used 30,000 steps against the reference implementation's 200,000, so absolute success rates are well below published figures. Comparisons within experiments remain valid, as all conditions share the budget.
- In Experiment 1, the full dataset received fewer passes per episode than the subsets under the fixed step budget, which confounds the full-versus-subset comparison. The curated-versus-random comparison is unaffected.
- Corruptions were synthetic and comparatively severe. Real-world defects are likely subtler and more varied.
- The detector observes only the relationship between actions and proprioception. Mislabeled language instructions, incorrect outcome labels, near-duplicate episodes, and camera calibration errors are outside its scope and may be considerably more common.

---

## 8. Reproducing

The pipeline comprises episode scoring, controlled corruption injection, defect detection, subset construction, training orchestration, and multi-seed result aggregation with dispersion reporting. All experiments run on a single consumer GPU; the complete set reported here represents roughly 30 GPU-hours.
