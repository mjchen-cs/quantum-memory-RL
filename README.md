# Machines can learn to use quantum memory from a single bit of feedback

Code for the manuscript

> *Machines can learn to use quantum memory from a single bit of feedback*
> (anonymous submission, under review).

All hardware experiments ran on the Origin Quantum **WK_C180** superconducting processor through its cloud service in November 2026. Every number and figure in the paper can be recomputed offline from the raw data, without access to the hardware.

**Data.** The raw hardware counts, training logs, processed results and the Source Data files of the figures are deposited separately on Figshare: https://figshare.com/s/0d5117d472fa966f0045. Download `quantum-memory-rl_data.zip` and extract it into the root of this repository, which creates `hw/runs/` and `source_data/`.

## Repository layout

```
hw/                     code for the hardware experiments, training, analysis and pruning
hw/runs/                data (from the Figshare archive, see "Data" below)
  <STEP>_origin/        source data: raw counts of every circuit run on WK_C180
  train_*.json          training logs (on-chip, reward-only training)
  adapt_*.json, detect_s5*.json, train_nt_*.json   adaptation, detection fine-tuning, no-sign ablation
  train_sim/            simulation-only training runs (development)
  *_summary_origin.json, *_bootstrap_*.json, ... processed results used in the paper
  per_pauli/            per-Pauli results (Figs 3h,i and 4h)
  chip.json, chip_WK_C180_2.json   calibration snapshots of the processors
  prune.json            pruned front-ends (hw_prune.py)
hw/agent_n2.json        front-end trained in simulation at n = 2 ("simulation-trained")
hw/sim/                 simulated training controls (Fig. 7c,d, Supplementary Note 7); results in hw/sim/results/ (from the Figshare archive)
theory/                 numerical checks of the bounds (Supplementary Note 1, Supplementary Fig. 1)
figures/                scripts that draw Figs 1-7 from hw/runs (output in figures/out/) and write the Source Data workbooks (source_data.py)
                        (the photograph of the processor in Fig. 1a, figures/chip_photo.png, is not distributed; fig1.py draws a placeholder without it)
source_data/            Source Data workbooks, one per figure (from the Figshare archive)
```

## Installation

```
pip install -r requirements.txt
```

Python >= 3.10. `pyqpanda3` is only needed to submit new circuits to Origin Quantum hardware; all analyses run offline.

## Data

### Source data: `hw/runs/<STEP>_origin/`

Each hardware step is a folder with

| file | content |
|---|---|
| `manifest.json` | every circuit of the step: `key`, `n` (qubits per copy), `kind` (`quantum` two-copy front-end or `classical` random Pauli bases), hidden Pauli `P` (0=I, 1=X, 2=Y, 3=Z per qubit), `shots`, and `meta` (shared sign `sign`, repetition `rep`, front-end `policy`, random bases `setting_a`/`setting_b`, time stamps of the drift task); also the physical couplers `sites`, the calibration time `chip_fetched` and the front-end angles `theta` / `thetas_extra` |
| `raw.json` | measured counts per circuit key: `{"counts": {outcome_index: count}, "job": job_id}`. Outcome indices are logical, qubit 0 most significant; qubits `0..n-1` are copy A and `n..2n-1` copy B, site `i` is the pair `(i, n+i)` |
| `jobs.json` | cloud job ids, circuits per job, submission and collection time, timing information returned by the cloud |
| `failed_jobs.json` | jobs that failed or were abandoned in the cloud queue; their circuits were resubmitted and appear in `raw.json` once |

**Important for any re-analysis:** each circuit prepares one random product state of the mixture that defines the copies; the shots of a circuit repeat the same state and are *not* independent copies. All analyses use one random shot per circuit and never combine two shots of the same circuit in an episode.

| step | n | content | paper |
|---|---|---|---|
| S1, S2 (`_origin_sm`), D1 | 2 | job timing, two-copy visibility, compiler-mapping test | Methods |
| S4 | 3 | Bell (CRX), simulation-trained front-end, random bases | Fig. 3b,g |
| S3P (+ S3 `_origin_sm`) | 4 | Bell (CRX), simulation-trained, random bases | Fig. 3c,g |
| S5, S5C | 5 | Bell (CRX), simulation-trained; random bases | Fig. 3d,g |
| S6, S6X | 6 | Bell (CRX), simulation-trained | Fig. 3g |
| R4, R5 | 4, 5 | repeat with the first chip-trained front-end (seed 1) | Supp. Table 2 |
| T4, T5 | 4, 5 | chip-trained seeds 20, 22 | Fig. 3g,j |
| S5R, S6R | 5, 6 | repeats on a later day | Fig. 3j |
| S7 | 7 | Bell (CRX), chip-trained seed 20 (as trained) | Supp. Table 2 |
| PR5, PR6, PR7 | 5, 6, 7 | as trained vs pruned (native, 1 CZ per site), Bell CRX vs native | Figs 3, 4 |
| SC3, SC4 | 3, 4 | shortcut front-ends trained without the shared sign, Bell (native) | Fig. 5 |
| DT5, DT6, DT7, DTC5 | 5-7 | detection: signal and null circuits; random bases (DTC5) | Fig. 6h-k |
| DR5 | 5 | drift tracking streamed in time order | Fig. 6d-g |
| X8, X9, X10, X10R | 8-10 | Bell (native) and chip-trained seed 20 pruned; X10R repeats X10 without the faulty coupler (qubits 80, 89) | Fig. 7a,b |
| CH5, CH6 | 5, 6 | channel task: Bell pair input and Bell measurement (reference) | Fig. 7h,i |
| CH5A, CH6A | 5, 6 | channel task: input learned on the chip (seed 70) with the reused measurement, same Paulis and qubits as CH5/CH6 | Fig. 7h,i |

### Training data: `hw/runs/*.json`

| file | content | paper |
|---|---|---|
| `train_s1.json` | first protocol (v1) | Fig. 2a |
| `train_s10/11/12.json` (+ `_v1_snapshot`) | v1, then decoder restarts (v2) from iteration 151 | Fig. 2a |
| `train_s20/21/22.json` | v3: decoder restarts + annealing, unscreened seeds | Fig. 2b |
| `train_s30/31/32.json` | v4: + front-end restart | Fig. 2c |
| `train_nt_s20/22.json` | ablation without the shared random sign (shortcut) | Fig. 5a-d |
| `adapt_s20/22.json` | online adaptation after an injected R_Z(pi/2) error | Fig. 6a-c |
| `detect_s50/51.json` | fine-tuning for detection | Fig. 6h-k |
| `train_s2x_c2.json` | runs on WK_C180-2, abandoned after a few iterations (not used) | - |
| `train_sc_s60/61.json` | single-copy agent (no two-qubit gate) | Fig. 7e |
| `train_ch_s70/71.json` | channel task: input circuit learned on the chip, measurement of seed 20 reused | Fig. 7g |
| `prune_channel.json` | channel agents after pruning (hw_prune_channel.py) and the analytic reference | Fig. 7h,i |
| `chip_ext.json`, `chip_ext2.json` | calibration snapshots of the later steps (X8-X10, X10R and CH) | - |
| `*_bign_origin.json`, `*_channel_origin.json` | results of hw_bign.py and hw_channel_analyze.py | Fig. 7 |
| `*.opt.pt` | optimizer state (PyTorch) for resuming a run | - |

Each training log contains `args` (all hyperparameters), `hist` (one record per iteration: `reward`, ideal two-copy visibility `vis_ideal`, single-copy visibility `vis_single`, step-size multiplier `lr_mult`, decoder restart `dec_restart`, front-end restart `fe_restart`, processor time `info.qpu_ms`), the final front-end `theta`, the decoder `agent`, and `best` (front-end and decoder with the best ten-iteration mean reward, used for deployment).

## Reproducing the results

All commands run from `hw/` and read `hw/runs/`.

```
# identification: copies for 90 % (prints the values of Fig. 3 and Supp. Table 2)
python hw_analyze.py --step PR5
python bootstrap_ci.py --step PR5 --B 200 --level pauli --only bell,bell_native,onchip20,onchip20_pruned,onchip22,onchip22_pruned
python bootstrap_large.py --step PR7 --B 40 --trials 150 --decoders bayes --policies bell_native

# pruning (Fig. 4, Supp. Table 3)
python hw_prune.py

# shortcut (Fig. 5)
python hw_shortcut.py --step SC3

# drift tracking streamed on the chip (Fig. 6d-g)
python hw_drift_live.py

# detection (Fig. 6h-k)
python hw_detect.py --step DT5 --trials 400 --seq_runs 100

# deployment at n = 8-10 (Fig. 7a,b): exact Bayes over all 4^n - 1 Paulis via a Walsh-Hadamard transform
python hw_bign.py --step X10R

# channel task (Fig. 7h,i)
python hw_prune_channel.py runs/train_ch_s70.json runs/train_ch_s71.json
python hw_channel_analyze.py --step CH5A

# simulated controls (Fig. 7c,d): about 1 h on two cores
python sim/run_controls.py && python sim/summarize.py

# per-Pauli analysis (Figs 3h,i and 4h)
python ../figures/per_pauli.py PR5 bell,bell_native,onchip20,onchip20_pruned,onchip22,onchip22_pruned 100

# figures and Source Data workbooks
cd ../figures && python source_data.py && python fig1.py && python fig2.py && python fig3.py && python fig4.py && python fig5.py && python fig6.py && python fig7.py
```

Bootstrap and simulation steps are random; with the default seeds the numbers match the paper up to Monte Carlo fluctuations.

### Running new experiments on hardware

`hw_run.py` (fixed circuit sets) and `hw_train.py` (closed-loop training) submit circuits to Origin Quantum through `pyqpanda3`. Set your own key in the environment variable `ORIGIN_API_KEY`; no key is stored in this repository. Examples:

```
python hw_run.py --fetch_chip
python hw_run.py --step PR5 --mode dryrun          # local simulation of the same circuits
python hw_train.py --backend sim --iters 300 --out runs/train_sim/test.json
python hw_train.py --backend origin --iters 250 --seed 30 --anneal 0.7 --dec_restart_every 20 --fe_restart_patience 40 --out runs/train_s30.json
python hw_train.py --backend origin --frontend product6 --iters 80 --seed 60 --dec_restart_every 20 --out runs/train_sc_s60.json
python hw_train.py --backend origin --frontend channel_in13 --iters 80 --seed 70 --anneal 0.7 --dec_restart_every 20 --fe_restart_patience 40 --out runs/train_ch_s70.json
```

## Code overview (`hw/`)

| file | purpose |
|---|---|
| `hw_core.py` | state preparation, site front-end V(theta) and its native one-CZ form, circuit simulation, pyqpanda3 translation |
| `fast_site.py` | response tables g_k of V(theta), exact likelihoods, Bell front-end |
| `hw_noise.py` | calibration-based qubit selection and local noise model |
| `hw_run.py` | all hardware steps (circuit sets, submission, resumption, `--abandon` for stuck jobs) |
| `hw_train.py` | reward-only training on the chip (evolution strategies + REINFORCE, decoder/front-end restarts, annealing); `--frontend` selects the two-copy agent, the single-copy agent or the channel task |
| `hw_detect_train.py` | fine-tuning for detection |
| `hw_analyze.py`, `bootstrap_ci.py`, `bootstrap_large.py` | identification analysis and Pauli bootstrap (site-factorized for n = 7) |
| `hw_prune.py` | hardware-aware pruning |
| `hw_shortcut.py` | analysis of the shortcut experiment |
| `hw_detect.py` | detection analysis (mixture likelihood ratio, GLR-CUSUM) |
| `bign.py`, `hw_bign.py` | identification at n = 8-10: Walsh-Hadamard evaluation of the exact likelihoods of all 4^n - 1 Paulis |
| `hw_prune_channel.py`, `hw_channel_analyze.py` | channel task: pruning and analysis |
| `sim/simlab.py`, `sim/run_controls.py`, `sim/summarize.py` | simulated training with other front-end families, single-copy and channel agents |
| `hw_drift_live.py` | drift-tracking analysis; `hw_drift.py` is the earlier replay of recorded data (superseded) |
| `shortcut_scale.py` | noiseless extrapolation of the shortcut front-ends |
| `list_backends.py`, `probe.py`, `probe_pinned.py`, `inspect_mapping.py` | cloud utilities |

## Notes

- The detection analysis was corrected during the study (one random shot per circuit, no reuse of circuits); `hw_detect.py` is the corrected version used in the paper.


## License

MIT (see `LICENSE`). Citation information will be added after review.
