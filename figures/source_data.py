# Source Data workbooks for Figs 1-6 (one .xlsx per figure, one sheet per panel), from hw/runs.
import os, sys, json, itertools, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(_ROOT, "hw", "runs"); PP = os.path.join(R, "per_pauli")
OUT = os.path.join(os.path.dirname(__file__), "source_data"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(_ROOT, "hw"))
from fast_site import tables, bell_theta
J = lambda f: json.load(open(os.path.join(R, f)))
hist = lambda f: J(f + ".json")["hist"]
BOUND = {n: (2 ** n + 1) * (0.9 * np.log(4 ** n - 1) - np.log(2)) for n in range(2, 11)}
PRS = "_bell_bell_native_onchip20_onchip20_pruned_onchip22_onchip22_pruned.json"
F = "Arial"

def book(name, title, panels):
    """panels: list of (sheet, description, DataFrame or None, formulas) ; formulas: list of (col_name, template)"""
    wb = Workbook(); ws = wb.active; ws.title = "README"
    ws["A1"] = f"Source Data for {title}"; ws["A1"].font = Font(name=F, bold=True, size=12)
    ws["A2"] = "Manuscript: Machines can learn to use quantum memory from a single bit of feedback."
    ws["A3"] = "All values are computed from the raw hardware counts and training logs in the accompanying data archive (hw/runs); script: figures/source_data.py."
    ws["A5"] = "Sheet"; ws["B5"] = "Content"
    for c in ("A5", "B5"): ws[c].font = Font(name=F, bold=True)
    r = 6
    for sheet, desc, df, _ in panels:
        ws.cell(r, 1, sheet).font = Font(name=F); ws.cell(r, 2, desc).font = Font(name=F); ws.cell(r, 2).alignment = Alignment(wrap_text=True, vertical="top"); r += 1
    for c in ("A2", "A3"): ws[c].font = Font(name=F)
    ws.column_dimensions["A"].width = 14; ws.column_dimensions["B"].width = 120
    for sheet, desc, df, formulas in panels:
        s = wb.create_sheet(sheet)
        s["A1"] = desc; s["A1"].font = Font(name=F, italic=True); s["A1"].alignment = Alignment(wrap_text=False)
        if df is None:
            s["A3"] = "Schematic panel: no data."; s["A3"].font = Font(name=F); continue
        cols = list(df.columns)
        for j, c in enumerate(cols, 1):
            cell = s.cell(3, j, c); cell.font = Font(name=F, bold=True); cell.fill = PatternFill("solid", fgColor="EBEBE6")
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for i, row in enumerate(df.itertuples(index=False), 4):
            for j, v in enumerate(row, 1):
                if isinstance(v, (float, np.floating)) and not np.isfinite(v): v = None
                elif isinstance(v, np.generic): v = v.item()
                s.cell(i, j, v).font = Font(name=F)
        nrow = len(df)
        for k, (cname, tpl) in enumerate(formulas or []):
            j = len(cols) + 1 + k
            cell = s.cell(3, j, cname); cell.font = Font(name=F, bold=True); cell.fill = PatternFill("solid", fgColor="EBEBE6"); cell.alignment = Alignment(wrap_text=True, vertical="top")
            for i in range(4, 4 + nrow):
                s.cell(i, j, tpl.format(r=i)).font = Font(name=F); s.cell(i, j).number_format = "0.000"
        for j in range(1, len(cols) + len(formulas or []) + 1):
            s.column_dimensions[get_column_letter(j)].width = 16
        s.freeze_panes = "A4"
    wb.save(os.path.join(OUT, name)); print("saved", name)

def curves(rows):
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- Fig 1
chip = J("chip.json"); man7 = J("PR7_origin/manifest.json")
used = {(s["a"], s["b"]) for s in man7["sites"][:7]}
seen = {}
for x in chip["double_qubits"]:
    a_, b_ = sorted(x["qubits"]); seen[(a_, b_)] = x.get("fidelity") or 0
d1 = pd.DataFrame([dict(qubit_a=a_, qubit_b=b_, cz_fidelity=f_, used_at_n7=((a_, b_) in used or (b_, a_) in used)) for (a_, b_), f_ in sorted(seen.items())])
alive = {int(x["qubit"]) for x in chip["single_qubits"] if x["gate_fidelity"] > 0}
used_q = {q for ab in used for q in ab}
bq = pd.DataFrame([dict(qubit=q, row=q // 9, column=q % 9, available=q in alive, used_at_n7=q in used_q) for q in range(180)])
g1 = pd.DataFrame([dict(qubit=int(x["qubit"]), t1_us=x["t1"], t2_us=x["t2"], readout_fidelity=x["readout_fidelity"], single_qubit_gate_fidelity=x["gate_fidelity"], used_at_n7=int(x["qubit"]) in used_q)
                   for x in chip["single_qubits"] if x["gate_fidelity"] > 0])
e_rows = [("identification, n = 5", "copies for 90%"), ("identification, n = 7", "copies for 90%"), ("identification, n = 10", "copies for 90%"),
          ("channel, n = 6", "channel uses for 90%"), ("tracking, q = 0.02", "regret after 400 rounds"), ("tracking, q = 0.1", "regret after 400 rounds"),
          ("detection, n = 5", "copies for 90% balanced accuracy")]
b5 = J("PR5_bootstrap_pauli_origin" + PRS)["results"]; b7 = J("PR7_bootstrap_large_origin_onchip20_pruned_bayes.json")["results"]["onchip20_pruned_bayes"][0]
x10 = J("X10R_bign_origin.json"); ch6 = J("CH6A_channel_origin.json")
dr = J("DR5_drift_origin.json")
def drm(q, ag): return float(np.mean([dr["results"][ag][f"{dr['q'].index(q)}_{r}"]["final"] for r in range(4)]))
dt5 = J("DT5_detect_origin.json")
e = pd.DataFrame(dict(experiment=[r[0] for r in e_rows], quantity=[r[1] for r in e_rows],
                      agent_value=[b5["onchip20_pruned_bayes"]["point"], b7, x10["policies"]["onchip20_pruned"]["point"],
                                   ch6["policies"]["train_ch_s70"]["point"], drm(0.02, "onchip20_pruned"), drm(0.1, "onchip20_pruned"),
                                   max(dt5["agents"]["det50"]["copies90_cv"], dt5["agents"]["det51"]["copies90_cv"])],
                      single_copy_limit=[BOUND[5], BOUND[7], x10["bound"], ch6["bound"], dr["lower_bound_q0.02"], dr["lower_bound_q0.1"], np.ceil(1.6 * 33)]))
book("Source_Data_Fig1.xlsx", "Fig. 1", [
    ("Fig1a", "Photograph of the WK_C180 processor package", None, None),
    ("Fig1b_couplers", "Coupling graph of WK_C180 in the calibration snapshot of 15 November 2026: every coupler with its CZ fidelity (0 = not calibrated); used_at_n7 marks the seven couplers of the n = 7 runs.", d1, None),
    ("Fig1b_qubits", "Qubits of the coupling graph: position in the schematic layout (row = qubit // 9, column = qubit % 9; odd rows shifted by half a column), availability on that day and use at n = 7.", bq, None),
    ("Fig1c", "Task schematic", None, None), ("Fig1d", "Agent schematic", None, None), ("Fig1e", "Training-loop schematic", None, None),
    ("Fig1f", "CZ fidelity of the calibrated couplers (fidelity > 0.9 shown in the histogram) and the seven couplers used at n = 7; same table as Fig1b_couplers.", d1[d1.cz_fidelity > 0.9], None),
    ("Fig1g", "Single-qubit calibration of the available qubits (same snapshot); used_at_n7 marks the 14 qubits of the n = 7 runs.", g1, None),
    ("Fig1h", "Summary: value of the chip-trained agent (pruned seed 20 for identification and tracking; best detection agent for detection) and the corresponding single-copy limit on ideal hardware.", e, [("ratio = agent_value / single_copy_limit", '=IF(OR(C{r}="",D{r}=""),"",C{r}/D{r})')])])

# ---------------------------------------------------------------- Fig 2
def run_table(files):
    rows = []
    for f in files:
        for x in hist(f):
            rows.append(dict(run=f, iteration=x["it"], reward=x["reward"], two_copy_visibility=x.get("vis_ideal"), single_copy_visibility=x.get("vis_single"),
                             step_size_multiplier=x.get("lr_mult"), front_end_restart=bool(x.get("fe_restart")), processor_time_s=(x.get("info") or {}).get("qpu_ms", np.nan) / 1000 if (x.get("info") or {}).get("qpu_ms") else None))
    return pd.DataFrame(rows)
dec = []
for f in ("train_s10", "train_s11", "train_s12", "train_s20", "train_s21", "train_s22", "train_s30", "train_s31", "train_s32"):
    for x in hist(f):
        d = x.get("dec_restart")
        if d and "current" in d:
            dec.append(dict(run=f, iteration=x["it"], reward_current_decoder=d["current"], **{f"fresh_{j + 1}": (d["cands"][j] if j < len(d["cands"]) else None) for j in range(3)}, accepted=d["accepted"]))
succ = pd.DataFrame(dict(protocol=["v1", "v2", "v3", "v4"], runs=[4, 3, 3, 3], learned=[1, 2, 2, 3],
                         runs_included=["seeds 1, 10, 11, 12 (iterations 1-150)", "seeds 10, 11, 12 (from iteration 151)", "seeds 20, 21, 22", "seeds 30, 31, 32"]))
book("Source_Data_Fig2.xlsx", "Fig. 2", [
    ("Fig2a", "Reward per training iteration (the figure shows five-iteration means), protocol v1 and, from iteration 151, decoder restarts (v2).", run_table(["train_s1", "train_s10", "train_s11", "train_s12"]), None),
    ("Fig2b", "Reward per iteration, protocol v3 (seeds 20-22).", run_table(["train_s20", "train_s21", "train_s22"]), None),
    ("Fig2c", "Reward per iteration, protocol v4 (seeds 30-32); front_end_restart marks restarts.", run_table(["train_s30", "train_s31", "train_s32"]), None),
    ("Fig2d", "Runs that learned (ten-iteration reward above 0.6) per protocol version.", succ, None),
    ("Fig2e-f", "Ideal two-copy visibility (n = 2, mean over 15 Paulis) and reward per iteration, runs of protocols v3 and v4.", run_table(["train_s20", "train_s21", "train_s22", "train_s30", "train_s31", "train_s32"])[["run", "iteration", "reward", "two_copy_visibility"]], None),
    ("Fig2g", "Offline decoder restarts: reward of the current decoder and of three freshly trained decoders on held-out episodes; accepted if the best fresh decoder is better by more than 0.02.", pd.DataFrame(dec), [("best_fresh", "=MAX(D{r}:F{r})")]),
    ("Fig2h", "Processor time per iteration (s).", run_table(["train_s20", "train_s21", "train_s22", "train_s30", "train_s31", "train_s32"])[["run", "iteration", "processor_time_s"]], None)])

# ---------------------------------------------------------------- Fig 3
S = lambda s: J(f"{s}_summary_origin.json")
def acc_rows(step, keys, extra=None):
    d = S(step); rows = []
    for k in keys:
        r = d["results"].get(f"{k}_bayes")
        if r is None: continue
        g = d["grid_c"] if k == "classical" else d["grid_q"]
        for M, a in zip(g, r["acc"]):
            if a is not None and np.isfinite(a): rows.append(dict(step=step, n=d["n"], front_end=k, copies=M, accuracy=a, single_copy_limit_copies=BOUND[d["n"]]))
    if extra:
        d = S(extra); r = d["results"]["classical_bayes"]
        for M, a in zip(d["grid_c"], r["acc"]):
            if a is not None and np.isfinite(a): rows.append(dict(step=extra, n=d["n"], front_end="classical (random bases)", copies=M, accuracy=a, single_copy_limit_copies=BOUND[d["n"]]))
    return pd.DataFrame(rows)
def bs(f, k):
    d = J(f)["results"][k]
    if isinstance(d, dict): return d["point"], d["ci95"][0], d["ci95"][1]
    a = np.array(d, float); b = a[1:][np.isfinite(a[1:])]; return a[0], np.percentile(b, 2.5), np.percentile(b, 97.5)
g_rows = []
for series, pts in (("Bell", [(3, "S4_bootstrap_pauli_origin.json", "bell_bayes"), (4, "S3P_bootstrap_pauli_origin.json", "bell_bayes"), (5, "PR5_bootstrap_pauli_origin" + PRS, "bell_native_bayes"), (6, "PR6_bootstrap_pauli_origin" + PRS, "bell_native_bayes"), (7, "PR7_bootstrap_large_origin_bell_native_bayes.json", "bell_native_bayes")]),
                    ("chip-trained seed 20", [(4, "T4_bootstrap_pauli_origin.json", "onchip20_bayes"), (5, "PR5_bootstrap_pauli_origin" + PRS, "onchip20_pruned_bayes"), (6, "PR6_bootstrap_pauli_origin" + PRS, "onchip20_pruned_bayes"), (7, "PR7_bootstrap_large_origin_onchip20_pruned_bayes.json", "onchip20_pruned_bayes")]),
                    ("simulation-trained", [(3, "S4_bootstrap_pauli_origin.json", "learned_bayes"), (4, "S3P_bootstrap_pauli_origin.json", "learned_bayes"), (5, "S5_bootstrap_pauli_origin.json", "learned_bayes"), (6, "S6X_bootstrap_pauli_origin_learned.json", "learned_bayes")]),
                    ("random bases", [(3, "S4_bootstrap_pauli_origin.json", "classical_bayes"), (4, "S3P_bootstrap_pauli_origin.json", "classical_bayes"), (5, "S5C_bootstrap_pauli_origin_classical.json", "classical_bayes")])):
    for n, f, k in pts:
        p, lo, hi = bs(f, k); g_rows.append(dict(series=series, n=n, result_file=f, key=k, copies90=p, ci95_low=lo, ci95_high=hi, single_copy_limit=BOUND[n]))
g_rows += [dict(series="adaptive elimination (ideal hardware, simulation)", n=n, result_file="Supplementary Note 2", key="", copies90=v, ci95_low=None, ci95_high=None, single_copy_limit=BOUND[n]) for n, v in ((3, 50), (4, 108), (5, 235), (6, 517))]
pp = {st: json.load(open(os.path.join(PP, f"{st}_perpauli.json"))) for st in ("PR5", "PR6", "PR7")}
def pp_rows(st, keys):
    rows = []
    for k in keys:
        for P, v in pp[st][k]["paulis"].items():
            rows.append(dict(step=st, n=len(P), front_end=k, pauli="".join("IXYZ"[int(c)] for c in P), weight=v["w"], copies90=v["copies90"], circuits_eval=v["max_copies"] // 2, single_copy_limit=BOUND[len(P)]))
    return pd.DataFrame(rows)
rep = []
for st, f, k, lab in (("S5", "S5_bootstrap_pauli_origin.json", "bell_bayes", "Bell (CRX)"), ("R5", "R5_bootstrap_pauli_origin.json", "bell_bayes", "Bell (CRX)"), ("T5", "T5_bootstrap_pauli_origin.json", "bell_bayes", "Bell (CRX)"),
                      ("S5R", "S5R_bootstrap_pauli_origin.json", "bell_bayes", "Bell (CRX)"), ("PR5", "PR5_bootstrap_pauli_origin" + PRS, "bell_bayes", "Bell (CRX)"),
                      ("T5", "T5_bootstrap_pauli_origin.json", "onchip20_bayes", "chip-trained seed 20"), ("T5", "T5_bootstrap_pauli_origin.json", "onchip22_bayes", "chip-trained seed 22"),
                      ("S5R", "S5R_bootstrap_pauli_origin.json", "onchip20_bayes", "chip-trained seed 20"), ("PR5", "PR5_bootstrap_pauli_origin" + PRS, "onchip20_bayes", "chip-trained seed 20"), ("PR5", "PR5_bootstrap_pauli_origin" + PRS, "onchip22_bayes", "chip-trained seed 22")):
    p, lo, hi = bs(f, k); rep.append(dict(step=st, front_end=lab, copies90=p, ci95_low=lo, ci95_high=hi, single_copy_limit=BOUND[5]))
book("Source_Data_Fig3.xlsx", "Fig. 3", [
    ("Fig3a", "Deployment schematic", None, None),
    ("Fig3b", "Identification accuracy vs copies, n = 3 (step S4).", acc_rows("S4", ["bell", "learned", "classical"]), None),
    ("Fig3c", "Identification accuracy vs copies, n = 4 (step S3P).", acc_rows("S3P", ["bell", "learned", "classical"]), None),
    ("Fig3d", "Identification accuracy vs copies, n = 5 (step PR5; random bases from step S5C).", acc_rows("PR5", ["bell_native", "onchip20_pruned", "onchip20"], extra="S5C"), None),
    ("Fig3e", "Identification accuracy vs copies, n = 6 (step PR6).", acc_rows("PR6", ["bell_native", "onchip20_pruned", "onchip20"]), None),
    ("Fig3f", "Identification accuracy vs copies, n = 7 (step PR7).", acc_rows("PR7", ["bell_native", "onchip20_pruned"]), None),
    ("Fig3g", "Copies for 90% identification with 95% bootstrap intervals over Paulis, and the non-adaptive single-copy limit (Theorem 1).", pd.DataFrame(g_rows), [("fraction_of_limit", '=IF(OR(E{r}="",H{r}=""),"",E{r}/H{r})')]),
    ("Fig3h", "Copies for 90% identification of each Pauli, n = 5 (step PR5).", pp_rows("PR5", ["bell", "bell_native", "onchip20", "onchip20_pruned"]), None),
    ("Fig3i", "Per-Pauli copies for 90% at n = 5-7 (the figure shows the mean per weight).", pd.concat([pp_rows(st, ["bell_native", "onchip20_pruned"]) for st in ("PR5", "PR6", "PR7")]), None),
    ("Fig3j", "Repeat runs at n = 5 on different days (step names as in Supplementary Table 2).", pd.DataFrame(rep), None)])

# ---------------------------------------------------------------- Fig 4
pr = J("prune.json")["agents"]
names = ["train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32", "detect_s50", "detect_s51"]
ang = []
for nm in names:
    for i in range(13):
        ang.append(dict(agent=nm, parameter=f"theta_{i + 1}", gate=["RZ", "RY", "RZ", "RZ", "RY", "RZ", "CRX", "RZ", "RY", "RZ", "RZ", "RY", "RZ"][i],
                        learned_angle_wrapped=float((pr[nm]["theta"][i] + np.pi) % (2 * np.pi) - np.pi), pruned_angle=float(pr[nm]["theta_pruned"][i])))
tb = []
for lab, th in (("Bell", bell_theta()), ("seed 20, as trained", pr["train_s20"]["theta"]), ("seed 20, pruned", pr["train_s20"]["theta_pruned"]), ("shortcut (no sign), seed 20", J("train_nt_s20.json")["best"]["theta"])):
    g = tables(np.array(th, float))[2]
    for si, sg in enumerate("XYZ", 1):
        tb.append(dict(front_end=lab, sigma=sg, **{f"beta_{b:02b}": float(g[si][b]) for b in range(4)}))
vj = pd.DataFrame([dict(agent=k, V_as_trained=pr[k]["V"], V_pruned=pr[k]["V_pruned"], J_as_trained=pr[k]["J"][0], J_pruned=pr[k]["J"][1],
                        CZ_as_trained=pr[k]["cz"][0], CZ_pruned=pr[k]["cz"][1], pulses_as_trained=pr[k]["pulses"][0], pulses_pruned=pr[k]["pulses"][1]) for k in names])
kap = []
for st in ("PR5", "PR6", "PR7"):
    for k, v in S(st)["kappa_q"].items(): kap.append(dict(step=st, n=S(st)["n"], front_end=k, fitted_per_site_visibility_kappa=v))
fb = []
for st in ("PR5", "PR6"):
    d = J(f"{st}_bootstrap_pauli_origin" + PRS)
    for k in ("bell", "bell_native", "onchip20", "onchip20_pruned", "onchip22", "onchip22_pruned"):
        r = d["results"][f"{k}_bayes"]; fb.append(dict(step=st, n=d["n"], front_end=k, copies90=r["point"], ci95_low=r["ci95"][0], ci95_high=r["ci95"][1], single_copy_limit=d["single_copy_lower_bound_ideal"]))
for k in ("bell_native", "onchip20_pruned"):
    p, lo, hi = bs(f"PR7_bootstrap_large_origin_{k}_bayes.json", f"{k}_bayes"); fb.append(dict(step="PR7", n=7, front_end=k, copies90=p, ci95_low=lo, ci95_high=hi, single_copy_limit=BOUND[7]))
ratios = []
for st in ("PR5", "PR6"):
    d = J(f"{st}_bootstrap_pauli_origin" + PRS)["results"]
    for lab, a, b in (("Bell: native / CRX", "bell", "bell_native"), ("seed 20: pruned / as trained", "onchip20", "onchip20_pruned"), ("seed 22: pruned / as trained", "onchip22", "onchip22_pruned")):
        X = np.array(d[f"{a}_bayes"]["reps"], float); Y = np.array(d[f"{b}_bayes"]["reps"], float); rr = (Y / X)[1:]; rr = rr[np.isfinite(rr)]
        ratios.append(dict(step=st, comparison=lab, ratio=Y[0] / X[0], ci95_low=np.percentile(rr, 2.5), ci95_high=np.percentile(rr, 97.5), fraction_of_resamples_below_1=float(np.mean(rr < 1))))
ph = []
for st in ("PR5", "PR6"):
    for a, b, lab in (("bell", "bell_native", "Bell"), ("onchip20", "onchip20_pruned", "seed 20"), ("onchip22", "onchip22_pruned", "seed 22")):
        for P in pp[st][a]["paulis"]:
            ph.append(dict(step=st, agent=lab, pauli="".join("IXYZ"[int(c)] for c in P), copies90_as_trained_or_CRX=pp[st][a]["paulis"][P]["copies90"], copies90_pruned_or_native=pp[st][b]["paulis"][P]["copies90"]))
book("Source_Data_Fig4.xlsx", "Fig. 4", [
    ("Fig4a", "Learned angles of the ten front-ends trained on the chip (wrapped to [-pi, pi)) and their pruned values (rad).", pd.DataFrame(ang), None),
    ("Fig4b", "Two-copy response tables g3[sigma, beta] = <beta|V (sigma x sigma) V^dag|beta>.", pd.DataFrame(tb), None),
    ("Fig4c-d", "Ideal two-copy visibility V, hardware-aware objective J, CZ gates and single-qubit pulses per site before and after pruning.", vj, None),
    ("Fig4e", "Per-site visibility kappa fitted on the chip.", pd.DataFrame(kap), None),
    ("Fig4f", "Copies for 90% identification before and after pruning (95% bootstrap intervals).", pd.DataFrame(fb), None),
    ("Fig4g", "Paired ratios of copies with 95% paired-bootstrap intervals (shared Pauli resamples, 200 resamples).", pd.DataFrame(ratios), None),
    ("Fig4h", "Per-Pauli copies for 90% before and after pruning.", pd.DataFrame(ph), [("ratio = pruned / as trained", '=IF(OR(D{r}="",E{r}=""),"",E{r}/D{r})')])])

# ---------------------------------------------------------------- Fig 5
def svis(th, k_list, n=2):
    g = tables(np.asarray(th, float)); P = np.array(list(itertools.product(range(4), repeat=n)))[1:]; B = np.array(list(itertools.product(range(4), repeat=n)))
    out = []
    for k in k_list:
        p_ = np.ones((len(P), len(B)))
        for i in range(n): p_ *= g[k][P[:, i]][:, B[:, i]]
        out.append(float((p_ ** 2).mean()))
    return max(out)
fin = []
for k in ("train_s1", "train_s11", "train_s12", "train_s20", "train_s22", "train_s30", "train_s31", "train_s32"):
    th = pr[k]["theta"]; fin.append(dict(front_end=k, trained_with_sign=True, single_copy_visibility=svis(th, [0, 1]), two_copy_visibility=svis(th, [2])))
for k in ("train_nt_s20", "train_nt_s22"):
    th = J(k + ".json")["best"]["theta"]; fin.append(dict(front_end=k, trained_with_sign=False, single_copy_visibility=svis(th, [0, 1]), two_copy_visibility=svis(th, [2])))
th = bell_theta(); fin.append(dict(front_end="Bell", trained_with_sign=None, single_copy_visibility=svis(th, [0, 1]), two_copy_visibility=svis(th, [2])))
g = tables(np.array(J("train_nt_s20.json")["best"]["theta"], float)); rt = []
for k, lab in ((0, "copy A (g1)"), (1, "copy B (g2)"), (2, "both (g3)")):
    for si, sg in enumerate("XYZ", 1): rt.append(dict(table=lab, sigma=sg, **{f"beta_{b:02b}": float(g[k][si][b]) for b in range(4)}))
def sc_rows(st):
    d = J(f"{st}_shortcut_origin.json"); rows = []
    for k, r in d["agents"].items():
        for i, M in enumerate(r["copies"]):
            a = r["acc"][i]
            if a is None or not np.isfinite(a): continue
            rows.append(dict(step=st, n=d["n"], front_end=k, copies=M, accuracy=a, ci95_low=r["acc_lo"][i], ci95_high=r["acc_hi"][i], noise_model_accuracy=r["model_acc"][i]))
    return pd.DataFrame(rows)
sc = J("shortcut_scaling_ideal.json")["copies90"]
ex = [dict(n=int(n), front_end=k, source="noiseless extrapolation", copies90=v) for n, row in sc.items() for k, v in row.items() if k in ("bell", "shortcut20", "shortcut22")]
for st in ("SC3", "SC4"):
    d = J(f"{st}_shortcut_origin.json")
    for k, r in d["agents"].items():
        ex.append(dict(n=d["n"], front_end=k, source=f"measured on chip ({st}); n.r. = not reached within {r['max_copies']} copies" if not np.isfinite(r["copies90"]) else f"measured on chip ({st})", copies90=r["copies90"]))
ab = [dict(step=st, n=J(f"{st}_shortcut_origin.json")["n"], front_end=k, single_copy_a=r["a"], two_copy_b=r["b"]) for st in ("SC3", "SC4") for k, r in J(f"{st}_shortcut_origin.json")["agents"].items() if k != "bell_native"]
book("Source_Data_Fig5.xlsx", "Fig. 5", [
    ("Fig5a-b", "Reward and ideal two-copy visibility per iteration, training with (train_s20, train_s22) and without (train_nt_s20, train_nt_s22) the shared sign (first 120 iterations shown).", run_table(["train_s20", "train_s22", "train_nt_s20", "train_nt_s22"])[["run", "iteration", "reward", "two_copy_visibility"]], None),
    ("Fig5c", "Single-copy and two-copy visibility (n = 2) of the final front-ends.", pd.DataFrame(fin), None),
    ("Fig5d", "Response tables of the shortcut front-end of seed 20.", pd.DataFrame(rt), None),
    ("Fig5e", "Identification accuracy vs copies on the task without the sign, n = 3 (step SC3), with 95% bootstrap intervals and the noise-model prediction.", sc_rows("SC3"), None),
    ("Fig5f", "Same as Fig5e for n = 4 (step SC4).", sc_rows("SC4"), None),
    ("Fig5g", "Copies for 90%: noiseless extrapolation of the learned front-ends and measurements on the chip.", pd.DataFrame(ex), None),
    ("Fig5h", "Per-site visibilities of single-copy (a) and two-copy (b) terms fitted on the chip.", pd.DataFrame(ab), None)])

# ---------------------------------------------------------------- Fig 6
def veff(th, dB=np.pi / 2):
    t = np.array(th, float).copy(); t[3] += dB; return float((tables(t)[2][1:] ** 2).mean())
def v0(th): return float((tables(np.array(th, float))[2][1:] ** 2).mean())
cc = []
for f0, f1 in (("train_s20", "adapt_s20"), ("train_s22", "adapt_s22")):
    t0 = J(f0 + ".json")["best"]["theta"]; t1 = J(f1 + ".json")["theta"]
    cc.append(dict(agent=f1, no_error=v0(t0), error_not_adapted=veff(t0), error_adapted=veff(t1)))
reg = []
for qi, q in enumerate(dr["q"]):
    for ag in ("bell_native", "onchip20_pruned", "classical"):
        for r in range(4):
            for i, v in enumerate(dr["results"][ag][f"{qi}_{r}"]["curve"]):
                reg.append(dict(q=q, agent=ag, run=r + 1, round=i * 5 + 1, cumulative_regret=v))
fr = [dict(q=q, run=r + 1, Bell_native=dr["results"]["bell_native"][f"{qi}_{r}"]["final"], chip_trained_pruned=dr["results"]["onchip20_pruned"][f"{qi}_{r}"]["final"],
           random_bases=dr["results"]["classical"][f"{qi}_{r}"]["final"], single_copy_limit=dr[f"lower_bound_q{q}"]) for qi, q in enumerate(dr["q"]) for r in range(4)]
sim = pd.DataFrame(dict(per_site_visibility=[1.0, 0.95, 0.866, 1.0, 0.95, 0.866], q=[0.02] * 3 + [0.1] * 3, simulated_Bell_regret=[40, 69, 142, 133, 220, 318],
                        measured_Bell_regret=[None, None, 170, None, None, 332], single_copy_limit=[dr["lower_bound_q0.02"]] * 3 + [dr["lower_bound_q0.1"]] * 3))
det = []
for n in (5, 6, 7):
    d = J(f"DT{n}_detect_origin.json")
    for k, a in d["agents"].items():
        for M, v in zip(d["grid"], a["acc_cv"]):
            if v is not None and np.isfinite(v): det.append(dict(n=n, front_end=k, copies=M, balanced_accuracy=v, single_copy_ceiling=min(1, 0.5 + M / (4 * (2 ** n + 1)))))
dc = J("DTC5_detect_origin.json")
for M, v in zip(dc["grid"], dc["agents"]["classical"]["acc_cv"]):
    if v is not None and np.isfinite(v): det.append(dict(n=5, front_end="random bases", copies=M, balanced_accuracy=v, single_copy_ceiling=min(1, 0.5 + M / 132)))
cu = [dict(n=n, front_end=k, mean_delay_copies=2 * a["delay_pairs_mean"], sem_copies=2 * a["delay_pairs_sem"], fraction_censored=a.get("delay_censored_frac"))
      for n in (5, 6, 7) for k, a in J(f"DT{n}_detect_origin.json")["agents"].items()]
cu.append(dict(n=5, front_end="random bases", mean_delay_copies=2 * dc["agents"]["classical"]["delay_pairs_mean"], sem_copies=2 * dc["agents"]["classical"]["delay_pairs_sem"], fraction_censored=dc["agents"]["classical"].get("delay_censored_frac")))
book("Source_Data_Fig6.xlsx", "Fig. 6", [
    ("Fig6a-b", "Online adaptation: reward, two-copy visibility evaluated without the injected error and step-size multiplier per iteration (error injected from iteration 11).", run_table(["adapt_s20", "adapt_s22"])[["run", "iteration", "reward", "two_copy_visibility", "step_size_multiplier"]], None),
    ("Fig6c", "Site two-copy visibility without the error, and with the R_Z(pi/2) error before and after adaptation.", pd.DataFrame(cc), None),
    ("Fig6d-e", "Cumulative regret of the streamed drift-tracking task (every fifth round; mean over ten shot draws), per run.", pd.DataFrame(reg), None),
    ("Fig6f", "Regret after 400 rounds per run; the agents of a run share the hidden sequence.", pd.DataFrame(fr), None),
    ("Fig6g", "Simulated regret of the Bell measurement for three per-site visibilities, measured values and single-copy limits.", sim, None),
    ("Fig6h-j", "Detection: balanced accuracy (threshold by two-fold cross-validation) vs copies, and the single-copy ceiling 1/2 + M/(4(d+1)).", pd.DataFrame(det), None),
    ("Fig6k", "GLR-CUSUM mean detection delay at a false-alarm horizon of 500 pairs.", pd.DataFrame(cu), None)])


# ---------------------------------------------------------------- Fig 7
SIM = os.path.join(_ROOT, "hw", "sim", "results", "ablation_summary.json")
ab = json.load(open(SIM))
def bsr(f, k):
    d = J(f)["results"][k]; a = np.array(d["reps"] if isinstance(d, dict) else d, float); b = a[1:][np.isfinite(a[1:])]
    return a[0], np.percentile(b, 2.5), np.percentile(b, 97.5)
a7 = []
for st, n, src in (("PR5", 5, "bs"), ("PR6", 6, "bs"), ("PR7", 7, "bs"), ("X8", 8, "bign"), ("X9", 9, "bign"), ("X10R", 10, "bign")):
    for pol in ("bell_native", "onchip20_pruned"):
        if src == "bs":
            f = f"PR{n}_bootstrap_large_origin_{pol}_bayes.json" if n == 7 else f"PR{n}_bootstrap_pauli_origin" + PRS
            p, lo, hi = bsr(f, f"{pol}_bayes")
        else:
            q = J(f"{st}_bign_origin.json")["policies"][pol]; p, (lo, hi) = q["point"], q["ci"]
        a7.append(dict(step=st, n=n, qubits=2 * n, front_end=pol, copies90=p, ci95_low=lo, ci95_high=hi, single_copy_limit=BOUND[n]))
b7 = []
for st in ("X8", "X9", "X10R", "X10"):
    d = J(f"{st}_bign_origin.json"); dd = 2 ** d["n"]
    for m, acc in zip(d["grid"], d["policies"]["bell_native"]["curve"]):
        b7.append(dict(step=st, n=d["n"], copies=m, accuracy_bell_native=acc,
                       single_copy_max_accuracy=min(1.0, (m / (dd + 1) + np.log(2)) / np.log(dd * dd - 1))))
c7 = [dict(front_end=k, seed=r["seed"], iterations=r["iters"], final_reward=r["final"], information_bits=r["info"],
           site_visibility=(r.get("site_vis") or [None])[0], learned=r["final"] > 0.5)
      for k in ("site13", "su4_shared", "su4_free", "hea4", "product6", "channel26", "channel_in13", "channel_echo13") for r in ab[k]]
hz = ab["haar_site_vis"]; d7 = pd.DataFrame(dict(bin_low=np.linspace(0, 1, 41)[:-1], bin_high=np.linspace(0, 1, 41)[1:], count=hz["hist"]))
e7 = []
for f, lb in (("train_s30", "two-copy, seed 30"), ("train_s31", "two-copy, seed 31"), ("train_s32", "two-copy, seed 32"),
              ("train_sc_s60", "single-copy, seed 60"), ("train_sc_s61", "single-copy, seed 61")):
    for h_ in hist(f):
        e7.append(dict(run=lb, iteration=h_["it"], reward=h_["reward"], information_bits=h_.get("info_ideal")))
g7 = [dict(run=f"seed {sd}", iteration=h_["it"], reward=h_["reward"], information_bits=h_.get("info_ideal"))
      for sd in (70, 71) for h_ in hist(f"train_ch_s{sd}")]
hi7 = []
for st, pol in (("CH5", "bell_channel"), ("CH5A", "train_ch_s70"), ("CH6", "bell_channel"), ("CH6A", "train_ch_s70")):
    d = J(f"{st}_channel_origin.json")
    for m, acc in zip(d["grid"], d["policies"][pol]["curve"]):
        hi7.append(dict(step=st, n=d["n"], front_end=pol, channel_uses=m, accuracy=acc, uses90=d["policies"][pol]["point"], limit_without_memory=d["bound"]))
j7 = pd.DataFrame(dict(strategy=["best input without entanglement (numerical optimum)", "learned input, seed 70 (final)", "Bell pair and Bell measurement"],
                       information_bits=[0.191, hist("train_ch_s70")[-1]["info_ideal"], 0.962]))
book("Source_Data_Fig7.xlsx", "Fig. 7", [
    ("Fig7a", "Copies for 90% identification at n = 5-10 with 95% bootstrap intervals, and the non-adaptive single-copy limit.", pd.DataFrame(a7),
     [("ratio = copies90 / single_copy_limit", '=IF(OR(E{r}="",H{r}=""),"",E{r}/H{r})')]),
    ("Fig7b", "Accuracy of the native Bell measurement against copies at n = 8-10 (X10: first n = 10 run with one faulty coupler) and the largest accuracy of any non-adaptive single-copy agent (Fano).", pd.DataFrame(b7), None),
    ("Fig7c", "Simulated training at n = 2 (hw/sim): final ten-iteration reward, information of the final front-end and site visibility per seed and front-end family.", pd.DataFrame(c7), None),
    ("Fig7d", "Site visibility of 20,000 Haar-random two-qubit unitaries (histogram); learned values in sheet Fig7c.", d7, None),
    ("Fig7e", "Reward per iteration on WK_C180 of the two-copy agents (seeds 30-32) and the single-copy agents (seeds 60, 61).", pd.DataFrame(e7), None),
    ("Fig7f", "Schematic of the channel task", None, None),
    ("Fig7g", "Training of the input circuit of the channel task on WK_C180 (seeds 70, 71).", pd.DataFrame(g7), None),
    ("Fig7h-i", "Channel task: accuracy against channel uses at n = 5, 6 for the reference (Bell pair, Bell measurement) and the learned input.", pd.DataFrame(hi7), None),
    ("Fig7j", "Information about the hidden Pauli per channel use at n = 2 (noiseless).", j7, None)])
