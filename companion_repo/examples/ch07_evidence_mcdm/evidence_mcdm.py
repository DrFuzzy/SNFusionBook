"""Chapter 7 worked examples: evidence-theoretic fusion of heterogeneous maritime cues (Dempster
and Smets combination, conflict, pignistic probabilities), the cable-event interpretation example,
and AHP-weighted MCDM alert prioritisation with a one-weight sensitivity analysis.
Deterministic (no random numbers).  Writes output/mcdm_sensitivity.csv.
"""
# %% [markdown]
# # Evidence combination and MCDM prioritisation (Chapter 7)
# %%
import os
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Evidence theory helpers (focal elements as frozensets)
# %%
def combine_smets(m1, m2):
    out = {}
    for a, va in m1.items():
        for b, vb in m2.items():
            c = a & b
            out[c] = out.get(c, 0.0) + va * vb
    return out

def combine_dempster(m1, m2):
    m = combine_smets(m1, m2)
    K = m.pop(frozenset(), 0.0)
    return {a: v / (1 - K) for a, v in m.items()}, K

def pignistic(m, omega):
    return {w: sum(v / len(a) for a, v in m.items() if w in a and len(a) > 0) for w in omega}

def show(m, title):
    print(title)
    for a, v in sorted(m.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
        print(f"   m({{{','.join(sorted(a))}}}) = {v:.4f}")

# %% [markdown]
# ## Maritime example: frame {N: normal transit, L: loitering/fishing, T: infrastructure threat}
# %%
fs = frozenset
Om = fs("NLT")
m_ais = {fs("N"): 0.60, Om: 0.40}                              # AIS: declared transit, plausible speed
m_rad = {fs("LT"): 0.50, fs("T"): 0.20, Om: 0.30}               # radar: 400 m from the AIS position, slow
m_img = {fs("L"): 0.40, fs("LT"): 0.30, Om: 0.30}               # imagery: gear deployed over the side
m_cab = {fs("T"): 0.50, fs("LT"): 0.20, Om: 0.30}               # cable DAS: windowed-NIS alarm nearby
sources = [("AIS", m_ais), ("radar", m_rad), ("imagery", m_img), ("cable", m_cab)]
m = m_ais; Ks = []
for name, mi in sources[1:]:
    m, K = combine_dempster(m, mi); Ks.append((name, K))
    print(f"after combining {name}: pairwise conflict K = {K:.4f}")
show(m, "fused BBA (Dempster):")
bet = pignistic(m, "NLT"); print("pignistic:", {k: round(v, 4) for k, v in bet.items()})
# total conflict (Smets, unnormalised, all four sources)
ms = m_ais
for _, mi in sources[1:]:
    ms = combine_smets(ms, mi)
print(f"total conflict mass m(empty) under the Smets rule, four sources: {ms[fs()]:.4f}")
m12, K12 = combine_dempster(m_ais, m_rad)
print(f"AIS vs radar alone: K = {K12:.4f}; Bel(T) = {m12.get(fs('T'),0):.4f}; Pl(T) = {sum(v for a,v in m12.items() if 'T' in a):.4f}")
print(f"fused: Bel(T) = {m.get(fs('T'),0):.4f}, Pl(T) = {sum(v for a,v in m.items() if 'T' in a):.4f}, "
      f"Bel({{L,T}}) = {sum(v for a,v in m.items() if a <= fs('LT')):.4f}")

# %% [markdown]
# ## Cable example: frame {V: passing vessel, A: anchor/trawl contact, S: seismic, C: coupling artefact}
# %%
Oc = fs("VASC")
m_spec = {fs("VA"): 0.50, fs("S"): 0.10, Oc: 0.40}             # DAS spectrum: low-frequency broadband
m_res = {fs("AC"): 0.60, Oc: 0.40}                              # residuals: localised, persistent over 4 s
m_aisp = {fs("VA"): 0.70, Oc: 0.30}                             # AIS: vessel within 300 m, 1.5 kn
m_seis = {fs("VAC"): 0.80, Oc: 0.20}                            # seismic network: no regional event
mc = m_spec; Kc = []
for name, mi in [("residual", m_res), ("AIS", m_aisp), ("seismic", m_seis)]:
    mc, K = combine_dempster(mc, mi); Kc.append(K)
    print(f"cable example, after {name}: K = {K:.4f}")
show(mc, "cable fused BBA:")
print("cable pignistic:", {k: round(v, 4) for k, v in pignistic(mc, "VASC").items()})

# %% [markdown]
# ## MCDM: AHP weights (principal eigenvector) and additive scores for four alerts
# %%
crit = ["distance", "association", "deviation", "innovation", "persistence"]
A = np.array([[1, 3, 2, 2, 4],
              [1/3, 1, 1/2, 1/2, 2],
              [1/2, 2, 1, 1, 3],
              [1/2, 2, 1, 1, 3],
              [1/4, 1/2, 1/3, 1/3, 1]])
vals, vecs = np.linalg.eig(A)
i = np.argmax(vals.real); w = np.abs(vecs[:, i].real); w /= w.sum()
lam = vals[i].real; n = len(crit); CI = (lam - n) / (n - 1); RI = 1.12; CR = CI / RI
print("AHP weights:", dict(zip(crit, np.round(w, 4))), f" lambda_max = {lam:.4f}, CI = {CI:.4f}, CR = {CR:.4f}")
alerts = ["A1 slow near cable", "A2 corridor exit", "A3 AIS/radar mismatch", "A4 loitering anchorage"]
Qm = np.array([[0.90, 0.80, 0.40, 0.60, 0.70],
               [0.30, 0.90, 0.90, 0.20, 0.50],
               [0.60, 0.50, 0.30, 0.95, 0.40],
               [0.20, 0.95, 0.10, 0.10, 0.90]])
S = Qm @ w
for a, s in sorted(zip(alerts, S), key=lambda z: -z[1]):
    print(f"   S({a}) = {s:.4f}")
# sensitivity: vary the innovation-test weight, renormalise the other four proportionally
rows = []
j_inn = crit.index("innovation")
for wj in np.linspace(0.05, 0.60, 111):
    ww = w.copy(); ww *= (1 - wj) / (1 - w[j_inn]); ww[j_inn] = wj
    Sx = Qm @ ww
    rows.append({"w_innovation": wj, **{f"S{i+1}": Sx[i] for i in range(4)}, "top": alerts[int(np.argmax(Sx))][:2]})
df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "mcdm_sensitivity.csv"), index=False, float_format="%.4f")
chg = df.top.ne(df.top.shift()).to_numpy().nonzero()[0]
print("rank-one alert as the innovation weight varies:", [(round(float(df.w_innovation[j]), 3), df.top[j]) for j in chg])
# exact crossing of S(A1) and S(A3) as a function of the innovation weight (linear in w_j)
f = lambda wj: (lambda ww: (Qm @ ww)[0] - (Qm @ ww)[2])(np.r_[w * (1 - wj) / (1 - w[j_inn])][:j_inn].tolist() + [wj] + (w * (1 - wj) / (1 - w[j_inn]))[j_inn+1:].tolist())
from scipy.optimize import brentq
print(f"S(A1) = S(A3) at w_innovation = {brentq(f, 0.05, 0.6):.4f} (nominal {w[j_inn]:.4f})")
