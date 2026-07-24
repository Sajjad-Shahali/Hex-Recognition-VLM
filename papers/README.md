# Papers

Citation library backing the architecture/RL design choices in
`docs/system_design.md`. Found via Consensus MCP search during this
project (6 of a 12-query budget used total). Each file below is a short,
self-contained note per paper: title, authors, year, why it was searched
for, what claim it backs, and the abstract as returned by the search — not
the full paper PDF (not redistributed; each file links to its Consensus
page for the original source).

| # | Paper | Backs |
|---|---|---|
| [1](01_liu2022_crnn_ctpn.md) | Liu (2022), *Sequence Recognition of Scene Text Based on CRNN and CTPN Models* | CRNN+CTC architecture choice (§2) |
| [2](02_su2025_reward_bridge.md) | Su et al. (2025), *Crossing the Reward Bridge: RLVR Across Diverse Domains* | Sparse-reward RLVR framing (§3.1) |
| [3](03_liu2026_gdpo.md) | Liu et al. (2026), *GDPO: Group reward-Decoupled Normalization Policy Optimization* | Multi-objective reward decomposition, GRPO (§3.1, §3.3) |
| [4](04_yousef2018_fcn_ocr.md) | Yousef et al. (2018), *Accurate, Data-Efficient, Unconstrained Text Recognition with CNNs* | FCN (recurrence-free) ablation variant (§2.1) |
| [5](05_hernandez_diaz2021_rethinking.md) | Hernandez Diaz et al. (2021), *Rethinking Text Line Recognition Models* | ConvAttn (self-attention) ablation variant (§2.1) |
| [6](06_coquenet2020_gated_fcn.md) | Coquenet et al. (2020), *Recurrence-free unconstrained handwritten text recognition using gated FCN* | Corroborates FCN design (§2.1) |

Full citation list with inline reference markers: `docs/system_design.md`
References section.
