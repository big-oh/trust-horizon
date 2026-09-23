# trust-horizon

Code and paper source for *Lusser's Law for Agents: The Compounding-Reliability Gap in Long-Horizon Agent Harnesses*
(A. Siddiqui, WebridgeAI Research, preprint, September 2026).

The paper models an AI agent loop as a series reliability system. Its parameters are per-step error ε₀, sensor (verifier) recall *r* and false-alarm rate *f*, retry budget *m*, context degradation γ, handoff loss ρ per context reset, and reset interval *L*. From this model it derives closed forms for the trust horizon, the task length an agent completes with a target reliability such as 80%. These cover the invariance of the trust-to-capability ratio under constant hazard, task heterogeneity (frailty), the verification gain, error laundering under retries, a square-root law for context resets, checkpoint cost, and a combined Trust Horizon Equation. This repository has every closed form, the Monte Carlo simulator used to check them, the scripts that produce every number and figure in the paper, and the LaTeX source.

**All results are synthetic.** They are closed-form derivations and Monte Carlo simulations of a stylized agent loop. No language model is called, and no number in this repository is a measurement of a real agent. The parameter values are illustrative assumptions, not estimates from deployed systems.

## Reproducing the paper

Tested with Python 3.14 on macOS (arm64). The pinned versions matter: simulated values depend on NumPy's random-number streams.

```bash
pip install -r requirements.txt
make            # validate.py -> experiments.py -> ablation.py -> figures.py -> paper_numbers.py
make paper      # optional: rebuild paper/main_ieee.pdf (needs pdflatex and bibtex)
make cover      # optional: rebuild the cover page and paper/main_ieee_with_cover.pdf (needs lualatex and pypdf)
```

Without `make`, run the scripts in this order from the repository root:

```bash
python validate.py      # closed forms vs. simulation (Table V)            -> validation.json
python experiments.py   # tau-bench Beta fits, nines table, Pareto sweep    -> results.json
python ablation.py      # harness ablation, error-laundering check           -> ablation.json, abl_curves.json
python figures.py       # all figures (needs the three JSON files above)     -> paper/figs_ieee/*.pdf, returns_pts.json
python paper_numbers.py # checks every number quoted in the paper against the outputs; exits 1 on a mismatch
```

The full pipeline takes under a minute on a laptop. All random seeds are fixed, so repeated runs give identical output.

To rebuild the PDF by hand:

```bash
cd paper && pdflatex main_ieee && bibtex main_ieee && pdflatex main_ieee && pdflatex main_ieee
```

## Contents

| Path | What it is |
|---|---|
| `trust_horizon.py` | Every closed form in the paper, and the vectorized simulator of the harnessed loop |
| `validate.py` | Closed-form predictions versus simulation |
| `experiments.py` | τ-bench heterogeneity fits, per-step reliability table, cost–reliability (Pareto) sweep |
| `ablation.py` | Ablation of harness controls with bootstrap intervals; stuck-step laundering check |
| `figures.py` | All figures in the paper |
| `paper_numbers.py` | Checks each simulated or computed number in the paper against the outputs |
| `*.json` | Outputs of the scripts above (committed, so they can be diffed after a rerun) |
| `paper/` | LaTeX source (`main_ieee.tex`), bibliography (`refs.bib`, `main_ieee.bbl`), figures, the compiled PDF, and the same PDF with a cover page (`main_ieee_with_cover.pdf`) |
| `paper/cover/` | Cover page source (`gradient.tex`), its background generator (`gradient_bg.py`), and the script that prepends it to the paper (`merge.py`) |

## License

MIT (see `LICENSE`).
