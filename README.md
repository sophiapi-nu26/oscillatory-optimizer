# Oscillatory Information Optimizer

Interactive Streamlit tool for studying oscillatory information signals in the
Shi repelling-with-information opinion-dynamics model.

## Run

```
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```
pip install -r requirements-dev.txt
pytest tests/
```

## Layout

- `network_generators.py` — signed network templates
- `shi_model.py` — builds $W = I - \alpha L_+ + \beta L_- - D$ and stability diagnostics
- `response.py` — frequency-response matrix $B_\omega = (e^{i\omega}I - W)^{-1}D$ and trajectories
- `optimizer.py` — closed-form / iterative solvers for $\max_a a^*Qa$ on $\ell_p$ balls
- `plots.py` — Plotly figures
- `app.py` — Streamlit entry point

See `../mvp_plan.md` for the design rationale.
