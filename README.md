# Umami-MRNN web demo

Predicts umami peptides and their taste thresholds from sequence. It runs entirely in the browser,
so it can be hosted for free on GitHub Pages with no server or ML framework.

**Live demo:** https://daibiaoxuwu.github.io/umami-mrnn/

> Lulu Qi, Jialuo Du, Yue Sun, Yongzhao Xiong, Xinyao Zhao, Daodong Pan, Yueru Zhi, Yali Dang, Xinchang Gao.
> *Umami-MRNN: Deep learning-based prediction of umami peptide using RNN and MLP.* Food Chemistry, 2023.
> [doi:10.1016/j.foodchem.2022.134935](https://doi.org/10.1016/j.foodchem.2022.134935)

Training code: [Umami-MRNN-train](https://github.com/daibiaoxuwu/Umami-MRNN-train).
This replaces the earlier Flask/Heroku demo ([umami-mrnn-flask-heroku](https://github.com/daibiaoxuwu/umami-mrnn-flask-heroku)).

## Files

| Path | Purpose |
|---|---|
| `index.html` | The web page |
| `predict.js` | Feature extraction and both networks in plain JavaScript |
| `model/` | Trained weights (float32) and their layout |
| `example.fasta` | Example input |
| `tools/BP.h5`, `tools/RNN.h5` | Original Keras models |
| `tools/export_weights.py` | Rebuilds `model/` from the H5 files (needs `h5py`, `numpy`) |
| `tools/make_fixture.py`, `tools/prepdata.py` | Generate reference predictions with TensorFlow using the original feature code |
| `tests/` | Checks `predict.js` against those reference predictions |

## Run locally

```sh
python -m http.server     # then open http://localhost:8000
node tests/predict.test.mjs
```

The JavaScript port matches the original TensorFlow models on all 109 test peptides
(largest difference 2e-4 mmol/L; every umami/non-umami call agrees).

## Model

- **MLP:** 1,093 descriptors (AAC, CTD, DDE, DPC; definitions from [iFeature](https://github.com/Superzchen/iFeature)) → 256 → 32 → 1.
- **RNN:** each residue encoded by a 10-bit overlapping property code → two stacked 32-unit SimpleRNNs → 1.
- Prediction = 0.67 × MLP + 0.33 × RNN. A threshold below 40 mmol/L is classed as umami.
- Accepts peptides of 2–39 standard amino acids.
