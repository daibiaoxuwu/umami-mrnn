"""Export the Keras H5 models to model/weights.bin (float32) + model/manifest.json.

Usage: python tools/export_weights.py   (needs only h5py and numpy)
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Keras layer name -> tensor names, in the order predict.js expects them.
LAYERS = {
    'BP.h5': {'dense': 'bp.d1', 'dense_1': 'bp.d2', 'dense_2': 'bp.d3'},
    'RNN.h5': {'simple_rnn': 'rnn.r1', 'simple_rnn_1': 'rnn.r2', 'dense_3': 'rnn.d'},
}

manifest, chunks, offset = {}, [], 0
for file, layers in LAYERS.items():
    with h5py.File(ROOT / 'tools' / file, 'r') as f:
        for layer, prefix in layers.items():
            def visit(name, obj):
                global offset
                if not isinstance(obj, h5py.Dataset):
                    return
                key = f"{prefix}.{name.split('/')[-1].split(':')[0]}"  # e.g. bp.d1.kernel
                arr = np.asarray(obj, dtype='<f4')
                manifest[key] = {'shape': list(arr.shape), 'offset': offset}
                chunks.append(arr.tobytes())
                offset += arr.size
            f['model_weights'][layer].visititems(visit)

(ROOT / 'model').mkdir(exist_ok=True)
(ROOT / 'model' / 'weights.bin').write_bytes(b''.join(chunks))
(ROOT / 'model' / 'manifest.json').write_text(json.dumps(manifest, indent=1))
print(f'{len(manifest)} tensors, {offset} floats')
