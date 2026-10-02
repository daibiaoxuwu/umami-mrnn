"""Write tests/fixture.json with predictions from the original Keras models.

Run in an environment with TensorFlow (tf_keras is used if available):
    python tools/make_fixture.py
"""
import json
from pathlib import Path

import numpy as np

try:
    import tf_keras as keras
except ImportError:
    from tensorflow import keras

import prepdata  # original feature code from the Heroku app

ROOT = Path(__file__).resolve().parent.parent
OPF = {'A': '0000100110', 'R': '1101000000', 'N': '1000000100', 'D': '1011000100', 'C': '1000100110',
       'Q': '1000000000', 'E': '1011000000', 'G': '0000100110', 'H': '1101101000', 'I': '0000110000',
       'L': '0000110000', 'K': '1101100000', 'M': '0000100000', 'F': '0000101000', 'P': '0000000101',
       'S': '1000000110', 'T': '1000100100', 'W': '1000101000', 'Y': '1000101000', 'V': '0000110100'}

lines = (ROOT / 'example.fasta').read_text().split()
seqs = lines[1::2] + ['AC', 'WY', 'AAAAAA', 'KKKKKKKKKK', 'ACDEFGHIKLMNPQRSTVWY',
                      'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRST', 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTV',
                      'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVW', 'CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC']
seqs = list(dict.fromkeys(seqs))

bp_model = keras.models.load_model(ROOT / 'tools' / 'BP.h5')
rnn_model = keras.models.load_model(ROOT / 'tools' / 'RNN.h5')
x_bp = np.array([prepdata.work(s) for s in seqs], dtype='float64')
x_rnn = np.array([[[int(c) for c in OPF[a]] for a in s] + [[2] * 10] * (39 - len(s)) for s in seqs], dtype='float64')
bp = bp_model.predict(x_bp)[:, 0]
rnn = rnn_model.predict(x_rnn)[:, 0]

fixture = [{'seq': s, 'bp': float(b), 'rnn': float(r), 'threshold': float(0.67 * b + 0.33 * r),
            'descriptors': x_bp[i].tolist() if i < 3 else None}
           for i, (s, b, r) in enumerate(zip(seqs, bp, rnn))]
(ROOT / 'tests' / 'fixture.json').write_text(json.dumps(fixture))
print(len(fixture), 'rows')
