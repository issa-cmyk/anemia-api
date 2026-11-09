"""python3 -m pytest tests  (downloads the weights on first run)"""
import numpy as np

from anemia_eye import AnemiaEye
from anemia_eye.pipeline import CALIBRATION

eye = AnemiaEye(device='cpu')


def test_score_is_monotonic():
    # A paler conjunctiva (lower a_norm) must never give a lower probability of anemia.
    p = [eye.score(a)[0] for a in np.linspace(-5, 25, 50)]
    assert all(x >= y for x, y in zip(p, p[1:]))


def test_triage_thresholds():
    assert eye.score(30)[1] == 'green'
    assert eye.score(-10)[1] == 'red'
    assert CALIBRATION['t_green'] < CALIBRATION['t_red']


def test_rejects_unusable_images():
    assert not eye.predict(None).ok
    assert not eye.predict(np.zeros((800, 800, 3), np.uint8)).ok                  # too dark
    assert not eye.predict(np.full((100, 100, 3), 128, np.uint8)).ok              # too small
    noise = np.random.default_rng(0).integers(0, 255, (800, 800, 3), dtype=np.uint8)
    assert not eye.predict(noise).ok                                               # no conjunctiva


def test_series_keeps_the_median_photo():
    from anemia_eye import Result
    class Fake(AnemiaEye):
        def __init__(self, results): self.results = iter(results)
        def predict(self, img): return next(self.results)
    mk = lambda a: Result(True, None, eye.score(a)[0], a, eye.score(a)[1], None, {})
    r = Fake([mk(9.0), mk(3.0), mk(6.0)]).predict_series([0, 1, 2])
    assert r.a_norm == 6.0 and r.debug['n_photos'] == 3
    r = Fake([Result(False, 'blurry'), mk(9.0), mk(3.0)]).predict_series([0, 1, 2])
    assert r.a_norm == 3.0                     # two valid photos: the more cautious one
    assert not Fake([Result(False, 'blurry')] * 3).predict_series([0, 1, 2]).ok
