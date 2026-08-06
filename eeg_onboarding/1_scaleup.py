"""Scale up: full 9-subject cross-subject classification on BNCI2014_001."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.evaluations import CrossSubjectEvaluation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP

n_subjects = 9
t0 = time.time()

# fetch all 9 subjects
ds = BNCI2014_001()
print(f"[{time.time()-t0:5.0f}s] downloading {n_subjects} subjects...", flush=True)
ds.download(subject_list=ds.subject_list, accept=True)

paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
X, y, meta = paradigm.get_data(ds, subjects=ds.subject_list)
print(f"[{time.time()-t0:5.0f}s] loaded: {X.shape[0]} trials x {X.shape[1]}ch x {X.shape[2]} samp, "
      f"subjects={meta['subject'].nunique()}", flush=True)

pipe = make_pipeline(CSP(n_components=8, log=True), LDA())

# CrossSubjectEvaluation with Leave-One-Subject-Out via CV class
eval_ = CrossSubjectEvaluation(paradigm, datasets=[ds], cv_class=LeaveOneGroupOut)
results = eval_.process({"csp_lda": pipe})
acc = results["score"].mean()
print(f"\n[{time.time()-t0:5.0f}s] cross-subject mean accuracy: {acc*100:.1f}%  (chance 50%)")
print(f"per-subject scores:\n{results[['subject', 'score']].to_string(index=False)}")
