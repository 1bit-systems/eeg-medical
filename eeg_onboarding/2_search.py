"""Catalog everything MOABB knows, filter by paradigm."""
import warnings
warnings.filterwarnings("ignore")
from moabb.datasets.utils import dataset_search

cutoffs = {"imagery": 40, "p300": 40, "ssvep": 25}
for paradigm, cutoff in cutoffs.items():
    res = dataset_search(paradigm=paradigm, min_subjects=5)
    print(f"\n===== {paradigm.upper()} : {len(res)} datasets, filter min_subjects>=5 -> {len(res)} shown =====")
    # res is a list of configured dataset instances; show code + subjects
    for ds in res[:cutoff]:
        nsubj = len(getattr(ds, "subject_list", []) or [])
        nch = getattr(ds, "n_channels", "?")
        try:
            freqs = ds.interval
        except Exception:
            freqs = "?"
        print(f"  {type(ds).__name__:28s} subj={nsubj:>4}  ch={nch:>3}  interval={freqs}")
