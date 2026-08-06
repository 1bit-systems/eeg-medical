"""List public S3 objects under an HBN-EEG release (keyless, plain requests)."""
import requests, sys, urllib.parse
import xml.etree.ElementTree as ET

NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
BUCKET = "https://fcp-indi.s3.amazonaws.com"
PREFIX = "data/Projects/HBN/BIDS_EEG/cmi_bids_R1/"


def list_prefix(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": 2, "prefix": prefix, "max-keys": 1000}
        if token:
            q["continuation-token"] = token
        u = BUCKET + "/?" + urllib.parse.urlencode(q)
        r = requests.get(u, timeout=60)
        root = ET.fromstring(r.content)
        for c in root.findall(".//s:Contents", NS):
            k = c.find("s:Key", NS).text
            if not k.endswith("/"):
                keys.append(k)
        nxt = root.find(".//s:NextContinuationToken", NS)
        if nxt is None and len(root.findall(".//s:Contents", NS)) < 1000:
            break
        token = nxt.text if nxt is not None else None
        if token is None:
            break
    return keys


if __name__ == "__main__":
    keys = list_prefix(PREFIX)
    import collections
    # summarize by extension and top-level
    by_ext = collections.Counter(k.split(".")[-1] for k in keys)
    by_top = collections.Counter(k.split("/")[2] for k in keys if len(k.split("/")) > 2)
    print(f"TOTAL OBJECTS: {len(keys)}")
    print("BY EXTENSION:", dict(by_ext))
    print("BY TOP-LEVEL (subdir):", dict(by_top.most_common(10)))
    print("\nSAMPLE PATHS:")
    for k in keys[:8]:
        print("  ", k)
