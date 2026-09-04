# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# ///
import csv
from pathlib import Path

p = "/tmp/opencode/arb/circle_dirty.csv"
with open(p, newline="") as fh:
    reader = csv.reader(fh)
    header = next(reader)
    rows = [r for r in reader if r]
print("header:", header)
print("len(header):", len(header))
print("row0:", rows[0])
print("row1:", rows[1])
print("vidx for x,y:", [header.index(c) for c in ["x", "y"]])
