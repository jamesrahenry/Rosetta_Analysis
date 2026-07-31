import json, glob
order = ["A", "G", "B", "H", "D", "C", "E", "F"]
di = {}
for c in ["A", "G", "H", "D", "C", "E", "F"]:
    fs = glob.glob("ccscr_%s/*.json" % c)
    if fs:
        di[c] = json.load(open(fs[0]))
# B from the earlier local run, if mirrored here; else skip (values known from ccscr_B_local)
for bp in ["ccscr_B/crossconcept_scramble.json", "ccscr_B_local/crossconcept_scramble.json"]:
    try:
        di["B"] = json.load(open(bp)); break
    except Exception:
        pass

hdr = "%-2s %5s %6s %6s %6s %6s %6s %6s %7s %7s %7s" % (
    "cl", "d", "same", "cL", "cLscr", "cP", "cPscr", "ovl", "cL/sam", "cP/sam", "cL-scr")
print(hdr)
for c in order:
    if c not in di:
        print("%-2s  (no json — B ran as ccscr_B_local; see prior fold-in)" % c); continue
    d = di[c]
    same = d["same_true"]; cL = d["cross_true_L"]; cLs = d["cross_scr_L"]
    cP = d["cross_true_P"]; cPs = d["cross_scr_P"]; ov = d["overlap_abs"]
    print("%-2s %5s %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %7.3f %7.3f %7.3f" % (
        c, d["d"], same, cL, cLs, cP, cPs, ov, cL / same, cP / same, cL - cLs))
