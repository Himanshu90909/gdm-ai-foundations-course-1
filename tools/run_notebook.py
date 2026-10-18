"""Execute a solved notebook locally and report every course feedback check.

Colab-only cells are adapted for a local run: the `!pip install` line is dropped (the course package is put on
PYTHONPATH instead) and `%%capture` is removed so import errors are visible. The notebook that is committed is NOT
modified; only an in-memory copy is executed.
"""
import json, sys, re, copy, pathlib, nbformat
from nbclient import NotebookClient

def prepare(nb):
    nb = copy.deepcopy(nb)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        s = c.source
        s = re.sub(r"^%%capture\s*\n", "", s)
        s = "\n".join(l for l in s.splitlines() if not l.strip().startswith("!pip"))
        c.source = s
    return nb

def main(path, out=None, timeout=1800):
    nb = nbformat.read(path, as_version=4)
    nb = prepare(nb)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True)
    client.execute()
    fails, checks = 0, []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            if o.output_type == "error":
                fails += 1
                print(f"[cell {i}] ERROR {o.ename}: {o.evalue[:200]}")
            txt = o.get("text", "") if o.output_type == "stream" else ""
            if o.output_type in ("display_data", "execute_result"):
                txt += str(o.get("data", {}).get("text/html", ""))[:400] + str(o.get("data", {}).get("text/plain", ""))[:200]
            if "✅" in txt or "Nice" in txt or re.search(r"not correct|Sorry|incorrect|Error:", txt, re.I):
                checks.append((i, txt.strip().replace("\n", " ")[:160]))
    for i, t in checks:
        print(f"[cell {i}] {t}")
    print(f"== {pathlib.Path(path).name}: {fails} cell error(s), {len(checks)} feedback line(s)")
    if out:
        nbformat.write(nb, out)
    return fails

if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None) else 0)
