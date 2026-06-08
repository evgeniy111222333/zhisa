"""Tiny helper script used by the run_pipeline CLI smoke test.

The orchestrator renders YAML ``args: {out: X, text: Y}`` as
``--out X --text Y`` on the command line, so this script reads those
two named flags and writes ``Y`` to file ``X``.
"""
import sys
import pathlib

if __name__ == "__main__":
    argv = sys.argv[1:]
    out = ""
    text = ""
    for i, a in enumerate(argv):
        if a == "--out" and i + 1 < len(argv):
            out = argv[i + 1]
        elif a == "--text" and i + 1 < len(argv):
            text = argv[i + 1]
    pathlib.Path(out).write_text(text)
    print("wrote " + out)
