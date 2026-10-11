#!/usr/bin/env python3
"""Convert a cell-annotated Python script into a Jupyter notebook (nbformat 4), no dependencies.

Cell markers (same convention as VS Code / Spyder / jupytext "percent" format):
    # %%                      -> code cell
    # %% [markdown]           -> markdown cell (each line's leading "# " is stripped)
Lines before the first marker become a leading code cell (module docstring becomes markdown).

usage:  python tools/py2nb.py script.py [out.ipynb]
"""
import json, re, sys, os, uuid


def split_cells(src: str):
    lines = src.splitlines()
    cells = []
    cur_type, cur = "code", []
    marker = re.compile(r"^\s*#\s*%%\s*(\[markdown\])?\s*(.*)$")
    for line in lines:
        m = marker.match(line)
        if m:
            if cur and any(l.strip() for l in cur):
                cells.append((cur_type, cur))
            cur_type = "markdown" if m.group(1) else "code"
            cur = []
            title = m.group(2).strip()
            if title and cur_type == "markdown":
                cur.append("# " + title if not title.startswith("#") else title)
            continue
        cur.append(line)
    if cur and any(l.strip() for l in cur):
        cells.append((cur_type, cur))
    return cells


def make_cell(kind, lines):
    if kind == "markdown":
        text = "\n".join(re.sub(r"^#\s?", "", l) for l in lines).strip("\n")
        return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text}
    text = "\n".join(lines).strip("\n")
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
            "outputs": [], "source": text}


def convert(path, out=None):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    cells = []
    # module docstring -> markdown intro
    m = re.match(r'\s*(?:#![^\n]*\n)?\s*(?:from __future__[^\n]*\n)?\s*("""|\'\'\')(.*?)\1', src, re.S)
    if m:
        cells.append(make_cell("markdown", m.group(2).strip().splitlines()))
        src = src[m.end():]
    for kind, lines in split_cells(src):
        cells.append(make_cell(kind, lines))
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = out or os.path.splitext(path)[0] + ".ipynb"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    print(convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
