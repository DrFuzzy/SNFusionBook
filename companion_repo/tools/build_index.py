#!/usr/bin/env python3
"""Regenerate the per-chapter example and exercise tables of README.md from the folder contents.

Examples:  every examples/<chNN>_<slug>/<script>.py (first docstring line = description)
Exercises: every exercises/<chNN>/solutions/*.py and the first heading of exercises/<chNN>/README.md

usage: python3 tools/build_index.py          (rewrites the blocks between the INDEX markers in README.md)
"""
import glob, os, re, ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER_TITLES = {
    "ch03": "Classical sensor fusion",
    "ch04": "Learning-based and hybrid fusion",
    "ch05": "Autonomous driving application",
    "ch06": "Multi-agent sensor fusion",
    "ch07": "Applications beyond autonomous vehicles",
    "ch08": "Open challenges and future directions",
}


def first_doc_line(path):
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        doc = ast.get_docstring(tree) or ""
        line = doc.strip().splitlines()[0] if doc.strip() else ""
        return line.strip()
    except Exception:
        return ""


def examples_table():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "examples", "ch*_*"))):
        ch = os.path.basename(d)[:4]
        if ch == "ch03":
            continue  # chapter 3 has its own detailed table
        scripts = sorted(glob.glob(os.path.join(d, "*.py")))
        for s in scripts:
            nb = os.path.splitext(s)[0] + ".ipynb"
            rows.append("| %s | `%s/%s` | %s | %s |" % (
                ch[2:].lstrip("0"), os.path.relpath(d, ROOT), os.path.basename(s),
                first_doc_line(s).replace("|", "/"),
                "`%s`" % os.path.basename(nb) if os.path.exists(nb) else "–"))
    header = "| Chapter | Script (run from its folder) | What it reproduces | Notebook |\n|---|---|---|---|\n"
    return header + "\n".join(rows) + "\n"


def exercises_table():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "exercises", "ch*"))):
        ch = os.path.basename(d)
        readme = os.path.join(d, "README.md")
        topics = ""
        if os.path.exists(readme):
            with open(readme, encoding="utf-8") as f:
                txt = f.read()
            probs = re.findall(r"^#{2,3}\s*(?:Problem|Exercise)\s*([\d.]+)[^\n]*", txt, re.M)
            topics = "%d problem statements" % len(probs) if probs else "see README"
        sols = sorted(glob.glob(os.path.join(d, "solutions", "*.py")))
        scripts = ", ".join("`%s`" % os.path.basename(s) for s in sols) or "–"
        rows.append("| %s — %s | `%s/` | %s | %s |" % (
            ch[2:].lstrip("0"), CHAPTER_TITLES.get(ch, ""), os.path.relpath(d, ROOT), topics, scripts))
    header = "| Chapter | Folder | Problem statements | Solution scripts (each with a notebook) |\n|---|---|---|---|\n"
    return header + "\n".join(rows) + "\n"


def replace_block(text, name, body):
    start, end = "<!-- %s:start -->" % name, "<!-- %s:end -->" % name
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        return pre + start + "\n" + body + end + post
    return None


if __name__ == "__main__":
    readme = os.path.join(ROOT, "README.md")
    with open(readme, encoding="utf-8") as f:
        text = f.read()
    for name, body in (("examples-index", examples_table()), ("exercises-index", exercises_table())):
        new = replace_block(text, name, body)
        if new is None:
            print("marker block %s not found in README.md; printing table instead:\n%s" % (name, body))
        else:
            text = new
    with open(readme, "w", encoding="utf-8") as f:
        f.write(text)
    print("README.md updated")
