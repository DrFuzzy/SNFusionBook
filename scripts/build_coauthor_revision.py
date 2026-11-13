#!/usr/bin/env python3
"""Build the 11 November co-author revision, retaining the previous marked source.

The cumulative comparison is authored in main_red_blue.tex; do not regenerate it
against a clean baseline, because earlier additions replaced in round two must remain visible.
"""
from pathlib import Path
import shutil,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[1];ROUND=ROOT/'review/round_2026-11-11';OUT=ROOT/'output/pdf'
def verify_preservation():
 t=(ROOT/'main_red_blue.tex').read_text();ops=json.loads((ROUND/'markup_operations.json').read_text())
 for o in reversed(ops):
  assert o['replacement'] in t,o['reason'];t=t.replace(o['replacement'],o['before'],1)
 previous=(ROUND/'before/main_red_blue.tex').read_text();assert t==previous,'Previous review source changed outside recorded insertions'
 info=dict(exact_reconstruction=True,previous_marked_sha256=hashlib.sha256(previous.encode()).hexdigest(),reconstructed_previous_sha256=hashlib.sha256(t.encode()).hexdigest(),operation_count=len(ops))
 (ROUND/'preservation_check.json').write_text(json.dumps(info,indent=2));return info

def main():
 OUT.mkdir(exist_ok=True,parents=True);verify_preservation()
 for source,name in [('main','sensor_fusion_revised_2026-11-11'),('main_red_blue','sensor_fusion_red_blue_cumulative_2026-11-11')]:
  with (ROUND/(source+'_build.log')).open('w') as f:
   subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',source],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
  shutil.copy2(ROOT/(source+'.pdf'),OUT/(name+'.pdf'))
 print('Built clean and cumulative versions; all previous marked source is preserved exactly.')
if __name__=='__main__':main()
