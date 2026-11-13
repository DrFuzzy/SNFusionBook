from pathlib import Path
import json,zipfile,hashlib
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
files={}
def add(p,rel=None):
 if p.is_file() and p.name!='.DS_Store':files[rel or str(p.relative_to(R))]=p
for name in ['main.tex','main_red_blue.tex','README.md','Makefile','latexmkrc']:add(R/name)
for folder in ['frontmatter','chapters','backmatter','bib','style','figures','data','scripts','companion_repo']:
 for p in (R/folder).rglob('*'):
  if any(x in p.parts for x in ['data_cache','__pycache__','.pytest_cache','.ipynb_checkpoints']):continue
  if p.suffix in ['.aux','.log','.pyc']:continue
  add(p)
for name in ['sensor_fusion_revised_2026-11-11.pdf','sensor_fusion_red_blue_cumulative_2026-11-11.pdf','book_review_and_reassessment.pdf']:add(R/'output/pdf'/name)
for p in D.rglob('*'):
 if p.name=='release_manifest.json' or p.suffix in ['.aux','.pyc'] or '__pycache__' in p.parts:continue
 add(p)
for name in ['audit.html','reference_audit.csv','reference_audit.json','exercise_audit.csv','exercise_audit.json','audit_summary.json','comma2k19_manifest.json','source_changes.patch','release_manifest.json']:
 add(R/'review'/name)
for name in ['tests_final.txt','examples_final.txt','exercise_runs_final.txt']:
 add(R/'review/logs'/name)
manifest=[dict(path=k,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sorted(files.items())]
(D/'release_manifest.json').write_text(json.dumps(manifest,indent=2));add(D/'release_manifest.json')
p=R/'output/sensor_fusion_coauthor_revision_2026-11-11.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for k,v in sorted(files.items()):z.write(v,'sensor_fusion_revision/'+k)
with zipfile.ZipFile(p) as z:
 assert z.testzip() is None
 for item in manifest:assert hashlib.sha256(z.read('sensor_fusion_revision/'+item['path'])).hexdigest()==item['sha256'],item['path']
 for f in ['main.tex','main_red_blue.tex','review/round_2026-11-11/before/main_red_blue.tex','review/round_2026-11-11/preservation_check.json']:assert 'sensor_fusion_revision/'+f in z.namelist()
print('Verified archive:',p,round(p.stat().st_size/1e6,1),'MB;',len(files),'files')
