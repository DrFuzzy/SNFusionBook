from pathlib import Path
import json,re,sys,hashlib,subprocess,concurrent.futures
from pypdf import PdfReader
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'scripts'))
from build_coauthor_revision import verify_preservation
checks={'preservation':verify_preservation()};PDFS={'clean':'sensor_fusion_revised_2026-11-11.pdf','cumulative':'sensor_fusion_red_blue_cumulative_2026-11-11.pdf'}
for key,name in PDFS.items():
 p=R/'output/pdf'/name;reader=PdfReader(p);texts=[x.extract_text() or '' for x in reader.pages];whole='\n'.join(texts)
 assert '??' not in whole,(key,'Unresolved reference')
 assert 'https://github.com/DrFuzzy/SNFusionCompanion' in whole.replace('\n',''),(key,'Repository URL')
 for token in ['Companion Code and Data','E3-UKF','D3-UKF','P8.3','Monte Carlo option']:
  assert token in whole,(key,token)
 if key=='clean':
  assert 'Senior Member, IEEE' not in whole
  assert 'appear in the back matter' not in whole
  assert 'Generated numerical data are also shipped' not in whole
 else:
  for token in ['Senior Member, IEEE','Generated numerical data are also shipped','10.1109/7.705920','10.1109/7.532261','Additional co-author revisions']:
   assert token in whole,(key,'Lost old or new wording: '+token)
  stream=b''.join(x.get_contents().get_data() for x in reader.pages if x.get_contents() is not None)
  assert b'1 0 0 rg' in stream and b'0 0 1 rg' in stream
 loc={}
 for token in ['A typical sensor fusion pipeline','The script for this comparison is','Worked solutions to all 48','Senior Member, IEEE','Companion Code and Data','UKF rate-sweep command and output']:
  loc[token]=[i+1 for i,t in enumerate(texts) if token in t]
 checks[key]=dict(file=name,pages=len(reader.pages),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),locations=loc)
for name in ['main.log','main_red_blue.log']:
 s=(R/name).read_text(errors='replace');assert 'Overfull' not in s,name;assert 'Missing character' not in s,name
 assert not re.search(r'Warning: (?:Reference|Citation).*undefined',s),name
 checks[name]=dict(overfull_boxes=0,missing_glyph_warnings=0,undefined_references=0)
# Every linked identifier has exactly one guide target and all listed source paths exist.
s=(R/'backmatter/companion_code_guide.tex').read_text();labels=re.findall(r'\\label\{code:([^}]+)\}',s);assert len(labels)==len(set(labels))
for folder in ['frontmatter','chapters','backmatter']:
 for p in (R/folder).glob('*.tex'):
  for ident in re.findall(r'\\codeentry\{([^}]+)\}',p.read_text()):assert ident in labels,(p,ident)
for e in json.loads((D/'code_entries.json').read_text()):assert (R/'companion_repo'/e['path']).exists(),e
checks['code_links']=dict(unique_targets=len(labels),all_references_resolved=True,all_local_paths_exist=True)
# Keep the 10 November PDFs unchanged, byte-for-byte.
for name in ['sensor_fusion_revised.pdf','sensor_fusion_red_blue.pdf']:
 assert (R/'output/pdf'/name).read_bytes()==(D/'before/output/pdf'/name).read_bytes(),name
checks['prior_pdfs_unchanged']=True
(D/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2),flush=True)
if '--no-render' in sys.argv: sys.exit(0)
QA=R/'tmp/pdfs/coauthor_2026-11-11_final';QA.mkdir(parents=True,exist_ok=True)
def render(item):
 key,name=item;dest=QA/key;dest.mkdir(exist_ok=True)
 p=subprocess.run(['pdftoppm','-scale-to','750','-png',str(R/'output/pdf'/name),str(dest/'page')],capture_output=True)
 assert p.returncode==0,p.stderr[-1000:]
 paths=sorted(dest.glob('page-*.png'))
 assert len(paths)==checks[key]['pages']
 for start in range(0,len(paths),30):
  batch=paths[start:start+30];cols=6;w=220;h=305;sheet=Image.new('RGB',(cols*w,((len(batch)+cols-1)//cols)*h),'#dbe0e6');draw=ImageDraw.Draw(sheet)
  for j,p in enumerate(batch):
   im=Image.open(p).convert('RGB');im.thumbnail((w-8,h-20));x=(j%cols)*w+(w-im.width)//2;y=(j//cols)*h+18;sheet.paste(im,(x,y));draw.text(((j%cols)*w+5,(j//cols)*h+2),p.stem,fill='black')
  sheet.save(QA/(key+'_sheet_'+str(start//30+1).zfill(2)+'.jpg'),quality=90)
 print('Rendered',key,len(paths),'pages',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:list(e.map(render,PDFS.items()))
