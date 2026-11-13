#!/usr/bin/env python3
"""Build clean PDF and a red/blue comparison against review/baseline.
Requires latexmk and latexdiff. Run from the book root. Baseline is immutable.
Bibliography bodies are inlined so changes to the printed references are visible.
"""
from pathlib import Path
import re,subprocess,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf';TMP=ROOT/'review/marked_build';OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(exist_ok=True)
if not (TMP/'style').exists(): (TMP/'style').symlink_to('../../style')
def call(args,log,cwd=ROOT):
 with (ROOT/log).open('w') as f:subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
def flat(base,path):
 s=(base/path).read_text()
 s=re.sub(r'(?m)(?<!\\)%.*','',s)
 def inc(m):
  f=m[2];f=f if f.endswith('.tex') else f+'.tex'
  t=flat(base,f)
  return '\\clearpage\n'+t+'\n\\clearpage\n' if m[1]=='include' else t
 s=re.sub(r'\\(input|include)\{([^{}]+)\}',inc,s)
 s=re.sub(r'\\InputIfFileExists\{([^{}]+)\}\{\}\{\}',lambda m:flat(base,m[1]),s)
 s=re.sub(r'\\bibliographystyle\{[^{}]+\}\s*\\bibliography\{[^{}]+\}',lambda m:(base/'main.bbl').read_text(),s)
 # Keep preamble/layout of the final book identical in both comparison inputs.
 return s

def main():
 call(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main'],'review/logs/clean_build_final.txt')
 shutil.copy2(ROOT/'main.pdf',OUT/'sensor_fusion_revised.pdf')
 old=flat(ROOT/'review/baseline','main.tex');new=flat(ROOT,'main.tex')
 pre=new.split('\\begin{document}',1)[0]
 old=pre+'\\begin{document}'+old.split('\\begin{document}',1)[1]
 (TMP/'old.tex').write_text(old);(TMP/'new.tex').write_text(new)
 # Compare one source unit at a time: avoids quadratic full-book token matching.
 def diffunit(a,b,name):
  aa=TMP/(name+'_old.tex');bb=TMP/(name+'_new.tex')
  aa.write_text(pre+'\\begin{document}\n'+a+'\n\\end{document}')
  bb.write_text(pre+'\\begin{document}\n'+b+'\n\\end{document}')
  proc=subprocess.run(['latexdiff','--type=CFONT','--math-markup=whole','--graphics-markup=none','--disable-citation-markup','--config','PICTUREENV=(?:picture|DIFnomarkup|tikzpicture)',str(aa),str(bb)],capture_output=True,text=True,check=True)
  with (ROOT/'review/logs/latexdiff.txt').open('a') as f:f.write(name+'\n'+proc.stderr)
  head,body=proc.stdout.split('\\begin{document}',1)
  return head,body.rsplit('\\end{document}',1)[0]
 body=new.split('\\begin{document}',1)[1]
 # Root structure is stable. Substitute each included unit with its comparison.
 skeleton=(ROOT/'main.tex').read_text().split('\\begin{document}',1)[1]
 skeleton=re.sub(r'(?m)(?<!\\)%.*','',skeleton)
 def unit(m):
  nonlocal pre
  f=m[1]+'.tex'
  if f=='backmatter/solutions.tex':
   t=(ROOT/f).read_text()
   def sol(mm):
    name=Path(mm[1]).stem
    h,b=diffunit((ROOT/'review/baseline'/mm[1]).read_text(),(ROOT/mm[1]).read_text(),name)
    return b
   return '\\clearpage\n'+re.sub(r'\\InputIfFileExists\{([^{}]+)\}\{\}\{\}',sol,t)+'\n\\clearpage\n'
  h,b=diffunit(flat(ROOT/'review/baseline',f),flat(ROOT,f),Path(f).stem)
  return '\\clearpage\n'+b+'\n\\clearpage\n'
 skeleton=re.sub(r'\\include\{([^{}]+)\}',unit,skeleton)
 # Diff each bibliography entry's body, retaining exactly one bibitem and label.
 def bibparts(base):
  t=(base/'main.bbl').read_text();ms=list(re.finditer(r'\\bibitem\{([^{}]+)\}',t));d={}
  for i,m in enumerate(ms):d[m[1]]=t[m.end():ms[i+1].start() if i+1<len(ms) else t.index('\\end{thebibliography}')]
  return t[:ms[0].start()],d
 bh,od=bibparts(ROOT/'review/baseline');_,nd=bibparts(ROOT);bl=[bh]
 for key,b in nd.items():
  if key not in od:bb='\\DIFadd{'+b+'}'
  elif b==od[key]:bb=b
  else:_,bb=diffunit(od[key],b,'bib_'+key.replace(':','_'))
  bl.append('\\bibitem{'+key+'}'+bb)
 bl.append('\\end{thebibliography}')
 skeleton=re.sub(r'\\bibliographystyle\{[^{}]+\}\s*\\bibliography\{[^{}]+\}',lambda m:'\n'.join(bl),skeleton)
 h,_=diffunit('unchanged','unchanged','preamble')
 s=h+'\\begin{document}'+skeleton
 # Colour only: no font change or strikeout; explicit legend in the comparison copy.
 s=s.replace(r'\protect\color{blue} \sf #1',r'\protect\color{blue} #1').replace(r'\protect\color{red} \scriptsize #1',r'\protect\color{red} #1')
 s=s.replace(r'\begin{document}',r'''\begin{document}
\frontmatter
\extrachap{Revision comparison: 10 November 2026}
This comparison uses the supplied clean manuscript as its baseline.
\textcolor{red}{Red text was removed or replaced.}
\textcolor{blue}{Blue text was added.} Black text is retained.
Both old and new passages are printed for review; equation, figure, and page numbers
can therefore differ from the clean revised book. Existing colours in diagrams retain
their original meaning. Computational figures use the revised data; their earlier
numerical results and the code changes are documented in the accompanying review
report and source comparison. Layout-only changes and PDF metadata are not coloured.
The clean revised book is the reading copy.
\clearpage
''',1)
 from finalize_comparison import finalize
 s=finalize(s)
 # Class files/data paths resolve from the book root even though the .tex is elsewhere.
 (TMP/'marked.tex').write_text(s)
 call(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(TMP),str(TMP/'marked.tex')],'review/logs/marked_build.txt')
 shutil.copy2(TMP/'marked.pdf',OUT/'sensor_fusion_red_blue.pdf')
 shutil.copy2(TMP/'marked.tex',ROOT/'main_red_blue.tex')
 print('Built both book versions in',OUT)
if __name__=='__main__':
 if (ROOT/'review/round_2026-11-11/preservation_check.json').exists():
  from build_coauthor_revision import main as cumulative_main
  cumulative_main()
 else:
  main()
