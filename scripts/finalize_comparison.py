"""Make bibliography removals visible, including commands latexdiff suppresses."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
def parts(p):
 s=p.read_text();m=list(re.finditer(r'\\bibitem\{([^{}]+)\}',s));d={}
 for i,x in enumerate(m):d[x[1]]=s[x.end():m[i+1].start() if i+1<len(m) else s.index(r'\end{thebibliography}')]
 return s[:m[0].start()],d

def finalize(s):
 h,old=parts(ROOT/'review/baseline/main.bbl');_,new=parts(ROOT/'main.bbl');out=[h]
 for k,b in new.items():
  out.append(r'\bibitem{'+k+'}')
  if k not in old:out.append(r'{\color{blue} '+b+'\par}')
  elif b!=old[k]:out.extend([r'{\color{red} '+old[k]+'\par}',r'\smallskip{\color{blue} '+b+'\par}'])
  else:out.append(b)
 out.append(r'\end{thebibliography}')
 a=s.index(r'\begin{thebibliography}');e=s.index(r'\end{thebibliography}',a)+len(r'\end{thebibliography}')
 s=s[:a]+'\n'.join(out)+s[e:]
 s=s.replace(r'\DIFdelFL{Ultrasonic }',r'\DIFdelFL{Ultra-}\allowbreak\DIFdelFL{sonic }')
 s=s.replace(r'\DIFdel{\small\setlength',r'\DIFdel{\setlength')
 s=s.replace(r'\DIFdel{$\mathbf{w}=[\,0.376\;\;0.121\;\;0.215\;\;0.215\;\;0.074\,]^{\mathsf T}$ }',r'\[\DIFdel{\mathbf{w}=[\,0.376\;\;0.121\;\;0.215\;\;0.215\;\;0.074\,]^{\mathsf T}}\]')
 s=s.replace(r'\chapter*{Solutions to the Problems}',r'\chapter*{Solutions to \DIFdel{Selected }\DIFadd{the }Problems}')
 # Expand changed prose macro at its printed use; the final preamble remains clean.
 b=s.index(r'\begin{document}');preamble=s[:b];body=s[b:]
 body=body.replace(r'\repourl',r'\DIFdel{\textbf{[URL of the companion repository, to be inserted]}}\DIFadd{Companion Python code and notebooks are supplied in the \repopath{companion\_repo} folder of the accompanying source bundle.}')
 s=preamble+body
 s=s.replace(r'\begin{document}',r'''\pdfstringdefDisableCommands{\def\DIFadd#1{#1}\def\DIFdel#1{}}
\begin{document}''',1)
 s=s.replace('Layout-only changes and PDF metadata are not coloured.','Changed bibliography records are printed twice: the supplied record in red, then the revised record in blue, so that removed DOI and URL commands remain visible. Layout-only changes and PDF metadata are not coloured.')
 return s
if __name__=='__main__':
 p=ROOT/'review/marked_build/marked.tex';p.write_text(finalize(p.read_text()))
