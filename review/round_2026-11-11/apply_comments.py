"""Apply co-author edits without accepting any previous review markup."""
from pathlib import Path
import re,json,hashlib
ROOT=Path(__file__).resolve().parents[2];ROUND=Path(__file__).resolve().parent;BEFORE=ROUND/'before'
REPO='https://github.com/DrFuzzy/SNFusionCompanion'
def balanced(s,start):
 depth=0
 for i in range(start,len(s)):
  if s[i] in '{}':
   n=0;j=i-1
   while j>=0 and s[j]=='\\':n+=1;j-=1
   if n%2:continue
   depth+=1 if s[i]=='{' else -1
   if depth==0:return i+1
 raise ValueError('Unbalanced '+s[start:start+100])
def cut(s,command,start):
 m=re.search(r'\\'+command+r'\{',s[start:]);a=start+m.start();end=balanced(s,start+m.end()-1);return a,end
records=[]
def edit(rel,old,new,reason):
 p=ROOT/rel;s=p.read_text();assert old in s,(rel,old[:80]);p.write_text(s.replace(old,new,1));records.append(dict(file=rel,before=old,after=new,reason=reason))
# Inventory of executable files actually present in the supplied companion tree.
example_ids={
'ch03_mean_variance_gaussian':('E3-GAUSS','Measurement uncertainty'),
'ch03_ekf_radar_cutin':('E3-EKF','EKF radar cut-in'),
'ch03_ukf_radar_cutin':('E3-UKF','UKF radar cut-in and Monte Carlo rate sweep'),
'ch03_zonotope_radar_cutin':('E3-ZONO','Zonotope radar cut-in'),
'ch03_evidence_occlusion':('E3-EVID','Evidence fusion under occlusion'),
'ch03_incomplete_ahp':('E3-AHP','Incomplete AHP'),
'ch04_hybrid_hetero_kf':('E4-HYBRID','Learned uncertainty and Kalman tracking'),
'ch04_comma2k19_learned_gnss':('E4-GNSS','Learned GNSS error model on comma2k19'),
'ch04_pose_graph_1d':('E4-POSE','One-dimensional pose graph'),
'ch05_camera_radar_ci_fusion':('E5-CI','Camera--radar covariance intersection'),
'ch05_gnss_imu_odometry_ekf':('E5-GNSS','GNSS--IMU--odometry localisation'),
'ch06_distributed_tracking':('E6-TRACK','Distributed tracking'),
'ch06_resilient_consensus':('E6-WMSR','Resilient consensus'),
'ch07_ais_innovation_monitoring':('E7-AIS','AIS--radar innovation monitoring'),
'ch07_cable_das_kf':('E7-CABLE','Idealised cable strain benchmark'),
'ch07_evidence_mcdm':('E7-MCDM','Maritime evidence and prioritisation')}
exercise_ids={
'ch03_hand_computations.py':('P3-HAND','Chapter 3 hand computations'),
'ch03_prob10_ekf_process_noise.py':('P3.10','Process noise and innovation consistency'),
'prob6_miscalibrated_head.py':('P4.6','Miscalibrated uncertainty head'),
'prob7_pose_graph_2d.py':('P4.7','Planar pose graph'),
'prob9_10_comma2k19.py':('P4.9-10','GNSS calibration and temporal correlation'),
'prob04_gnss_outage_variants.py':('P5.4','GNSS outage variants'),
'prob06_ci_sensitivity.py':('P5.6','Covariance-intersection sensitivity'),
'prob05_consensus_rounds.py':('P6.5','Consensus rounds and topology'),
'prob06_packet_loss.py':('P6.6','Packet loss'),
'prob07_wmsr_path.py':('P6.7','W-MSR graph robustness'),
'prob06_spoof_sweep.py':('P7.6','AIS spoofing and window-length sweep'),
'prob07_cable_sweep.py':('P7.7','Cable location and coupling sweep'),
'ch08_p2_ece.py':('P8.2','Expected calibration error'),
'ch08_p3_timestamp_offset.py':('P8.3','Timestamp offset and detection power')}
entries=[];aliases={}
for folder,(ident,title) in example_ids.items():
 ps=[p for p in (ROOT/'companion_repo/examples'/folder).glob('*.py') if not p.name.startswith(('_','download_'))];assert len(ps)==1,folder;p=ps[0];rel=str(p.relative_to(ROOT/'companion_repo'))
 entries.append(dict(id=ident,title=title,path=rel,kind='example'));aliases.update({folder:ident,'companion_repo/examples/'+folder+'/':ident,rel:ident,'companion_repo/'+rel:ident})
for p in sorted((ROOT/'companion_repo/exercises').glob('*/solutions/*.py')):
 if p.name.startswith('_'):continue
 ident,title=exercise_ids[p.name];rel=str(p.relative_to(ROOT/'companion_repo'));entries.append(dict(id=ident,title=title,path=rel,kind='exercise'));aliases.update({p.name:ident,rel:ident,'companion_repo/'+rel:ident})
for ch in range(3,9):
 ident='P'+str(ch);rel=f'exercises/ch{ch:02d}/';entries.append(dict(id=ident,title=f'Chapter {ch} problems and computational solutions',path=rel,kind='collection'))
 for a in [rel,rel+'solutions/','companion_repo/'+rel,'companion_repo/'+rel+'solutions/']:aliases[a]=ident
aliases.update({'companion_repo/examples':'Examples','companion_repo/examples/':'Examples','companion_repo/exercises/chNN':'Exercises',r'companion_repo/\ldots':'Repository','output/ukf_rate_sweep.csv':'D3-UKF'})
def norm(x):return x.replace(r'\allowbreak ','').replace(r'\allowbreak','').replace(r'\_','_').strip()
# Unambiguous explanations in the preface.
oldpref=(ROOT/'frontmatter/preface.tex').read_text().split('\n\n')[-1].rstrip()
newpref=r'''The companion repository, \url{https://github.com/DrFuzzy/SNFusionCompanion}, provides the Python library, worked examples, exercise scripts, and notebooks. The numerical files used to draw the book's figures accompany the LaTeX sources, so readers can compile the book without first running the example programs. The code records simulation settings, random seeds, and evaluation procedures. Examples using external datasets give access instructions and explain the limitations of their reference measurements. The ``Companion Code and Data'' guide at the end of the book gives the exact script locations and data-access instructions; short references in the text link to its entries.

Worked solutions to all 48 problems in Chapters~\ref{sec:classicalSF}--\ref{sec:open-challenges} are collected in ``Solutions to the Problems'' at the end of the book. For design and discussion problems, the proposed solutions illustrate reasonable approaches; other approaches may also satisfy the requirements.'''
edit('frontmatter/preface.tex',oldpref,newpref,'Clarify generated results versus external datasets, name the solutions section, explain open-ended answers, and identify the repository.')
oldurl=r'Companion Python code and notebooks are supplied in the \repopath{companion\_repo} folder of the accompanying source bundle.'
newurl=r'The companion repository is \url{https://github.com/DrFuzzy/SNFusionCompanion}. Short code references link to the ``Companion Code and Data'' guide, which lists exact script locations and data-access instructions.'
edit('main.tex',oldurl,newurl,'Provide the repository URL supplied by the authors.')
macros=r'''
% Short code identifiers link to the complete companion-code directory.
\DeclareRobustCommand{\codeentry}[1]{\hyperref[code:#1]{\texttt{#1}}}
'''
edit('main.tex',r'\newcommand{\repopath}[1]{\texttt{#1}}',r'\newcommand{\repopath}[1]{\texttt{#1}}'+macros,'Use stable, short code identifiers in prose and captions.')
oldscript=r"The script's \texttt{--montecarlo} option exports \repopath{output/ukf\_rate\_sweep.csv}."
newscript=r'The script for this comparison is the UKF radar cut-in script, identified as \codeentry{E3-UKF} in the companion-code guide. Its Monte Carlo option reproduces Table~\ref{tab:ekf-ukf-rate} and saves the rate-sweep results as a CSV file; the command and output file are listed under \codeentry{D3-UKF}.'
edit('chapters/ch03_classical_sensor_fusion.tex',oldscript,newscript,'Identify the script in a full sentence and move the exact command/output path to the code guide.')
oldgraphic=r'\includegraphics[width=0.85\columnwidth]{figures/fusion_pipeline_v02.pdf}'
newgraphic=r'\includegraphics[width=0.68\columnwidth]{figures/fusion_pipeline_v02.pdf}'
edit('chapters/ch02_principles_and_components.tex',oldgraphic,newgraphic,'Reduce Figure 2.1 by 20 percent in both dimensions, preserving aspect ratio.')
for old,new in [(' (Senior Member, IEEE) received',' received'),('~(Senior Member, IEEE) received',' received')]:
 p=ROOT/'backmatter/author_biographies.tex';s=p.read_text()
 if old in s:edit('backmatter/author_biographies.tex',old,new,'Remove IEEE membership designation from author biographies.');s=p.read_text();
 # The first spelling occurs twice; remove both.
 if old in s:edit('backmatter/author_biographies.tex',old,new,'Remove IEEE membership designation from all author biographies.')
# Consistent short references throughout chapters and solutions.
for d in ['chapters','backmatter']:
 for p in sorted((ROOT/d).glob('*.tex')):
  s=p.read_text();hits=[]
  for m in re.finditer(r'\\repopath\{([^{}]*)\}',s):
   key=norm(m[1]);ident=aliases.get(key)
   if ident:hits.append((m.start(),m.end(),m[0],r'\codeentry{'+ident+'}'))
   else:assert key=='*.ipynb',(p,key)
  for a,b,old,new in reversed(hits):s=s[:a]+new+s[b:];records.append(dict(file=str(p.relative_to(ROOT)),before=old,after=new,reason='Replace long path with a linked entry in the companion-code guide.'))
  p.write_text(s)
# Update the introduction's explanation of the reference convention.
p='chapters/ch01_introduction.tex';s=(ROOT/p).read_text();old='whose paths are typeset as '+r'\codeentry{Repository}'+'.';new=r'identified by short references such as \codeentry{E3-UKF}. These references link to the ``Companion Code and Data'' guide, where the exact paths and data-access instructions are collected.'
assert old in s;edit(p,old,new,'Explain the new code-reference convention instead of referring to printed paths.')
# Build the guide from verified local paths; no unverified remote branch/file URLs.
def texpath(x):return x.replace('_',r'\_\allowbreak ').replace('/',r'/\allowbreak ')
guide=r'''\chapter*{Companion Code and Data}
\addcontentsline{toc}{chapter}{Companion Code and Data}
\phantomsection\label{sec:companion-code}
The companion repository is \url{https://github.com/DrFuzzy/SNFusionCompanion}.
The identifiers below are used throughout the text so that full file paths do not interrupt the explanation. Each identifier links to its entry in this guide. Paths are relative to the companion repository root; in the book source bundle this is the \repopath{companion\_repo} folder. Script names and paths below identify the files supplied with this revision. The repository README gives installation and execution instructions.

\section*{Repository and data access}
\phantomsection\label{code:Repository}
\textbf{Repository.} The URL above is the entry point for the code and supporting files.
\phantomsection\label{code:Examples}
\textbf{Examples.} Worked-example scripts are grouped by chapter in \repopath{examples/}.
\phantomsection\label{code:Exercises}
\textbf{Exercises.} Problems, solution scripts, and notebooks are grouped by chapter in \repopath{exercises/}; exact problem statements are supplied as \repopath{problems.tex}. Computational solutions are in each chapter's \repopath{solutions/} subfolder.

The numerical files used by the book's figures are generated results, distinct from the original observations used in real-data experiments. Copies accompany the LaTeX sources in \repopath{data/} and \repopath{figures/}, allowing compilation without running Python. Worked-example programs write generated results to their local \repopath{output/} folders; the export tool copies the files needed by the book into its source tree.

The comma2k19 driving observations are an external dataset, accessed at \url{https://huggingface.co/datasets/commaai/comma2k19} and described in \cite{schafer2018_comma2k19}. Download instructions are provided with \codeentry{E4-GNSS}; the raw-data cache is not part of the book's generated figure data. Run the downloader in that example's folder before the real-data script or Problems~\ref{prob:ch04:9}--\ref{prob:ch04:10}. The reference trajectory shares some GNSS observations with the evaluated measurements; it therefore supports reference-relative diagnostics, with the limitations stated in Chapter~\ref{sec:learning}.

\section*{Worked-example scripts}
'''
for kind,heading in [('example',None),('exercise','Computational exercise scripts'),('collection','Problem collections')]:
 if heading:guide+='\n\\section*{'+heading+'}\n'
 for e in entries:
  if e['kind']!=kind:continue
  assert (ROOT/'companion_repo'/e['path']).exists(),e
  guide+='\n\\par\\medskip\\phantomsection\\label{code:'+e['id']+'}\n\\noindent\\textbf{'+e['id']+' --- '+e['title']+'}\\par\n\\noindent\\repopath{'+texpath(e['path'])+'}\\par\n'
  if e['id']=='E4-GNSS':guide+='Download script: \\repopath{'+texpath('examples/ch04_comma2k19_learned_gnss/download_comma2k19.py')+'}.\\par\n'
guide+=r'''
\section*{UKF rate-sweep command and output}
\phantomsection\label{code:D3-UKF}
\textbf{D3-UKF.} From the repository root, run:
\begin{lstlisting}
python3 examples/ch03_ukf_radar_cutin/ukf_radar_cutin.py --montecarlo
\end{lstlisting}
The command uses the paired seeds and noise convention specified in Example~\ref{ex:ukf-radar-cutin} and saves the rate-sweep table as
\repopath{examples/\allowbreak ch03\_\allowbreak ukf\_\allowbreak radar\_\allowbreak cutin/\allowbreak output/\allowbreak ukf\_\allowbreak rate\_\allowbreak sweep.csv}.
The values are generated numerical results, not an independently collected dataset.

\section*{Notebooks and execution context}
Scripts have matching Jupyter notebooks in the same folders. Open each notebook from its own folder so relative paths resolve correctly. Notebooks are supplied unexecuted; their script source records the settings and seeds used in the examples. The repository README identifies required packages and the public-data download step.
'''
(ROOT/'backmatter/companion_code_guide.tex').write_text(guide)
edit('main.tex',r'\include{backmatter/author_biographies}',r'\include{backmatter/author_biographies}'+'\n'+r'\include{backmatter/companion_code_guide}','Add a central guide containing exact paths and commands.')
# CUMULATIVE COMPARISON: insert around the previous marked source; never regenerate it against a clean baseline.
s=(BEFORE/'main_red_blue.tex').read_text();origin=s
layer=r'''
% Second editorial round: existing DIF commands are retained verbatim.
\newcommand{\RtwoDel}[1]{{\color{red}%
  \renewcommand{\DIFadd}[1]{##1}\renewcommand{\DIFdel}[1]{##1}%
  #1}}
\newcommand{\RtwoAdd}[1]{{\color{blue}#1}}
\newcommand{\RtwoFigure}[1]{\resizebox{0.68\columnwidth}{!}{#1}}
\DeclareRobustCommand{\codeentry}[1]{\hyperref[code:#1]{\texttt{#1}}}
'''
# Record insertions as operations so the exact previous source can be reconstructed.
ops=[]
def insert_pair(a,b,new,why,style='text'):
 global s
 old=s[a:b]
 if style=='text':replacement=r'\RtwoDel{'+old+r'}\RtwoAdd{'+new+'}'
 elif style=='figure':replacement=r'\RtwoFigure{'+old+'}'+r'\par\RtwoAdd{\scriptsize Figure reduced by 20\% in the 11 November revision.}'
 else:raise ValueError(style)
 s=s[:a]+replacement+s[b:];ops.append(dict(before=old,replacement=replacement,reason=why))
# Whole final preface paragraph is balanced, including the previous red/blue wording.
a=s.index(r'\DIFadd{The companion source bundle contains');b=s.index('\n\n\n\\clearpage',a)
insert_pair(a,b,newpref,'Clarify preface while retaining its previous marked paragraph.')
# Locate whole balanced old add macro containing the former Monte Carlo sentence.
needle="The script's }\\texttt{\\DIFadd{--montecarlo}}"
i=s.index(needle);a=s.rfind(r'\DIFadd{',0,i);b=balanced(s,a+len(r'\DIFadd'))
# The existing add macro also contains the preceding rate-experiment explanation.
content=s[a+len(r'\DIFadd{'):b-1];cutpoint=content.index("The script's ")
prefix=content[:cutpoint]
# Leave the previous blue prefix intact, wrapping only its final sentence with the following macros.
# This entire multi-command tail is balanced once the first DIFadd is split explicitly.
start=a;tail_end=balanced(s,s.index(r'\DIFadd{option exports',b)+len(r'\DIFadd'))
oldregion=s[start:tail_end]
# Retain the original complete region in red and add its identical prefix plus new sentence in blue.
insert_pair(start,tail_end,prefix+newscript,'Identify the Monte Carlo script without a long path.')
# Repository footnote: retain original placeholder deletion and earlier added wording.
i=s.index(r'\footnote{\DIFdel{\textbf{[URL of the companion repository');a=i+len(r'\footnote{');b=balanced(s,i+len(r'\footnote'))-1
insert_pair(a,b,newurl,'Add the supplied repository URL while preserving earlier footnote changes.')
# Figure size: keep the previous includegraphics instruction in the source as an argument.
a=s.index(oldgraphic);insert_pair(a,a+len(oldgraphic),'','Reduce Figure 2.1 without displaying duplicate graphics.',style='figure')
# All three biography removals remain visible in red.
for m in list(re.finditer(re.escape('(Senior Member, IEEE)'),s))[::-1]:insert_pair(m.start(),m.end(),'','Remove IEEE membership designation.')
# Replace current code-path uses only; keep earlier deleted paths untouched.
def deletion_ranges(t):
 spans=[]
 for m in re.finditer(r'\\DIFdel(?:FL)?\{',t):spans.append((m.start(),balanced(t,m.end()-1)))
 for m in re.finditer(r'\\RtwoDel\{',t):spans.append((m.start(),balanced(t,m.end()-1)))
 return spans
spans=deletion_ranges(s)
for m in list(re.finditer(r'\\repopath\{([^{}]*)\}',s))[::-1]:
 if any(a<=m.start()<b for a,b in spans):continue
 # Preamble definitions are updated below rather than revised as printed prose.
 if m.start()<s.index(r'\begin{document}'):continue
 key=norm(m[1]);ident=aliases.get(key)
 if ident:insert_pair(m.start(),m.end(),r'\codeentry{'+ident+'}','Use a linked code-guide identifier.')
# Replace the introductory convention as a balanced paragraph, keeping the earlier markup.
a=s.index('Chapters~\\ref{sec:classicalSF} to~\\ref{sec:open-challenges} each end');b=s.index('\n\n',a)
oldregion=s[a:b];newregion=(ROOT/'chapters/ch01_introduction.tex').read_text();newregion=newregion[newregion.index('Chapters~\\ref{sec:classicalSF} to~\\ref{sec:open-challenges} each end'):];newregion=newregion[:newregion.index('\n\n')]
insert_pair(a,b,newregion,'Explain named code entries in the introduction.')
# Add the entirely new guide before the existing printindex.
i=s.rindex(r'\printindex');added='\\clearpage\n\\RtwoAdd{'+guide+'}\n';s=s[:i]+added+s[i:];ops.append(dict(before='',replacement=added,reason='Add the new guide in blue.'))
# Insert layer definitions and an explanatory legend, retaining the original date and legend.
i=s.index(r'\begin{document}');s=s[:i]+layer+s[i:];ops.append(dict(before='',replacement=layer,reason='Define round-two markup.'))
needle=r'\extrachap{Revision comparison: 10 November 2026}'
i=s.index(needle)+len(needle);legend=r'''
\par\medskip\textcolor{blue}{Additional co-author revisions: 11 November 2026.}
Both rounds remain unaccepted and visible. Earlier deletions and additions are retained.
When wording added in the first round is replaced in this round, the complete earlier
marked passage remains visible in red, followed by the new wording in blue.
The previous round's PDF is also preserved unchanged in the source package.
\par\medskip
'''
s=s[:i]+legend+s[i:];ops.append(dict(before='',replacement=legend,reason='Explain the cumulative comparison.'))
# Earlier source strings are not removed: each operation keeps its before text verbatim.
assert all((not o['before']) or o['before'] in o['replacement'] for o in ops)
reconstructed=s
for o in reversed(ops):
 assert o['replacement'] in reconstructed,o['reason']
 reconstructed=reconstructed.replace(o['replacement'],o['before'],1)
assert reconstructed==origin,'Previous marked source was not preserved exactly'
(ROOT/'main_red_blue.tex').write_text(s)
(ROUND/'changes.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
(ROUND/'markup_operations.json').write_text(json.dumps(ops,indent=2,ensure_ascii=False))
(ROUND/'code_entries.json').write_text(json.dumps(entries,indent=2))
(ROUND/'preservation_check.json').write_text(json.dumps(dict(previous_marked_sha256=hashlib.sha256(origin.encode()).hexdigest(),reconstructed_previous_sha256=hashlib.sha256(reconstructed.encode()).hexdigest(),exact_reconstruction=True,operation_count=len(ops)),indent=2))
print('Applied',len(records),'clean edits and',len(ops),'cumulative insertions; previous marked source reconstructed exactly.')
