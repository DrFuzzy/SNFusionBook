#!/usr/bin/env python3
"""Assemble the standalone LaTeX example, embedding the CSV via filecontents*."""

csv_text = open("ekf_radar_cutin.csv").read().rstrip("\n")

TEX = r"""%% ===================================================================
%%  Standalone example: EKF tracking of a cut-in vehicle from radar
%%  -----------------------------------------------------------------
%%  Compiles on its own with pdflatex (no external files needed: the
%%  simulation output is embedded below via filecontents*).
%%
%%  TO TRANSPLANT INTO THE SURVEY (main.tex / Overleaf):
%%    1. add to the preamble:      \usepackage{pgfplots}
%%                                 \usepgfplotslibrary{groupplots}
%%                                 \pgfplotsset{compat=1.18}
%%       (tikz and xcolor are already loaded there; \newtheorem{example}
%%        is already declared there too, so do NOT copy it again)
%%    2. copy the colour definitions marked  [PREAMBLE]  below;
%%    3. copy the filecontents* block below to just after \begin{document}
%%       -- or keep ekf_radar_cutin.csv as a separate uploaded file and
%%       delete the block entirely, the figure reads it either way;
%%    4. copy the block marked  [BODY]  into Sec. III (Bayesian Filtering),
%%       immediately after the NIS discussion.
%%
%%  The data are produced by ekf_radar_cutin.py (numpy only, fixed seed
%%  20260727, so the figure is reproducible bit for bit).
%% ===================================================================

\begin{filecontents*}[overwrite]{ekf_radar_cutin.csv}
__CSV__
\end{filecontents*}

\documentclass[10pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,amssymb,amsthm,bm}
\usepackage[usenames,dvipsnames]{xcolor}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}

%% ---- [PREAMBLE] colours used by the figure -------------------------
\definecolor{EkfBlue}{RGB}{31,86,158}
\definecolor{TruthBlack}{RGB}{25,25,25}
\definecolor{MeasGrey}{RGB}{160,160,160}
\definecolor{BandGrey}{RGB}{225,225,225}
\definecolor{ManoeuvreOrange}{RGB}{244,177,131}

%% ---- present in the survey preamble already; needed here to compile alone
\newtheorem{example}{Example}

%% ---- [DELETE WHEN TRANSPLANTING] -----------------------------------
%% The example cross-references four equations that live in the survey's
%% Bayesian-filtering subsection.  Compiling this file on its own would print
%% "??" for them, so stub labels are declared here.  Remove this whole block
%% when you paste the example into main.tex: the real labels are already there
%% and LaTeX would then report "Label multiply defined".
\makeatletter
\newlabel{eq:kf_linear_process}{{7a}{1}{}{}{}}
\newlabel{eq:kf_innovation}{{11}{1}{}{}{}}
\newlabel{eq:kf_gain}{{13}{1}{}{}{}}
\newlabel{eq:nis}{{22}{1}{}{}{}}
\makeatother
%% ---- [END OF BLOCK TO DELETE] --------------------------------------

\begin{document}

%% ===================================================================
%% [BODY] -- everything from here to the end of the figure goes in the paper
%% ===================================================================

\begin{example}[Extended Kalman filtering of an automotive-radar cut-in]
\label{ex:ekf-radar-cutin}
The classical illustration of Kalman filtering tracks an aircraft from radar
returns, with the radar reporting Cartesian coordinates so that the measurement
model is linear. An automotive radar does not behave that way: it reports
\emph{range}, \emph{azimuth}, and \emph{range rate}, and the resulting
measurement model is nonlinear in a Cartesian target state. The same tracking
problem therefore becomes an EKF problem, which makes it a compact vehicle for
the material of this subsection.

Consider an ego vehicle travelling at $27$\,m/s and a target vehicle travelling
at $22$\,m/s in the adjacent lane, initially $60$\,m ahead. All quantities are
expressed in the ego-vehicle frame, with $x$ forward and $y$ to the left and the
origin at the front-bumper radar. The state
\begin{equation}
\mathbf{x}_k=
\begin{bmatrix}
p_{x,k} & p_{y,k} & v_{x,k} & v_{y,k}
\end{bmatrix}^{\mathsf T}
\end{equation}
collects the relative position and velocity of the target, and is propagated by
the constant-velocity model of the form
\eqref{eq:kf_linear_process}, with
\begin{equation}
\mathbf{F}=
\begin{bmatrix}
1&0&T&0\\
0&1&0&T\\
0&0&1&0\\
0&0&0&1
\end{bmatrix},
\qquad T=0.05\ \mathrm{s},
\end{equation}
and a discretised white-noise-acceleration process covariance
\begin{equation}
\mathbf{Q}=\sigma_a^{2}
\begin{bmatrix}
\tfrac{T^{4}}{4}&0&\tfrac{T^{3}}{2}&0\\[2pt]
0&\tfrac{T^{4}}{4}&0&\tfrac{T^{3}}{2}\\[2pt]
\tfrac{T^{3}}{2}&0&T^{2}&0\\[2pt]
0&\tfrac{T^{3}}{2}&0&T^{2}
\end{bmatrix},
\qquad \sigma_a=0.8\ \mathrm{m/s^{2}},
\end{equation}
in which $\sigma_a$ expresses how much unmodelled acceleration the target is
permitted between updates. The radar observes
\begin{equation}
\mathbf{g}(\mathbf{x}_k)=
\begin{bmatrix}
\sqrt{p_x^{2}+p_y^{2}}\\[4pt]
\arctan\!\left(p_y/p_x\right)\\[4pt]
\dfrac{p_xv_x+p_yv_y}{\sqrt{p_x^{2}+p_y^{2}}}
\end{bmatrix},
\qquad
\mathbf{R}=\operatorname{diag}\!\left(\sigma_r^{2},\sigma_\theta^{2},\sigma_{\dot r}^{2}\right),
\label{eq:ekf-radar-meas}
\end{equation}
with $\sigma_r=0.30$\,m, $\sigma_\theta=0.5^{\circ}$, and
$\sigma_{\dot r}=0.10$\,m/s, values representative of an automotive
$77$\,GHz radar. Writing $d=\sqrt{p_x^{2}+p_y^{2}}$, the Jacobian required by
the EKF is
\begin{equation}
\mathbf{H}_k=
\begin{bmatrix}
\dfrac{p_x}{d} & \dfrac{p_y}{d} & 0 & 0\\[8pt]
-\dfrac{p_y}{d^{2}} & \dfrac{p_x}{d^{2}} & 0 & 0\\[8pt]
\dfrac{p_y(v_xp_y-v_yp_x)}{d^{3}} &
\dfrac{p_x(v_yp_x-v_xp_y)}{d^{3}} &
\dfrac{p_x}{d} & \dfrac{p_y}{d}
\end{bmatrix},
\label{eq:ekf-radar-jacobian}
\end{equation}
evaluated at $\widehat{\mathbf{x}}_{k|k-1}$. Three practical points are worth
noting, because each of them is a common source of divergence. First,
\eqref{eq:ekf-radar-jacobian} is singular at $d=0$, so a target crossing the
sensor origin must be handled separately. Second, the azimuth component of the
innovation \eqref{eq:kf_innovation} must be wrapped to $(-\pi,\pi]$ before it is
used, since the difference of two angles is not the angle of the difference.
Third, the track is initialised directly from the first detection,
$\widehat{p}_x=r\cos\theta$ and $\widehat{p}_y=r\sin\theta$, with the velocity
components set to zero and given a deliberately large prior variance; the
azimuth channel alone cannot resolve the velocity, and pretending otherwise
produces an overconfident initial covariance.

The scenario contains one event the model does not: between $t=5$\,s and
$t=8$\,s the target changes lane into the path of the ego vehicle, moving
laterally from $3.5$\,m to $0$\,m. Fig.~\ref{fig:ekf-radar-cutin} shows the
outcome over a $10$\,s run at $20$\,Hz. In the upper panel the raw detections,
converted to Cartesian coordinates for display, scatter visibly about the true
trajectory, chiefly in the lateral direction: at $60$\,m an azimuth error of
$0.5^{\circ}$ is already a lateral error of $0.5$\,m, which is why a bearing-only
lateral estimate degrades with range while the range estimate does not. Over the
run the EKF reduces the position RMSE from $0.42$\,m to $0.16$\,m, a reduction of
$62$\,\%, and it does so while also estimating the two velocity components, which
the radar never observes directly.

The lower panel shows why this example is placed after the discussion of filter
consistency. It plots the normalised innovation squared \eqref{eq:nis} against
the two-sided $95\,\%$ interval of the $\chi^{2}_{3}$ distribution. While the
constant-velocity assumption holds, the filter is statistically consistent: the
mean NIS is $2.57$ against an expected value of $3$, and $98.8\,\%$ of the
samples fall inside the interval. During the lane change the picture changes.
The mean NIS rises to $3.73$, only $90.2\,\%$ of the samples remain inside the
interval, and the peak reaches $13.5$. Nothing is wrong with the sensor; what
the diagnostic has detected is that the process model no longer describes the
target. The estimate degrades accordingly but gracefully, with a peak lateral
error of $0.40$\,m and a lateral velocity estimate that lags the true one by
roughly a factor of two at the middle of the manoeuvre, which is the
characteristic signature of a constant-velocity filter tracking a manoeuvring
object.

This is the practical lesson the example is meant to carry. The EKF is not
failing here, and enlarging $\mathbf{Q}$ until the NIS looks acceptable would
only trade the lag for a noisier estimate in the much longer intervals during
which the target is not manoeuvring. The structural answer is to stop insisting
on a single motion model, which is precisely the argument for the interacting
multiple-model filters mentioned earlier in this subsection, and for treating
the innovation sequence as a monitored signal in its own right rather than as an
intermediate quantity.
\end{example}

\begin{figure}[!t]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={group size=1 by 2, vertical sep=17mm},
    width=0.95\columnwidth,
    scale only axis,
    tick label style={font=\scriptsize},
    label style={font=\scriptsize},
    legend style={font=\scriptsize, draw=none, fill=white, fill opacity=0.85,
                  text opacity=1},
    grid=major,
    grid style={line width=0.2pt, draw=gray!25},
    axis line style={line width=0.5pt},
    tick align=outside,
    title style={font=\scriptsize, yshift=0.5mm},
]

%% ---------------- panel (a): bird's-eye view ----------------
\nextgroupplot[
    height=3.3cm,
    xlabel={longitudinal range $p_x$ [m]},
    ylabel={lateral offset $p_y$ [m]},
    xmin=8, xmax=63,
    ymin=-1.4, ymax=6.4,
    ytick={0,1,2,3,4,5},
    legend pos=north west,
    legend cell align=left,
    legend columns=3,
    title={(a) bird's-eye view in the ego-vehicle frame},
]
% lane divider the target crosses (ego lane centred on y = 0, adjacent lane on y = 3.5)
\addplot[draw=gray!55, dashed, line width=0.4pt, forget plot]
    coordinates {(8,1.75) (63,1.75)};

\addplot[only marks, mark=*, mark size=0.5pt, draw=MeasGrey, fill=MeasGrey]
    table[x=px_meas, y=py_meas, col sep=comma] {ekf_radar_cutin.csv};
\addlegendentry{radar detections}

\addplot[TruthBlack, line width=1.0pt]
    table[x=px_true, y=py_true, col sep=comma] {ekf_radar_cutin.csv};
\addlegendentry{ground truth}

\addplot[EkfBlue, line width=0.9pt, densely dashed]
    table[x=px_est, y=py_est, col sep=comma] {ekf_radar_cutin.csv};
\addlegendentry{EKF}

% the target closes on the ego vehicle, so time runs right to left
\draw[->, >=stealth, gray!70, line width=0.5pt]
    (axis cs:52,-0.75) -- (axis cs:20,-0.75);
\node[anchor=south, font=\scriptsize, text=black!60] at (axis cs:36,-0.9)
    {closing: $t=0$ at $p_x=60$\,m};

%% ---------------- panel (b): NIS consistency ----------------
\nextgroupplot[
    height=3.1cm,
    xlabel={time $t$ [s]},
    ylabel={NIS $d_k^{2}$},
    xmin=0, xmax=10,
    ymin=0, ymax=15,
    ytick={0,3,6,9,12,15},
    title={(b) innovation consistency against the $\chi^{2}_{3}$ interval},
]
% two-sided 95% interval of chi^2_3: [0.216, 9.348]  (drawn first, so the
% manoeuvre band below remains visible across the full height of the axis)
\addplot[draw=none, fill=BandGrey, forget plot]
    coordinates {(0,0.216) (10,0.216) (10,9.348) (0,9.348)} \closedcycle;
% lane-change window
\addplot[draw=none, fill=ManoeuvreOrange, fill opacity=0.45, forget plot]
    coordinates {(5,0) (8,0) (8,15) (5,15)} \closedcycle;
% expected value E[d^2] = 3
\addplot[draw=gray!70, line width=0.5pt, densely dotted, forget plot]
    coordinates {(0,3) (10,3)};

\addplot[EkfBlue, line width=0.5pt]
    table[x=t, y=nis, col sep=comma] {ekf_radar_cutin.csv};

\node[anchor=south, font=\scriptsize, text=black!70] at (axis cs:6.5,13.2) {lane change};
\node[anchor=west, font=\scriptsize, text=black!55] at (axis cs:0.2,10.4) {$95\,\%$ interval};

\end{groupplot}
\end{tikzpicture}
\caption{Extended Kalman filtering of a cut-in vehicle from automotive-radar
range, azimuth, and range-rate detections (Example~\ref{ex:ekf-radar-cutin}).
(a)~Raw detections converted to Cartesian coordinates, ground truth, and EKF
estimate. The target closes on the ego vehicle, so time runs from right to left;
the excursion at the right-hand edge is the initialisation transient, before the
velocity components have been observed. The detection scatter is dominated by
the azimuth channel and therefore grows with range. (b)~Normalised innovation
squared against the two-sided $95\,\%$ interval of $\chi^{2}_{3}$. The filter is
consistent while the constant-velocity assumption holds and becomes measurably
inconsistent during the lane change (shaded), which is the intended behaviour of
the diagnostic: it reports a model mismatch, not a sensor fault.}
\label{fig:ekf-radar-cutin}
\end{figure}

%% ===================================================================
%% [END OF BODY]
%% ===================================================================

\end{document}
"""

open("ekf_radar_cutin_example.tex", "w").write(TEX.replace("__CSV__", csv_text))
print("wrote ekf_radar_cutin_example.tex")
