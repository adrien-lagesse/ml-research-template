% filao.cls -- the filao house style for papers.
%
% A thin, self-contained layer over `article`. It owns LAYOUT ONLY: page
% geometry, section and title styling, a running header, and an optional
% preprint notice. The package stack (math, tables, plots, citations, theorems)
% stays in preamble.tex, which the wrapper \inputs right after this class -- the
% same split the NeurIPS style uses. The house float widths need nothing here:
% \floatcol / \floatfull (macros.tex) alias \columnwidth / \textwidth, and
% this is a single-column class, so both resolve to the one text width.
%
% Options:
%   preprint  (default)  a "Preprint." notice on the first page.
%   final                no notice (camera-ready / submitted elsewhere).
% Any other option is passed through to `article`.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{filao}[2026/07/04 filao house style]

% --- options -------------------------------------------------------------
\newif\iffilao@preprint \filao@preprinttrue
\DeclareOption{final}{\filao@preprintfalse}
\DeclareOption{preprint}{\filao@preprinttrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt]{article}

% --- fonts ---------------------------------------------------------------
% Latin Modern: a crisp Computer Modern with no font download. Math fonts stay
% with the preamble's amssymb/mathtools; this only sharpens the text face.
\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}

% --- page geometry -------------------------------------------------------
% A 5.6in single column on letter: a ~80-character measure, generous margins.
\RequirePackage[
  letterpaper,
  textwidth=5.6in, textheight=8.7in, centering,
  headheight=14pt, headsep=16pt, footskip=28pt,
]{geometry}

% --- accent colour -------------------------------------------------------
% A deep blue for the title rule. Same family as the plots' oiBlue but darker.
% Set it to 0072B2 to match the plot blue exactly.
\RequirePackage{xcolor}
\definecolor{filaoaccent}{HTML}{0B4F6C}

% --- section headings ----------------------------------------------------
\RequirePackage{titlesec}
\titleformat{\section}
  {\normalfont\Large\bfseries}{\thesection}{0.75em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}
  {\normalfont\normalsize\bfseries}{\thesubsubsection}{0.5em}{}
\titlespacing*{\section}     {0pt}{2.4ex plus 1ex minus .2ex}{1.2ex plus .2ex}
\titlespacing*{\subsection}  {0pt}{1.9ex plus .8ex minus .2ex}{0.7ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{1.5ex plus .6ex minus .2ex}{0.6ex plus .2ex}

% --- running header / footer ---------------------------------------------
% Body pages: short title left, page number right, thin rule below.
% \runningtitle{...} sets the left text (empty until then).
\RequirePackage{fancyhdr}
\gdef\filao@runningtitle{}
\newcommand{\runningtitle}[1]{\gdef\filao@runningtitle{#1}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\itshape\filao@runningtitle}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% First page: no header, the preprint notice (if any) at the foot.
\newcommand{\filao@notice}{Preprint.}
\newcommand{\notice}[1]{\renewcommand{\filao@notice}{#1}}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \iffilao@preprint\fancyfoot[L]{\footnotesize\color{gray}\filao@notice}\fi
  \fancyfoot[C]{\small\thepage}%
}

% --- title block ---------------------------------------------------------
% Centred title over an accent rule, then author and date.
\renewcommand{\maketitle}{%
  \thispagestyle{plain}%
  \begin{center}%
    {\LARGE\bfseries\@title\par}%
    \vspace{0.55em}%
    {\color{filaoaccent}\rule{\textwidth}{1pt}}\par
    \vspace{0.7em}%
    {\large\@author\par}%
    \ifx\@date\empty\else\vspace{0.45em}{\small\@date\par}\fi
  \end{center}%
  \vspace{1.2em}%
}

% --- abstract ------------------------------------------------------------
\renewenvironment{abstract}{%
  \begin{center}\small\bfseries Abstract\end{center}%
  \begin{quote}\small\noindent\ignorespaces
}{%
  \end{quote}\vspace{0.6em}%
}
