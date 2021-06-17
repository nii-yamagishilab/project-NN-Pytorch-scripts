AMFM_decompy
=============

version 1.0.8.1

This python package provides the tools necessary for decomposing the voiced part
of a speech signal into its modulated components, aka AM-FM decomposition. This
designation is used due the fact that, in this method, the signal is modeled as
a sum of amplitude- and frequency-modulated components.

The goal is to overcome the drawbacks from Fourier-alike techniques, e.g. SFFT,
wavelets, etc, which are limited in the time-frequency analysis by the so-called
Heisenberg-Gabor inequality.

The algorithms here implemented are the QHM (Quasi-Harmonic Model), and its
upgrades, aQHM (adaptive Quasi-Harmonic Model) and eaQHM (extended adaptive
Quasi-Harmonic Model). Their formulation can be found at references [2-4].

Since that the tools mentioned above require a fundamental frequency reference,
the package also includes the pitch tracker YAAPT (Yet Another Algorithm for
Pitch Tracking) [1], which is extremely robust for both high quality and
telephone speech.

The study of AM-FM decomposition algorithms was the theme from my Master Thesis.
The original YAAPT program in MATLAB is provided for free by its authors, while
the QHM algorithms I implemented by myself also in MATLAB. I'm porting them now
to python because:

* the python language is easier to share, read and understand, making it a
  better way to distribute the codes;
* is more resourceful than MATLAB (has different data structures, scripting
  options, etc), which will be useful for me in future studies;
* the computational performance from its numeric and scientific packages (numpy
  and scipy) is equivalent to MATLAB;
* python is free-to-use, while MATLAB is a proprietary software;

Evaluations and future expansions
=============

As for the algorithms computational performance, I optimized the YAAPT code, so
my pyhton version runs now about twice as fast as the original MATLAB one.
However, the QHM algorithms still run as fast as their counterparts in MATLAB.
That's because the main bottleneck of both versions are the matrix dot and
least-squares operations. Since numpy and MATLAB are already optimized to perform
these tasks using internal Fortran functions, as far as I investigated there's
no way to speed them up using Cython, for example. Nevertheless, recently I have
read about numba, which could be applied to improve the AMFM_decompy performance
substantially. Therefore, I may run some tests using it.

In [1] the YAAPT is compared with well-known pitch trackers like the YIN and
the RAPT, and presents the best results. In fact, so far I've been using it,
the algorithm has been proved to be indeed very robust. It must be emphasized
that I merely translated the code, so I only have an average knowledge about
its theoretical formulation. For deep questions concerning it, I would advise
to contact the original authors.

The QHM-like algorithms present some stability problems concerning small
magnitude modulated components, which are already documented at [2,3]. In my
python code I implemented a workaround to this problem, but it is still a
sub-optimal solution.

Actually, I dedicated a chapter in my Master Thesis to a deeper study about
this problem and came up with a better solution. Unfortunately, due stupid
bureaucratic issues, I don't know if and when my work will be defended and
published (to be short, the deadline was expired because me and my advisor
needed more time to correct and improve the thesis text. Then we required a
prorrogation, but the lecturers board declined it. So, basically, I was expelled
from the post-gradute program with a finished and working thesis). Anyway, I'm
still trying to figure out do now with my work and as soon as find a solution,
I'll add my own contributions to this package.

IMPORTANT - Considerations about version 1.0.7 and later versions
=============

In the latest release of the original YAAPT MATLAB source code (YAAPT v4.0)
the default values from the following parameters have been altered:

* `frame_length` parameter changed from 25 ms to 35 ms;
* `nccf_thresh1` parameter changed from 0.25 to 0.3;

Moreover, a new parameter called `frame_lengtht` was added (please pay atention
to the extra "t" at the end), which name is quite similar to `frame_length`.
In order to avoid confusion between them, an alternative (and preferred) alias
for `frame_lengtht` called `tda_frame_length` was used in pYAAPT.py. Nevertheless,
both inputs (`frame_lengtht` and `tda_frame_length`) are accepted.

Due these modifications, if you were running AMFM_decompy 1.0.6 or earlier
versions with their default settings, you may obtain slightly different results
from the ones obtained by running AMFM_decompy 1.0.7 and later versions. with 
the new default parameters.

Therefore, if you really need to obtain exactly the same results from previous
versions, you must provide the old parameter values to the yaapt function. For
example, a 1.0.6 or earlier code like

`pitch = pYAAPT.yaapt(signal)`

should be rewritten in the 1.0.7 and later versions as

`pitch = pYAAPT.yaapt(signal, **{'frame_length': 25.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})`

Installation
=============

The pypi page https://pypi.python.org/pypi/AMFM_decompy/1.0.8 is recommended for
a quick installation. But you can also copy all directories here and then run

```python setup.py install```

in command line. After that, run the test script by typing

`AMFM_test.py`

to check if everything is ok (it can take couple of minutes to calculate the
results). This script is a example about how to use the package.

I've tested the installation script and the package itself in Linux and Windows
systems (but not in iOS) and everything went fine. So, if a problem comes up,
it must be probably something about python not finding the files paths.

How to use
=============

Check the AMFM_decompy pdf documentation included in the docs folder or the
online documentation at http://bjbschmitt.github.io/AMFM_decompy. The amfm_decompy
folder contains the sample.wav file that is used to ilustrate the package's code
examples.

Credits and Publications
=============

The original MATLAB YAAPT program was written by Hongbing Hu and Stephen
A.Zahorian from the Speech Communication Laboratory of the State University of
New York at Binghamton.

It is available at http://www.ws.binghamton.edu/zahorian as free software.
Further information about the program can be found at

   [1] Stephen A. Zahorian, and Hongbing Hu, "A spectral/temporal method for robust
       fundamental frequency tracking," J. Acosut. Soc. Am. 123(6), June 2008.

The QHM algorithm and its upgrades are formulated and presented in the following publications:

   [2] Y. Pantazis, , PhD Thesis, University of Creta, 2010.

   [3] Y. Pantazis, O. Rosec and Y. Stylianou, , IEEE Transactions on Audio, Speech and
       Language Processing, vol. 19, n 2, 2011.

   [4] G. P. Kafentzis, Y. Pantazis, O. Rosec and Y. Stylianou, , in IEEE International Conference on Acoustics,
       Speech and Signal Processing (ICASSP), 2012.

Copyright and contact
=============

The AMFM_decompy is free to use, share and modify under the terms of the MIT
license.

Questions, comments, suggestions, and contributions are welcome. Please contact
me at

bernardo.jb.schmitt@gmail.com.
