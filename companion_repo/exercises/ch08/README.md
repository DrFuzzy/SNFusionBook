# Chapter 8 exercises

The complete revised problem statements, with the same numbering as the book, are in `problems.tex` (a LaTeX excerpt using the book preamble). Worked answers are in the book's **Solutions to the Problems**. Analytical and design exercises are intended to be solved by hand or discussed; the following computational scripts have matching generated notebooks.

- `solutions/ch08_p2_ece.py`
- `solutions/ch08_p3_timestamp_offset.py`

Run a script from this chapter folder, for example `python3 solutions/ch08_p2_ece.py`. Use Python 3.9+ with NumPy, SciPy and Matplotlib (pandas is needed for the public-driving-data example).

For Problem 8.3, 15.1 ms and 113.3 ms are mean-crossing offsets. At 95% detection power the offsets are 24.8 ms and 185.9 ms. These criteria must not be conflated. For Problem 8.4, unlabelled recordings screen for failures; they cannot certify transfer.
