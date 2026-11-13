# Chapter 6 exercises

The complete revised problem statements, with the same numbering as the book, are in `problems.tex` (a LaTeX excerpt using the book preamble). Worked answers are in the book's **Solutions to the Problems**. Analytical and design exercises are intended to be solved by hand or discussed; the following computational scripts have matching generated notebooks.

- `solutions/prob05_consensus_rounds.py`
- `solutions/prob06_packet_loss.py`
- `solutions/prob07_wmsr_path.py`

Run a script from this chapter folder, for example `python3 solutions/prob05_consensus_rounds.py`. Use Python 3.9+ with NumPy, SciPy and Matplotlib (pandas is needed for the public-driving-data example).

For Problem 6.6 the retained weights, including the original self-weight, are divided by their surviving row sum. Finite-run mean NEES does not prove covariance consistency.
