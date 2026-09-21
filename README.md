# On the diameter and distribution of Sidon sets
This repository hosts code and data for the paper "On the diameter and distribution of Sidon sets".

The code `verify.py` takes as input a folder containing a WT-triple witness, verifies it satisfies all of the required properties, and computes the resulting WT-triple and associated bounds. For instance, run `python.exe verify.py explicit` to verify the bound in Corollary 1.1.

The code `search.py` has the search procedure used to find high-quality approximate WT-triples. The files prefixed by `high_prec` have code used to convert an approximate solution to an exact solution. Further instructions for using that code may be found in `high_prec_instructions.txt`.