# ruff: noqa: E402
import os
import py_compile
import sys

# The extracted .py files always land in the dune workspace build root (_build/default/).
# Walk up from __file__ to find it — works whether or not dune-workspace is present.
_d = os.path.dirname(os.path.abspath(__file__))
while not (
    os.path.basename(_d) == "default"
    and os.path.basename(os.path.dirname(_d)) == "_build"
):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)

for f in [
    "nat_add.py",
    "mk_pair_r.py",
    "zeros.py",
    "uint_val.py",
    "float_val.py",
    "str_val.py",
    "todo_val.py",
]:
    py_compile.compile(os.path.join(_d, f), doraise=True)
del _d
print("All extracted .py files are syntactically valid Python.")
