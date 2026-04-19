import py_compile

for f in [
    "nat_add.py",
    "mk_pair_r.py",
    "zeros.py",
    "uint_val.py",
    "float_val.py",
    "str_val.py",
    "todo_val.py",
]:
    py_compile.compile(f, doraise=True)
print("All extracted .py files are syntactically valid Python.")
