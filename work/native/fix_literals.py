"""Repair C/C++ string literals whose \\n escape collapsed into a real newline.

Editing C++ through a shell heredoc into Python into a file puts three escaping layers
between intent and disk, and "\\n" loses a backslash at each one.  This puts it back.

    py -3.10 work/native/fix_literals.py <file> [...]
"""
import re
import sys

BROKEN = re.compile(r'"([^"\n]*)\r?\n(\s*)"')

for path in sys.argv[1:]:
    raw = open(path, 'rb').read().decode('utf-8')
    fixed, n = BROKEN.subn(lambda m: '"' + m.group(1) + '\\n"', raw)
    while True:
        fixed, extra = BROKEN.subn(lambda m: '"' + m.group(1) + '\\n"', fixed)
        n += extra
        if not extra:
            break
    if n:
        open(path, 'w', encoding='utf-8', newline='\n').write(fixed)
    print('%-60s repaired %d' % (path.split('\\')[-1], n))
