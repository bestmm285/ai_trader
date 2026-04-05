import re, sys, os
p = "app.py"
bak = p + ".bak"
# backup original if not exists
if not os.path.exists(bak):
    open(bak, "w", encoding="utf8").write(open(p, "r", encoding="utf8").read())
s = open(p, "r", encoding="utf8").read()
lines = s.splitlines()
code_re = re.compile(r'^\s*(import|from|def|class|if\s+|for\s+|while\s+|with\s+|@|\#|"""|\'\'\'|async\s+|try\s*:|except\s+:|\#!)')
idx = len(lines)
for i, line in enumerate(lines):
    if code_re.match(line):
        idx = i
        break
# comment header lines ก่อน idx
new = []
for i, line in enumerate(lines):
    if i < idx and not line.lstrip().startswith("#"):
        new.append("# " + line)
    else:
        new.append(line)
s2 = "\n".join(new)
# balance triple quotes by appending closers if needed
dq = s2.count('"""')
sq = s2.count("'''")
add = ""
if dq % 2 == 1:
    add += "\n\"\"\"\n"
if sq % 2 == 1:
    add += "\n'''\n"
out = "app_fixed3.py"
open(out, "w", encoding="utf8").write(s2 + add)
print(f"Wrote {out}. Commented first {idx} lines and appended closing triple quote(s) if needed.")
