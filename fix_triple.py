import re
p="app_fixed.py"
s=open(p,"r",encoding="utf8").read()
dq = len(re.findall('"""',s))
sq = len(re.findall("'''",s))
if dq % 2 == 1:
    s += "\n\"\"\"\n"
if sq % 2 == 1:
    s += "\n'''\n"
open("app_fixed_auto.py","w",encoding="utf8").write(s)
print("Wrote app_fixed_auto.py (appended closing triple quote(s) if needed).")
