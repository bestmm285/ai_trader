import re
s=open("app_fixed.py","r",encoding="utf8").read()
for m in re.finditer(r'("""|\'\'\')',s):
    line = s.count("\n",0,m.start())+1
    print(f"{m.group()} at line {line}")
print("Total triple-quote occurrences:", len(re.findall(r'(\"\"\"|\'\'\')',s)))
