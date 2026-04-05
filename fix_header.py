import re
p='app.py'
s=open(p,encoding='utf8').read().splitlines()
code_re = re.compile(r'^\s*(import|from|def|class|if\s+|for\s+|while\s+|with\s+|@|\#|\"\"\"|\'\'\'|async\s+|try\s*:|except\s+:)')
idx = len(s)
for i,line in enumerate(s):
    if code_re.match(line):
        idx = i
        break
# comment ทุกบรรทัดก่อน idx ที่ยังไม่ใช่ comment
new = []
for i,line in enumerate(s):
    if i < idx and not line.strip().startswith('#'):
        new.append('# ' + line)
    else:
        new.append(line)
out='app_fixed2.py'
open(out,'w',encoding='utf8').write('\n'.join(new))
print(f"Wrote {out}. Commented first {idx} lines (if any).")
