with open('main.py','r',encoding='utf-8') as f:
    lines=f.readlines()
for i in range(2515,2539):
    if i<len(lines) and lines[i].startswith('                ') and lines[i].strip():
        lines[i]='    '+lines[i]
with open('main.py','w',encoding='utf-8') as f:
    f.writelines(lines)
print('OK')
