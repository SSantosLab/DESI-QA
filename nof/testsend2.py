"""
Testing: 
    - sending commands 
    - keyboard stop CTRL+C
Usage:
    labmachine: python3 testsend2.py
    PBox :      tail -f ~/comm/watched.txt
"""

from src.phandler import ShellHandler
import time 

# ------------------
# Connecting to PetalBOX:
ip = '141.211.99.73'  # noqa: E225
port = 22
username = 'msdos'
password = ''


sh = ShellHandler(ip, username, password)
print("Connection started")

# sending commands
sin, sout, serr = sh.execute("cd /home/msdos/pclight/trenz; python3 fao_test.py")
sin, sout, serr = sh.execute(f"echo STARTING > /home/msdos/comm/watched.txt")

# printing remote screen:
print("err:", [line for line in serr])
for line in sout:
    print(line, end='')

for i in range(6):
    print(f"{i} sleeping...")
    time.sleep(1)

    sh.execute(f"echo {i} >> /home/msdos/comm/watched.txt")
    sin, sout, serr = sh.execute("pwd")

    print(sin, sout, serr)
    for line in sout:
        print(line, end='')

    if len(serr)!=0:
        print("Error:", end=" ")
        for errline in serr:
            print(errline)

# getting the last line of file watched
sin, sout, serr = sh.execute("cd /home/msdos/comm; tail -n1 watched.txt")
print([line.strip('\n') for line in sout])

