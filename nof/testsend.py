# trying invoke shell 
#
import paramiko
import re

import sys
import os
import time


# stdin, stdout, stderr = ssh.exec_command(f"cd /home/msdos/pclight/trenz; {cmd};")
# time.sleep(5)

# print(stdin.readlines(), end="\n\n")
# # print(stderr.readlines())

# stdin.close()
class ShellHandler:

    def __init__(self, host, user, psw):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(host, username=user, password=psw, port=22)

        channel = self.ssh.invoke_shell()
        self.stdin = channel.makefile('wb')
        self.stdout = channel.makefile('r')

    def __del__(self):
        self.ssh.close()

    def execute(self, cmd):
        """

        :param cmd: the command to be executed on the remote computer
        :examples:  execute('ls')
                    execute('finger')
                    execute('cd folder_name')
        """
        cmd = cmd.strip('\n')
        self.stdin.write(cmd + '\n')
        finish = 'end of stdOUT buffer. finished with exit status'
        echo_cmd = 'echo {} $?'.format(finish)
        self.stdin.write(echo_cmd + '\n')
        shin = self.stdin
        self.stdin.flush()

        shout = []
        sherr = []
        exit_status = 0
        for line in self.stdout:
            if str(line).startswith(cmd) or str(line).startswith(echo_cmd):
                # up for now filled with shell junk from stdin
                shout = []
            elif str(line).startswith(finish):
                # our finish command ends with the exit status
                exit_status = int(str(line).rsplit(maxsplit=1)[1])
                if exit_status:
                    # stderr is combined with stdout.
                    # thus, swap sherr with shout in a case of failure.
                    sherr = shout
                    shout = []
                break
            else:
                # get rid of 'coloring and formatting' special characters
                shout.append(re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]').sub('', line).
                             replace('\b', '').replace('\r', ''))

        # first and last lines of shout/sherr contain a prompt
        if shout and echo_cmd in shout[-1]:
            shout.pop()
        if shout and cmd in shout[0]:
            shout.pop(0)
        if sherr and echo_cmd in sherr[-1]:
            sherr.pop()
        if sherr and cmd in sherr[0]:
            sherr.pop(0)

        return shin, shout, sherr



# ------------------
# Connecting to PetalBOX:
ip = '141.211.99.73'  # noqa: E225
port = 22
username = 'msdos'
password = 'M@y@ll4m'

sh = ShellHandler(ip, username, password)

sin, sout, serr = sh.execute("cd /home/msdos/pclight/trenz; python3 fao_test.py")
sin, sout, serr = sh.execute(f"echo STARTING > /home/msdos/comm/watched.txt")

print(sin, sout, serr)
for line in sout:
    print(line, end='')

for i in range(10):
    print(f"{i} sleeping...")
    time.sleep(1)

    sh.execute(f"echo {i} >> /home/msdos/comm/watched.txt")
    sin, sout, serr = sh.execute("pwd")

    print(sin, sout, serr)
    for line in sout:
        print(line, end='')

