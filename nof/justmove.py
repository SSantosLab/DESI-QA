#
"""
Usage: python connect2.py
"""
import paramiko
import numpy as np
import sbigCam as sbc
import time
import sys
import os
from csv import reader
import spotfinder

# TODO: 
# add fipos.gentle_exit() in the other end


def send_move():
    """Place holder
    """
    pass


def movemount(mtpos):
    """Place holder for mount
    mtpos (): 
        list (?) with the mount position
    """
    pass



sys_status=True

print("WARNING: \t1.CHECK YOUR PHYSICAL LIMITS\n\t2.CHECK WITH POSITIONERS ARE IN HOME!")
# HumanCheck = input("Have you checked it? (y|n)")
# if HumanCheck.lower() !='y':
#     sys.exit("You shall not pass!!!")


# ------------------
# Connecting to PetalBOX:
ip = '141.211.99.73'  # noqa: E225
port = 22
username = 'msdos'
password = 'M@y@ll4m'

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())# surpass the known host polices
ssh.connect(hostname=ip, port=port, username=username, password=password,
            banner_timeout=100, auth_timeout=100)



import pdb #; pdb.set_trace()
movetable = [( 'cw', 'cruise', 'theta', '0'),
             # ('ccw', 'cruise', 'theta', '0'),
             # ( 'cw', 'cruise', 'theta', '0'),
             ('ccw', 'cruise', 'theta', '0')]


mountpos_table = [0]

print("Starting move sequence")
for j, mtpos in enumerate(mountpos_table):
    if mtpos !=0:  # noqa: W191
        sys.exit("NotImplemented: Only zero position for mount is allowed now")
    else: 
        print(f"starting positioners loop for mount in {mtpos}")
    movemount(mtpos)

    for i, imove in enumerate(movetable):

        #imove : direction speed motor angle 
        idirection, imotor, ispeed, iangle = imove
        
        mvlabel = time.strftime("%Y%m%d-%H%M%S")
        inp_hash = np.random.randint(10000)
        mvstring = f"{idirection} {imotor} {ispeed} {iangle} {inp_hash}"
        mvstring = f"{imove[0]} {imove[1]} {imove[2]} {imove[3]} {inp_hash}" # todo: remove me
        print(i, mvstring, mvlabel)
        cmd0 = "export PYTHONPATH=$PYTHONPATH:/home/msdos/pb_svn/petalbox/trunk:"
              +"/home/msdos/pb_svn/petalbox/trunk/control:/home/msdos/dos_home/dos_products"
        cmd =f"python3 fao_arcseq.py {mvstring}" 

        cmd = f"echo SAASDASDAS; python3 fao_test.py; echo"
        cmd = "pwd; /usr/bin/python3 -c 'import sys; print(sys.path)'"
        #pdb.set_trace()
        stdin, stdout, stderr = ssh.exec_command(f"cd /home/msdos/pclight/trenz && {cmd0} && {cmd};"); print(stderr.readlines())
        output = stdout.readlines()

        print(output)
        stdout.channel.set_combine_stderr(True)
        print("pb:",i, output)
        print("\n\n\n")
        
        # ret_hash = output[-1].replace('\n','')
        # print("************", inp_hash, ret_hash)
        # if "88"+str(inp_hash)+'88'== ret_hash:
        #     print("\t Successfully talked to PB")

# todo: close paramiko ! 
