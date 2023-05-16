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


def find_spot(fitsname, fitspath,  
              expected_spot_count=1, 
              regionsname='regions.reg', 
              verbose=False):
    """
    spotfinder handler
    input:
        fitsname:
        fitspath (str): relative or full path to the folder
        regionsname (str):
        verbose (bool):
    output: 
        centroids (dict): raw output from spotfinder

    """

    _ifn = f"{fitspath}/{fitsname}"

    if expected_spot_count != 1:
        raise NotImplementedError("This mode wasn't tested here")
    try: 
        sf=spotfinder.SpotFinder(_ifn, expected_spot_count)
        centroids = sf.get_centroids(print_summary = verbose, 
                                     region_file=regionsname)
        if verbose: print(centroids)
    
    except: #ignore photo if an error is raised
        print("Warning: spot not found ")
        inval_number = np.nan
        return {  'peaks': [inval_number], 
                      'x': [inval_number], 
                      'y': [inval_number], 
                   'fwhm': [inval_number], 
                 'energy': [inval_number]} 
    return centroids 


def start_cam(exposure_time=3000, is_dark=False):
    """Initialize sbigcam
    """
    cam = sbc.SBIGCam() 
    try: 
        cam.close_camera()
    except Exception as err:
        pass

    cam.open_camera()
    cam.select_camera('ST8300')
    cam.set_exposure_time(exposure_time)
    cam.set_dark(is_dark)
    return cam 


def get_picture(cam, imgname, rootout=None):
    """
    cam (SBIG.cam() obj): object from sbc
    """
    image = cam.start_exposure()
    # imgname = f"{rootout}/{time.strftime("%Y-%m-%d-%H%M%S")}"
    cam.write_fits(image, name = f"{rootout}/{imgname}.fits")
    print(f'Photo #{imgname} taken successfully.')


def write_data(loc_positioner, mtpos, mvlabel, session_label):
    """
    place holder for data writing
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


# ----------------
# starting CAM
# Start cam
# todo: replace by 
# cam = start_cam()
cam = sbc.SBIGCam() 
try: 
    cam.close_camera()
except Exception as err:
    pass

cam.open_camera()
cam.select_camera('ST8300')
cam.set_exposure_time(3000)
cam.set_dark(False)


#Prepare output files  
session_label = time.strftime("%Y%m%d-%H%M%S")
rootout = '/home/msdos/um_scripts/sbigpics/' # todo send this to yaml
picpath = os.path.join(rootout, session_label)
if not os.path.exists(picpath):
    os.makedirs(picpath)
print("session imgs in: ", picpath) 


# loop here
"""

1) Move mount
2) get coordinates:
    generate label: session_label + move_label(time stamp) + move_mount pos, 
    2.1) Move positioners
    2.2) take pics
    2.3) save pics with label and save nominal values for positioners move
"""

movetable = [('cw', 'cruise', 'theta', '10'),
             ('ccw', 'cruise', 'theta', '30'),
             ('cw', 'cruise', 'theta', '30'),
             ('ccw', 'cruise', 'theta', '10')]

movetable = [( 'cw', 'cruise', 'theta', '0'),
             ('ccw', 'cruise', 'theta', '0'),
             ( 'cw', 'cruise', 'theta', '0'),
             ('ccw', 'cruise', 'theta', '0')]

# todo: make it usable here
movetablefn ="movetables/movetest.txt"
moves = reader(open(movetablefn), delimiter=' ')

# For the mount:
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

        cmd =f"python3 fao_arcseq.py {mvstring}" 

        stdin, stdout, stderr = ssh.exec_command(f"cd /home/msdos/pclight/trenz; {cmd};")
        output = stdout.readlines()
        stdout.channel.set_combine_stderr(True)
        print("pb:",i, output)

        ret_hash = output[-1].replace('\n','')
        print("************", inp_hash, ret_hash)
        if "88"+str(inp_hash)+'88'== ret_hash:
            print("\t Successfully talked to PB")
        else:
            sys_status=False
            print()
            sys.exit(f"Error: Problem in positioners or in PB during \nmove: {imove}")

        if sys_status:
            get_picture(cam, mvlabel, rootout=picpath)
            pix_position = find_spot(f"{mvlabel}.fits", {picpath})
            write_data(pix_position, mtpos, mvlabel, session_label, imotor, ispeed, idirection, iangle,  idpos)

cam.close_camera() 
# todo: close paramiko ! 
