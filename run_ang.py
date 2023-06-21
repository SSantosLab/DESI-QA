"""
Using new firmware: 6.23
(Formely connect5.py)

todo: 

    deal with more positioners 
    pass canbus and devbycan
    now, all positioners are doing the same moves!
    
"""
import numpy as np
from csv import reader
import configparser
import time 
from datetime import datetime
import os
import sys
sys.path.append('/data/common/software/products/tsmount-umich/python')
import cem120func as cf
from src.phandler import ShellHandler
import sbigCam as sbc
from spotfinder import spotfinder
import xylib as xylib


# I/O Session: 
def read_movetable(ifn):
    """
    Read a positioners move table
    ifn (str): Input File Name w/ rows like:
                     'direction speed motor angle'
            e.g.  'cw cruise phi 180'
    """
    movelines = reader(open(ifn), skipinitialspace=True, delimiter=' ')
    movetable = [ i for i in movelines]
    for i, row in enumerate(movetable):
        assert len(row)==4, \
               f"Error: row {i} with {len(row)} Columns; should be 4!"
    return movetable


def read_mounttable(ifn):
    """
    Read a positioners move table
    ifn (str): Input File Name w/ rows like:
                     'direction speed motor angle'
            e.g.  'cw cruise phi 180'
    """
    if (ifn is None) or ifn=='':
        print("Mount table not found! Using home")
        return [0]
    mountlines = reader(open(ifn), skipinitialspace=True, delimiter=' ')
    mounttable = [ i for i in movelines]
    for i, row in enumerate(movetable):
        assert len(row)==1, \
               f"Error: mount row {i} with {len(row)} Columns; should be 1!"
    return mounttable


# Cam session:
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


def get_picture(cam, imgname, rootout=None, dryrun=False):
    """
    cam (SBIG.cam() obj): object from sbc
    """
    image = cam.start_exposure()
    # imgname = f"{rootout}/{time.strftime("%Y-%m-%d-%H%M%S")}"
    cam.write_fits(image, name = f"{rootout}/{imgname}.fits")
    print(f'Photo #{imgname} taken successfully.')


# Mount Session
def start_mount():
    """
    Instantiate the ts mount class
    returns:
        cem120 (serial obj): object for moving mount class

    """
    cem120 = cf.initialize_mount("/dev/ttyUSB0")
    cf.set_alt_lim(cem120, -89)
    cf.slew_rate(cem120, 9)
    return cem120


def movemount(mtpos, cem120=cem120):
    """Place holder for mount
    input:
        mtpos (): row (?) with the mount position
        move mount as Mount_[NOTSAFE].ipynb
    example: position up 
    #CAM UP
    """
    if mtpos==(90,90):
        mount_posdown(cem120)
    elif mtpos==(-90,90) or mtpos==(90,-90):
        mount_posup(cem120)
    elif mtpos==(0,0):
        cf.home(cem120)
    else:
        raise NotImplementedError("Only 0, 90, -90 positions are allowed now")
        
    return mtpos
    
def mount_posdown(cem120=cem120):
    """positioners down sequence, from home position

    Args:
        cem120 (_type_, optional): _description_. Defaults to cem120.
    """
    cf.home(cem120)
    cf.move_90(cem120, 0., 1, 0)
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 

def mount_posup(cem120=cem120):
    """positioners up sequence, from home position
    """
    cf.home(cem120)

    # Position 8 - U   -> HOME->7->8 CAM UP  eixo 1  = x
    cf.move_90(cem120, 0., 1, 1)
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 


# Petal Box Session:
def connect2pb():
    """Starts ShellHandler and returns the sh object
    """
    with open("conninfo.txt", "r") as ff: 
        pw = [i.strip('\n') for i in ff.readlines()]
    sh = ShellHandler(pw[0], pw[1], pw[2])
    print("Connection started")
    return sh


def send_posmove(mvargs, remote_script="fao_seq20.py", verbose=False):
    """
    Move a positioner 
    mvargs (str): 
        args for sequence
        'cw cruise phi 0 somehash'
    remote_script (str):
        path and filename in Petal Box for the script that calls 
        fiposcontroller.
    """
    cmd = f"cd {pbroot}; python3 {remote_script} {mvargs}"
    sin, sout, serr = sh.execute(cmd)
    if verbose:
        sh.print(sout)
        sh.print(serr, mark="::")
    return sout, serr


def get_remotehash(sh, hashfile="/home/msdos/pclight/trenz/umhash.txt"):
    """ 
    Read the last line of hash file to check if the fiposcontroller 
    worked in remote Petal Box
    hashfile (str):
        path in PB to a file that saved hash after move
    return:
        hash (str)
    """
    sin, sout, serr = sh.execute(f"tail -n1 {hashfile}")

    # Asserting no return error
    if len(serr) !=0:
        print("Error (manager):")
        for line in serr:
            print("\t\t", line.strip('\n'))
        return None

    return sout[-1].strip('\n')



def confirm_move(ihash, ohash):
    """
    Compares remote output_hash (read from get_remotehash()) with 
    input_hash
    """
    if ohash == f'#UM{ihash}fff':
        print("Sucessful reply from PB")
        return True
    else:
        print(ohash, f"#UM{ihash}fff")
        print("Error in PB reply")
        return False

# Analysis Session:
def get_spot(fitsname, fitspath,  
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
    assert isinstance(fitsname, str)

    _ifn = f"{fitspath}/{fitsname}"

    if expected_spot_count != 1:
        raise NotImplementedError("This mode wasn't tested here")
    try: 
        sf=spotfinder.SpotFinder(_ifn, expected_spot_count)
        centroids = sf.get_centroids(print_summary = verbose, 
                                     region_file=regionsname)
        if verbose: print(centroids)
    
    except Exception as err: #ignore photo if an error is raised
        print("Warning: spot not found ")
        print(f"\terr: {err}")
        inval_number = np.nan
        return {  'peaks': [inval_number], 
                      'x': [inval_number], 
                      'y': [inval_number], 
                   'fwhm': [inval_number], 
                 'energy': [inval_number]} 
    return centroids


def write_db(session_label, mtang1, mtang2, mvlabel, posid, imove, cent, xytgt=0,
             dbname="output/database.csv"):
    """
    cent = centroids from spotfinder 
    imove = [dir, speed, motor, angle ]
    # add is target in xytgt
    # add new mount position angles 
    """
    if not os.path.isfile(dbname):
        print(f"DB file not found. Initializing a new one at {dbname}")
        with open(dbname, 'w') as ndb:
            ndb.write("label,mtang1,mtang2,move,posid,direction,speed,motor,angle,xpix,ypix,xytgt,peaks,fwhm\n")

    idir, ispeed, imotor, iangle = imove
    with open(dbname, 'a') as fdb:
        fdb.write(f"{session_label},{mtang1:.6f},{mtang2:.6f},") #mount wise
        fdb.write(f"{mvlabel},{posid},{idir},{ispeed},{imotor},{float(iangle):.6f},") # asked move
        #spotfinder points 
        fdb.write(f"{cent['x'][0]:.6f},{cent['y'][0]:.6f},{xytgt:d},")
        fdb.write(f"{cent['peaks'][0]:.4f},{cent['fwhm'][0]:.4f}\n")
    return None

def print_info(centroids, posid):

    # -------------------------------------------------------------------------    
    # For users: 
    x_here = centroids['x'][0]
    y_here = centroids['y'][0]
    x_pos = x_here*pix2mm - center[posid][0]
    y_pos = y_here*pix2mm - center[posid][1]
    rc = np.hypot(x_pos, y_pos)
    r1, r2 = R1[posid], R2[posid]
    arccos_arg = (r1**2 + r2**2 -rc**2)/(2*r1*r2)
    # guessing if it is 180 or 0:
    if np.isclose(arccos_arg, 1):
        if rc >1.5* r1:
            netphi = 180.
        elif rc < 0.5 * r1:
            netphi = 0.0
    else:        
        netphi = np.rad2deg(np.arccos(np.round(arccos_arg, 4)))
    print(f"\t\tx,y:\n {x_pos:4.6f}, {y_pos:4.6f} \n {x_here:.6f}, {y_here:.6f}")
    print(f"dist_xy2c: {rc:.6f} mm\nphi: {netphi: .4f} deg")
    return netphi, rc

# def create_log(session_label):
#     """
#     Create a log file with the session information
#     """
#     logfile = f"logs/{session_label}.log"
#     if os.isfile(logfile):
#         print("warning: log file already exists")
        
#     with open logfile as flog:
#         flog.write("")
#     pass  
    
    
if __name__=='__main__':
    #TODO: receive config as parsed argument
    #      write logs at the end
    #      close connections to cam, mount

    # -------------------------------------------
    # configuration:
    import argparse

    # TODO: remove hardcoded stuff here
    pix2mm = 0.03536752443853155 #0.035337
    # todo, pass this as dictionary per positioner
    posid = '4852' # string
    hardstop_ang = {"4852": -162.62196} # in deg
    R1 = {"4852": 2.979753 } # R theta
    R2 = {"4852": 3.0684968} # R phi
    center = {"4852":[69.77972167673121,31.529107849530625]}#[ 69.91212, 31.49481]}

    pos_speed = {"4852":{"cruise": 33, "spinramp": 12, }}#"spindown": 12}}
    pos_backlash = {"4852": 1.9}

    # Todo: copy session config to remote
    #  session db should have
    #    session_label, pos_speed, pos_ramp, dev_bb, R1_pos, R2_pos, center_pos,
    #    pix2mm
    #  session_to_remove should have:
    #    scp: -> pos_speed, pos_ramp, dev_bb
    # -------------------------------------------
    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config', 
                        required=True, 
                        type=str, 
                        help="Configuration filename .ini at 'conf/' folder")

    args = parser.parse_args() 
    sess_config = args.config
    # e.g. example.ini


    cfg = configparser.ConfigParser()
    cfg.read(f'{sess_config}')

    pipeline = cfg.get('run', 'pipeline').split(" ")
    dryrun = cfg.getboolean('run', 'dryrun')
    session_label = cfg.get('run', 'session_label')

    pbroot = cfg.get('run', 'pbroot')
    rootout = cfg.get('run', 'rootout')

    hascam = 'cam' in pipeline
    hasmount = 'mount' in pipeline
    haspos = 'positioners' in pipeline
    hasspot = 'spotfinder' in pipeline

    # if haspos: 
    movetablefn = cfg.get('run', 'movetable') 
    
    # if hasmount: 
    mounttablefn = cfg.get('run', 'mounttable')
    
    if session_label is None:
        session_label = time.strftime("%Y%m%d-%H%M%S")
    else:
        _suf = time.strftime("%Y%m%d-%H%M%S")
        session_label = f"{session_label}-{_suf}"
    
   
    print("--"*8)
    print(pipeline)
    print(session_label)
    print()
    print("--"*35,f"\n\t# {movetablefn}\n","--"*35 )

    dbname = "output/database.csv"

    # -------------------------------------
    # Session Setup
    
    # Connecting to PB
    sh = connect2pb()

    if hascam and (not dryrun):
        cam = start_cam()

    picpath = os.path.join(rootout, session_label)

    if (not os.path.exists(picpath)) and (not dryrun):
        os.makedirs(picpath)
        print("----"*10 + "\nsession imgs in: ", picpath) 


  

    sys_status=True

    # 1. get_movetable
    # 2. get_mounttable
    # 3. PRINT Planning
    """
    LOOP MOUNT
        LOOP POSITIONERS
    """
    mounttable = read_mounttable(mounttablefn)
    movetable = read_movetable(movetablefn)
    
    #unpark theta and phi
    if False:
        print(" Unparking the positioners with 5deg creep move:")
        send_posmove("cw creep phi 5 000", verbose=False)
        send_posmove("cw creep theta 5 000", verbose=False)

    print(f" Loop positioner size: {len(movetable)}" )
    netphi = 0
    for i, imount in enumerate(mounttable):
        if imount !=0:  
            print(f"starting positioners loop for MOUNT in {imount}")
            mtang1, mtang2 = movemount(imount)
        else:
            mtang1, mtang2 = 0, 0

            print(f"starting positioners loop for MOUNT in {imount}")
            movemount(imount)        


        for j, imove in enumerate(movetable):
            mvlabel = time.strftime("%Y%m%d-%H%M%S")

            ihash = np.random.randint(100000,1000000-1 )

            # example mvargs = f"cw cruise phi angle {ihash}"
            mvargs = f"{imove[0]} {imove[1]} {imove[2]} {imove[3]} {ihash}"
            dang = [float(imove[3]) if imove[0]=='phi' else 0][0]

            # todo: remove comments below
            # if (dang +netphi > 200) and (imove[0]=='cw'):
            #     print("Error: Max Hardstop phi reached! ")
            #     sys.exit(1)
            # if (dang -netphi > -0.05) and (imove[0]=='ccw'):
            #     print("Error: Min Hardstop phi reached! ")    
            #     sys.exit(1)


            if haspos: #dryrun
                send_posmove(mvargs, verbose=False)    

                # confirm remote
                sys_status = confirm_move(ihash, get_remotehash(sh))


            if not sys_status:
                sys.exit(1)
            elif (not dryrun):
                # 1. get_pic
                # 2. analyze pics
                # 3. sanity checks related to position:
                #       - there's a pic
                #       - the spot was detected
                #       - next move is not passing physical limits
                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
                centroids = get_spot(f"{mvlabel}.fits", f"sbigpics/{session_label}", verbose=False)

                netphi, dist2center = print_info(centroids, posid)
                thobs, phobs = xylib.transform(hardstop_ang['4852'], 
                    R1['4852'], R2['4852'] + 0.15, 
                    centroids['x'][0]*pix2mm - center['4852'][0],
                    -(centroids['y'][0]*pix2mm -center['4852'][1]) )

                print(f"(th, ph): {thobs:.4f}, {phobs:.4f}")
                write_db(session_label, mtang1, mtang2, mvlabel, posid, imove, centroids, 
                        xytgt=0, dbname=dbname)
                # placeholder: _last_position = []
                # sanity_check_for_phys_lim(_last_position, next_pos, arccenter_posid, returns:sys_status)


    if hascam and (not dryrun):
        cam.close_camera()
    # todo generate a log!!!


   