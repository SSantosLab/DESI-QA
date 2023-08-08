
# todo: 
#     deal with more positioners 
#     pass canbus and devbycan
#     now, all positioners are doing the same moves!

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
import test_spotfinder as ts
import tslib as tslib
import subprocess
import pandas as pd
import reportlib as rl
import getArcVals as gAV
from astropy.io import fits

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
def start_mount(port='/dev/ttyUSB0'):
    """
    Instantiate the ts mount class
    returns:
        cem120 (serial obj): object for moving mount class

    """
    cem120 = cf.initialize_mount(port)
    cf.set_alt_lim(cem120, -89)
    cf.slew_rate(cem120, 9)
    return cem120

    
def mount_posdown(cem120):
    """positioners down sequence, from home position

    Args:
        cem120 (_type_, optional): cem120 object (from Tsmount Class).
    """
    cf.home(cem120)
    cf.move_90(cem120, 0., 1, 0)
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 

def mount_posup(cem120):
    """positioners up sequence, from home position
    
    Args:
        cem120 (object): cem120 object (from Tsmount Class).
    """
    cf.home(cem120)

    # Position 8 - U   -> HOME->7->8 CAM UP  eixo 1  = x
    cf.move_90(cem120, 0., 1, 1)
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 

def movemount(mtpos, cem120):
    """Place holder for mount
    input:
        mtpos (): row (?) with the mount position
        move mount as Mount_[NOTSAFE].ipynb
    example: position up 
    #CAM UP
    """
    mtpos = tuple(np.array(mtpos, dtype=float))

    if mtpos==(90,90):
        mount_posdown(cem120)
    elif mtpos==(-90,90) or mtpos==(90,-90):
        mount_posup(cem120)
    elif mtpos==(0,0):
        cf.home(cem120)
    else:
        raise NotImplementedError("Only 0, 90, -90 positions are allowed now")
        
    return mtpos

def connect2pb():
    """Starts ShellHandler and returns the sh object
    """
    with open("conninfo.txt", "r") as ff: 
        pw = [i.strip('\n') for i in ff.readlines()]
    sh = ShellHandler(pw[0], pw[1], pw[2])
    print("Connection started")
    return sh


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

def read_xytable(ifn):
    """
    Read a positioners move table
    ifn (str): Input File Name w/ rows like:
                     'direction speed motor angle'
            e.g.  'cw cruise phi 180'
    """
    movelines = reader(open(ifn), skipinitialspace=True, delimiter=' ')
    movetable = [ i for i in movelines]

    for i, row in enumerate(movetable):
        if len(row) == 0:
            print("No move")
            return None
        else:
            assert len(row)==2, \
               f"Error: row {i} with {len(row)} Columns; should be 2!"
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
    mounttable = [ i for i in mountlines]
    for i, row in enumerate(mounttable):
        assert len(row)==2, \
               f"Error: mount row {i} with {len(row)} Columns; should be 2!"
    return mounttable

def ramp_angle(defcruise, deframp):
    """
    from speed table 
    """
    gear_ratio = (46/14 + 1)**4
    ramp_angle = defcruise*(defcruise+1)* deframp / 20/ gear_ratio
    return ramp_angle

def argscorrection(direc, speed, motor, ang, def_speed, prevdir, val_backlash=None):
    """
    def_speed = {"cruise":XX, spinramp:XX}
    backlash and ramp added here. Agnostic to motor 
    todo: check backlash number for a given positioner
    todo: implement the table of speeds here
    """
    print("---\n",direc, motor, prevdir)

    if val_backlash is None:
        val_backlash= 1.8

    if prevdir[motor] is None:
        add_backlash = 0.0
    elif prevdir[motor]==direc:
        add_backlash = 0.0
    elif direc != prevdir[motor]:
        print(f"Adding {val_backlash} deg of backlash")
        add_backlash = val_backlash
        
    if speed=='creep':
        add_ramp = 0.0
        
    elif speed=='cruise':
        cruise = def_speed['cruise']
        ramp = def_speed['spinramp']
        add_ramp = ramp_angle(cruise, ramp)
        # if cruise==33 and ramp== 12:
        #     assert np.isclose(add_ramp, 1.99549), "Inconsistent ramp_ang"
        # else:
        #     print("NotImplemented spin and cruise")
        #     return None
        # This should be updated for new parameters in testing
    ang2 = ang + add_backlash - 2*add_ramp
    print("------->", ang2)
    if (speed==cruise) and (ang2 < add_backlash+ 5e-2) :
        print(f"{ang2:.3f} move too small for cruise < 4.0")
        ang2=0
    return direc, speed, motor, ang2


def angle_between(c, p1, p2):
    # p1, p2 are points (x,c); c is center (xc, yc)
    a = np.array(p1)
    b = np.array(c)
    c = np.array(p2)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


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


# def write_db(session_label, imount, mvlabel, posid, imove, cent, 
#              dbname="output/database.csv"):
#     """
#     cent = centroids from spotfinder 
#     imove = [dir, speed, motor, angle ]
#     """
#     if not os.path.isfile(dbname):
#         print("DB file not found. Initializing a new one at {dbname}")
#         with open(dbname, 'w') as ndb:
#             ndb.write("label mount move posid direction speed motor angle xpix ypix peaks fwhm\n")

#     idir, ispeed, imotor, iangle = imove
#     with open(dbname, 'a') as fdb:
#         fdb.write(f"{session_label},{imount},") #mount wise
#         fdb.write(f"{mvlabel},{posid},{idir},{ispeed},{imotor},{iangle},") # requested move
#         #spotfinder points 
#         fdb.write(f"{cent['x'][0]},{cent['y'][0]:},{cent['peaks'][0]:.4f},{cent['fwhm'][0]:.4f}\n")  
#     return None


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


# SM TODO - write these three functions
# done
def updateCenter(center,xfid,yfid,xfid_old,yfid_old,pix2mm,pix2mm_old):
    xfid,yfid,xfid_old,yfid_old = np.array(xfid,dtype=float)*pix2mm,np.array(yfid,dtype=float)*pix2mm,np.array(xfid_old,dtype=float)*pix2mm_old,np.array(yfid_old,dtype=float)*pix2mm_old
    x_offset,y_offset = np.array([],dtype=float),np.array([],dtype=float)

    for i in range(len(xfid)):
        x_offset = np.append(x_offset,xfid[i]-xfid_old[i])
        y_offset = np.append(y_offset,yfid[i]-yfid_old[i])

    x_offset_mean,y_offset_mean = np.mean(x_offset),np.mean(y_offset)
    center[0] = center[0]+x_offset_mean
    center[1] = center[1]+y_offset_mean
    print("Updating center")
    return center 

# Done
def updateR(R1,R2,pix2mm,pix2mm_old):
    R1_new = R1 * pix2mm/pix2mm_old
    R2_new = R2 * pix2mm/pix2mm_old
    print("Updating R's of positioner")
    return R1_new,R2_new


# def updateAngle(hardstop_ang,xfid,yfid,xfid_old,yfid_old,pix2mm,pix2mm_old):


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
    hardstop_ang = {posid: -163.88353} # in deg
    R1 = {posid: 2.9134069  } # R theta
    R2 = {posid:  3.0696125} # R phi
    center = {posid: [69.96182, 30.92488]} # in mm

    pos_speed = {posid:{"cruise": 33, "spinramp": 12, }}#"spindown": 12}}
    pos_backlash = {posid: 1.9}

    # Region of interest for positioner

    # SM: I increased these limits (within reason) to encompass all configs of the single positioner
    # I think initial limits were determined off of restrictive bounds.
    # New limits were determined using arctheta sequence from 2023-06-14 12:19:00 -> x:[1750,2200], y:[700,1100], 
    # and accounting for variance in x & y pixels in different configs -> delta_x:10, delta_y:50 [pixels]
    # IF adding another positioner, we would have to make more restrictive bounds, or think of a new way to do this
    reg = {posid: { 'x':[1740,2210], 'y':[650,1150]}}

    # Todo: copy session config to remote
    #  session db should have
    #    session_label, pos_speed, pos_ramp, dev_bb, R1_pos, R2_pos, center_pos,
    #    pix2mm
    #  session_to_remove should have:
    #    scp: -> pos_speed, pos_ramp, dev_bb,

    #todo: activate a flag for xy in ini file
    xyflag = True

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
    try: 
        ncycle = cfg.getint('run', 'repeatcycle')
    except Exception as err:
        ncycle = 1
        print(f">> Err: {err}\n>> ncycle set as 1 ")
    
    # if hasmount: 
    mounttablefn = cfg.get('run', 'mounttable')
    
    if session_label is None:
        session_label = time.strftime("%Y%m%d-%H%M%S")
    else:
        _suf = time.strftime("%Y%m%d-%H%M%S")
        session_label = f"{session_label}-{_suf}"
    
   
    print("--"*8)
    print(pipeline)
    print(session_label, '\n')
    print(f"!! Number of Cycle(s): {ncycle}")
    print()
    print("--"*35,f"\n\t# {movetablefn}\n","--"*35 )

    dbname = "output/database.csv"
    dbsess = "output/sessions.csv"

    # -------------------------------------
    # Session Setup
    
    # Start mount
    cem120 = start_mount()

    # Connecting to PB
    sh = connect2pb()



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
    if xyflag:
        movetable = read_xytable(movetablefn)
    else:
        movetable = read_movetable(movetablefn)
    n_posmov = len(movetable)

    # Repeat cycle
    if isinstance(ncycle, int):
        print(f">> Repeating cycle {ncycle} times")
        movetable = movetable * ncycle
        assert isinstance(movetable, list), "move table is not a list"
        assert len(movetable) == n_posmov * ncycle, "move table is not the right size"
    
    print(f" Loop positioner size: {len(movetable)}" )

    prevdir = {'theta':None, 'phi':None}
    nextdir = None

    n_corrections = 0 # todo, read from ini file
    for i, imount in enumerate(mounttable): # For each mount configuration
        x_here,y_here = None,None # This should be inside mount loop
        print(f"\n>>  Mount move #{i}: {imount}\n")
        if imount !=0:  
            print(f"starting positioners loop for MOUNT in {imount}")
            mtang1, mtang2 = movemount(imount, cem120)
            print("Sleep for 30s to let mount rest")
            time.sleep(30) # Wait 30s
        else:
            mtang1, mtang2 = 0, 0
            print(f"starting positioners loop for MOUNT in {imount}")
            movemount(imount, cem120)   
            print("Sleep for 30s to let mount rest")
            time.sleep(30) # Wait 30s

        print("-----"*10,"\nRunning arcsequences for",[mtang1,mtang2],"mount configuration")
        # Run Arctheta and phi run_ang.py
        if mtang1 == 0 and mtang2 == 0: # Mount is in horizon
            subprocess.call(["python","run_ang.py","-c","conf/arcph15.ini"])
            subprocess.call(["python","run_ang.py","-c","conf/arcth30.ini"])
        elif (mtang1==90 and mtang2==90) or (mtang1==-90 and mtang2==-90): # Mount is in positioner down
            subprocess.call(["python","run_ang.py","-c","conf/arcph15_posdown.ini"])
            subprocess.call(["python","run_ang.py","-c","conf/arcth30_posdown.ini"])
        elif (mtang1==90 and mtang2==-90) or (mtang1==-90 and mtang2==90): # Mount is in positioner up
            subprocess.call(["python","run_ang.py","-c","conf/arcph15_posup.ini"])
            subprocess.call(["python","run_ang.py","-c","conf/arcth30_posup.ini"])
        else:
            raise NotImplementedError("Mount position not supported")

        if hascam and (not dryrun):
            cam = start_cam()

        print("Reading dates to get calibration values")
        #Get two most recent unique dates from the database, store them as date1 and date2 for the below call
        time.sleep(2)
        df = pd.read_csv("/home/msdos/DESI-QA/output/database.csv",usecols=[0,3])[-43:]
        reducedCol = rl.toNumpy(df['label'].str[8:])
        dates = np.unique(reducedCol)
        assert len(dates)==2, \
            f"Length of dates array incorrect: length should be 2, measured length is {len(dates)}"
        if dates[0]>dates[1]:
            date1 = dates[1]
            date2 = dates[0]
        else:
            date1 = dates[0]
            date2 = dates[1]

        del df
        # Get relevant values of R1 and R2, Hardstop angle, and xc, yc for that configuration using the session labels 
        R1[posid], R2[posid], xc, yc, xc2, yc2, hardstop_ang[posid] = gAV.main(str(date2),str(date1))
        center[posid] = [xc,yc]
        # Run a 30 degree cw move for both theta and phi arms to move off of home, set prevdir
        send_posmove("cw cruise phi 30 000", verbose=False)
        send_posmove("cw cruise theta 30 000", verbose=False)
        prevdir = {'theta':'cw', 'phi':'cw'}


        for j, imove in enumerate(movetable): # For each move in the movetable

            #MOVE THIS BECAUSE OF ROWS
            mvlabel = time.strftime("%Y%m%d-%H%M%S")
            ihash = np.random.randint(100000,1000000-1 )
           
            if not xyflag:
                mvargs = f"{imove[0]} {imove[1]} {imove[2]} {imove[3]} {ihash}"
            else:
                x_tgt, y_tgt = float(imove[0]), float(imove[1])

           
            if (x_here is None) or (y_here is None):
                print("Find current position XY; Assuming last ")

                # prevdir = {'theta':'ccw', 'phi':'ccw'} # Assuming parked at hardstop - NOPE, CAN'T DO THAT, I THINK THIS SHOULD BE CW
                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
                ff = fits.getdata(picpath+'/'+mvlabel+".fits")

                centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
                centroids = ts.get_spotpos("4852", centroidall, reg=reg)

                while (len(centroids)!=5) or (np.unique(ff)==[0]).all() or (np.unique(ff)[-10:]<45000).any():
                    print("Software error, taking image again")
                    os.remove(picpath+'/'+mvlabel+'.fits')
                    print(picpath+'/'+mvlabel+'.fits deleted successfully')
                    
                    get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun) 
                    ff = fits.getdata(picpath+'/'+mvlabel+".fits")
                    centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
                    centroids = ts.get_spotpos("4852", centroidall, reg=reg)   

                # todo: should be positioner-wise
                # TODO: assuming a single positioner    
                x_here = centroids['x'][0] 
                y_here = centroids['y'][0]
                x_home,y_home = x_here,y_here
                fidmask = ts.select_fidregion(centroidall)
                xfid_old, yfid_old = ts.get_xyfid(centroidall, fidmask)
                pix2mm_old, sigpix2mm_old  = ts.get_pix2mm(xfid_old, yfid_old)                

            # todo: id-wise this should be outside the loop
            xc, yc = center[posid] 
            xpos, ypos = xylib.refpix2pos(pix2mm, xc,yc, x_here, y_here)
            print("\n\nxy here:",end="")
            print(xpos, ypos,"\n\n")

            rows = xylib.calc_movetables(hardstop_ang[posid],
                                         R1[posid],R2[posid], 
                                         xpos,ypos,
                                         x_tgt,y_tgt)
            
            if len(rows[0])==0:
                print("No Move!")

            # Move WITHOUT updating targets
            mvlabel = time.strftime("%Y%m%d-%H%M%S")
            ihash = np.random.randint(100000,1000000-1 )
            n_rows = len(rows)
            for ii, irow in enumerate(rows): 
                if ii == n_rows-1: 
                    xytgt = 1
                else:
                    xytgt =0

                _dir, _motor, _ang = irow
                # todo : change to creep if it is a correction move
                ispeed = 'cruise'
                
                mv2 = argscorrection(_dir, ispeed, _motor, _ang, 
                                      pos_speed[posid], prevdir, 
                                      val_backlash=pos_backlash[posid]) 
                _dir, ispeed, _motor, _ang =  mv2

                mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
                # print(prevdir)
                print(mvargs)
                prevdir.update({_motor:_dir})

                #fao
                # sys.exit(1)
                # ----- move it ------------------------------------
                send_posmove(mvargs, verbose=False)    
                sys_status = confirm_move(ihash, get_remotehash(sh))
                
                if not sys_status:
                    print("PB not replied")
                    sys.exit(1) 

            #fao
            time.sleep(0.3) # trying to solve central fit issue

            # getting picture after trying to perform the 2 arms moves!        
            get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
            ff = fits.getdata(picpath+'/'+mvlabel+".fits")
            centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
            centroids = ts.get_spotpos(posid, centroidall, reg=reg)
            # Conditional statement to catch the times when the camera takes images with little/no data
            while (len(centroids)!=5) or (np.unique(ff)==[0]).all() or (np.unique(ff)[-10:]<45000).any():
                print("Software error, taking image again")
                os.remove(picpath+'/'+mvlabel+'.fits')
                print(picpath+'/'+mvlabel+'.fits deleted successfully')
                
                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun) 
                ff = fits.getdata(picpath+'/'+mvlabel+".fits")

                centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
                centroids = ts.get_spotpos(posid, centroidall, reg=reg)    

            fidmask = ts.select_fidregion(centroidall)
            xfid, yfid = ts.get_xyfid(centroidall, fidmask)
            pix2mm, sigpix2mm  = ts.get_pix2mm(xfid, yfid)
            tslib.write_fiddb(session_label, mvlabel, xfid, yfid, pix2mm, sigpix2mm)

            
            # todo: Adapt for multiple 
            x_here = centroids['x'][0] 
            y_here = centroids['y'][0] 
            xpos, ypos = xylib.refpix2pos(pix2mm, xc,yc, x_here, y_here)
            # print(f"curpos: ({xpos} {ypos}) ({xc}, {yc})")
            netphi, dist2center = print_info(centroids, posid)


            # todo: assuming centroids is len(1) lists
            assert len(centroids['x']) == 1, \
                  f"Not implemented for list {len(centroids['x'])}"
            

            jmove = [_dir, ispeed, _motor, _ang]      
            # write_db(session_label, imount, mvlabel, posid, jmove, centroids, 
            #          dbname=dbname)
            #fao todo: put this inside the rows loop for register steps
            write_db(session_label, mtang1, mtang2, mvlabel, posid, jmove, centroids, 
                        xytgt=xytgt, dbname=dbname)            

            # corrections 
            i_corr = 0
            if (n_corrections>0) and (np.hypot(xpos-x_tgt, ypos-ytgt)<=0.100):
                while i_corr<=n_corrections:

                    xpos, ypos = xylib.refpix2pos(pix2mm, xc,yc, x_here, y_here)
                    print(f"\n\nxy here: {xpos:.4f} {ypos:.4f}\n\n")

                    rows = xylib.calc_movetables(hardstop_ang[posid],
                                             R1[posid],R2[posid], 
                                             xpos,ypos,
                                             x_tgt,y_tgt)
                    # fao add move rows here
                    i_corr +=1
                    if (np.hypot(xpos-x_tgt, ypos-ytgt)<=0.100):
                        # save_to_database
                        break

            os.remove(picpath+'/'+mvlabel+'.fits')
            print(picpath+'/'+mvlabel+'.fits deleted successfully')
                        

            # SM TODO - done, except hardstop angle which i am leaving for now
            # UPDATE xc,yc,Rtheta,Rphi,hardstop angle here
            center[posid] = updateCenter(center[posid],xfid,yfid,xfid_old,yfid_old,pix2mm,pix2mm_old)
            R1[posid],R2[posid] = updateR(R1[posid],R2[posid],pix2mm,pix2mm_old)
            # hardstop_ang[posid] = updateAngle(hardstop_ang[posid],xfid,yfid,xfid_old,yfid_old,pix2mm,pix2mm_old) # Left out for now

            xfid_old,yfid_old = xfid,yfid
            """
            todo: correction move here, after the target
            while corrections:
                run a loop of get_mov->send_mov-> get_position
            """


        print("Sending positioner to home after finished with",[mtang1,mtang2],"mount configuration")

        # Send positioner to (-6,0)
        x_tgt, y_tgt = -6,0

        rows = xylib.calc_movetables(hardstop_ang[posid],
                                         R1[posid],R2[posid], 
                                         xpos,ypos,
                                         x_tgt,y_tgt)
        
        if len(rows[0])==0:
            print("No Move!")

        # Move WITHOUT updating targets
        mvlabel = time.strftime("%Y%m%d-%H%M%S")
        ihash = np.random.randint(100000,1000000-1 )
        n_rows = len(rows)
        for ii, irow in enumerate(rows): 
            if ii == n_rows-1: 
                xytgt = 1
            else:
                xytgt =0

            _dir, _motor, _ang = irow
            # todo : change to creep if it is a correction move
            ispeed = 'cruise'
            
            mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                                  pos_speed[posid], prevdir, 
                                  val_backlash=pos_backlash[posid]) 
            _dir, ispeed, _motor, _ang =  mv2

            mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
            # print(prevdir)
            print(mvargs)
            prevdir.update({_motor:_dir})

            #fao
            # sys.exit(1)
            # ----- move it ------------------------------------
            send_posmove(mvargs, verbose=False)    
            sys_status = confirm_move(ihash, get_remotehash(sh))
            
            if not sys_status:
                print("PB not replied")
                sys.exit(1) 

        #fao
        time.sleep(0.3) # trying to solve central fit issue


        # Move positioner 180-hardstop_angle degrees in ccw direction


        _dir, _motor, _ang = 'ccw','theta',str(180+hardstop_ang[posid])
        # todo : change to creep if it is a correction move
        ispeed = 'cruise'
        
        mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                              pos_speed[posid], prevdir, 
                              val_backlash=pos_backlash[posid]) 
        _dir, ispeed, _motor, _ang =  mv2

        mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
        # print(prevdir)
        print(mvargs)
        prevdir.update({_motor:_dir})

        #fao
        # sys.exit(1)
        # ----- move it ------------------------------------
        send_posmove(mvargs, verbose=False)    
        sys_status = confirm_move(ihash, get_remotehash(sh))
        
        if not sys_status:
            print("PB not replied")
            sys.exit(1) 

        #fao
        time.sleep(0.3) # trying to solve central fit issue

        # Close phi arm

        _dir, _motor, _ang = 'ccw','phi','180'
        # todo : change to creep if it is a correction move
        ispeed = 'cruise'
        
        mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                              pos_speed[posid], prevdir, 
                              val_backlash=pos_backlash[posid]) 
        _dir, ispeed, _motor, _ang =  mv2

        mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
        # print(prevdir)
        print(mvargs)
        prevdir.update({_motor:_dir})

        #fao
        # sys.exit(1)
        # ----- move it ------------------------------------
        send_posmove(mvargs, verbose=False)    
        sys_status = confirm_move(ihash, get_remotehash(sh))
        
        if not sys_status:
            print("PB not replied")
            sys.exit(1) 

        #fao
        time.sleep(0.3) # trying to solve central fit issue

        # Two small corrective 10 degree moves to push to the hardstop

        # Take pic here

        get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
        ff = fits.getdata(picpath+'/'+mvlabel+".fits")
        centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
        centroids = ts.get_spotpos("4852", centroidall, reg=reg)
        # Conditional statement to catch the times when the camera takes images with little/no data
        while (len(centroids)!=5) or (np.unique(ff)==[0]).all() or (np.unique(ff)[-10:]<45000).any():
            print("Software error, taking image again")
            os.remove(picpath+'/'+mvlabel+'.fits')
            print(picpath+'/'+mvlabel+'.fits deleted successfully')
            
            get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun) 
            ff = fits.getdata(picpath+'/'+mvlabel+".fits")

            centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
            centroids = ts.get_spotpos("4852", centroidall, reg=reg)

        fidmask = ts.select_fidregion(centroidall)
        xfid, yfid = ts.get_xyfid(centroidall, fidmask)
        pix2mm, sigpix2mm  = ts.get_pix2mm(xfid, yfid)
        
        # todo: Adapt for multiple 
        x_here = centroids['x'][0] 
        y_here = centroids['y'][0] 

        # First move

        _dir, _motor, _ang = 'ccw','theta','10'
        # todo : change to creep if it is a correction move
        ispeed = 'cruise'
        
        mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                              pos_speed[posid], prevdir, 
                              val_backlash=pos_backlash[posid]) 
        _dir, ispeed, _motor, _ang =  mv2

        mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
        # print(prevdir)
        print(mvargs)
        prevdir.update({_motor:_dir})

        #fao
        # sys.exit(1)
        # ----- move it ------------------------------------
        send_posmove(mvargs, verbose=False)    
        sys_status = confirm_move(ihash, get_remotehash(sh))
        
        if not sys_status:
            print("PB not replied")
            sys.exit(1) 

        #fao
        time.sleep(0.3) # trying to solve central fit issue

        _dir, _motor, _ang = 'ccw','phi','10'
        # todo : change to creep if it is a correction move
        ispeed = 'cruise'
        
        mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                              pos_speed[posid], prevdir, 
                              val_backlash=pos_backlash[posid]) 
        _dir, ispeed, _motor, _ang =  mv2

        mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
        # print(prevdir)
        print(mvargs)
        prevdir.update({_motor:_dir})

        #fao
        # sys.exit(1)
        # ----- move it ------------------------------------
        send_posmove(mvargs, verbose=False)    
        sys_status = confirm_move(ihash, get_remotehash(sh))
        
        if not sys_status:
            print("PB not replied")
            sys.exit(1) 

        #fao
        time.sleep(0.3) # trying to solve central fit issue

        # take pic here


        get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
        ff = fits.getdata(picpath+'/'+mvlabel+".fits")
        centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
        centroids = ts.get_spotpos("4852", centroidall, reg=reg)
        # Conditional statement to catch the times when the camera takes images with little/no data
        while (len(centroids)!=5) or (np.unique(ff)==[0]).all() or (np.unique(ff)[-10:]<45000).any():
            print("Software error, taking image again")
            os.remove(picpath+'/'+mvlabel+'.fits')
            print(picpath+'/'+mvlabel+'.fits deleted successfully')
            
            get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun) 
            ff = fits.getdata(picpath+'/'+mvlabel+".fits")
            centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
            centroids = ts.get_spotpos("4852", centroidall, reg=reg)

        fidmask = ts.select_fidregion(centroidall)
        xfid, yfid = ts.get_xyfid(centroidall, fidmask)
        pix2mm, sigpix2mm  = ts.get_pix2mm(xfid, yfid)
        
        # todo: Adapt for multiple 
        x_now = centroids['x'][0] 
        y_now = centroids['y'][0]

        deltapix = np.sqrt(np.square(x_now-x_here)+np.square(y_now-y_here))

        while deltapix>30:
            # Do it all again

            print("Unable to reach home, attempting corrective move")

            x_here,y_here = x_now,x_here

            # First move

            _dir, _motor, _ang = 'ccw','theta','10'
            # todo : change to creep if it is a correction move
            ispeed = 'cruise'
            
            mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                                  pos_speed[posid], prevdir, 
                                  val_backlash=pos_backlash[posid]) 
            _dir, ispeed, _motor, _ang =  mv2

            mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
            # print(prevdir)
            print(mvargs)
            prevdir.update({_motor:_dir})

            #fao
            # sys.exit(1)
            # ----- move it ------------------------------------
            send_posmove(mvargs, verbose=False)    
            sys_status = confirm_move(ihash, get_remotehash(sh))
            
            if not sys_status:
                print("PB not replied")
                sys.exit(1) 

            #fao
            time.sleep(0.3) # trying to solve central fit issue

            _dir, _motor, _ang = 'ccw','phi','10'
            # todo : change to creep if it is a correction move
            ispeed = 'cruise'
            
            mv2 = argscorrection(_dir, ispeed, _motor, float(_ang), 
                                  pos_speed[posid], prevdir, 
                                  val_backlash=pos_backlash[posid]) 
            _dir, ispeed, _motor, _ang =  mv2

            mvargs = f"{_dir} {ispeed} {_motor} {_ang:.5f} {ihash}"
            # print(prevdir)
            print(mvargs)
            prevdir.update({_motor:_dir})

            #fao
            # sys.exit(1)
            # ----- move it ------------------------------------
            send_posmove(mvargs, verbose=False)    
            sys_status = confirm_move(ihash, get_remotehash(sh))
            
            if not sys_status:
                print("PB not replied")
                sys.exit(1) 

            #fao
            time.sleep(0.3) # trying to solve central fit issue

            # take pic here


            get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
            ff = fits.getdata(picpath+'/'+mvlabel+".fits")
            centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
            centroids = ts.get_spotpos("4852", centroidall, reg=reg)
            # Conditional statement to catch the times when the camera takes images with little/no data
            while (len(centroids)!=5) or (np.unique(ff)==[0]).all() or (np.unique(ff)[-10:]<45000).any():
                print("Software error, taking image again")
                os.remove(picpath+'/'+mvlabel+'.fits')
                print(picpath+'/'+mvlabel+'.fits deleted successfully')
                
                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun) 
                ff = fits.getdata(picpath+'/'+mvlabel+".fits")

                centroidall = ts.get_spot(f"{mvlabel}.fits", 
                                        f"sbigpics/{session_label}", 
                                        expected_spot_count=5,
                                        verbose=False)
                centroids = ts.get_spotpos("4852", centroidall, reg=reg)

            fidmask = ts.select_fidregion(centroidall)
            xfid, yfid = ts.get_xyfid(centroidall, fidmask)
            pix2mm, sigpix2mm  = ts.get_pix2mm(xfid, yfid)
            
            # todo: Adapt for multiple 
            x_now = centroids['x'][0] 
            y_now = centroids['y'][0]

            deltapix = np.sqrt(np.square(x_now-x_here)+np.square(y_now-y_here))


        if hascam and (not dryrun):
            cam.close_camera()
    # todo generate a log!!!


   