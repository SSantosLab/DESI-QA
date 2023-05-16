"""

todo: 
    deal with more positioners 
    pass canbus and devbycan
    now, all positioners are doing the same moves!
    
"""
 
from src.phandler import ShellHandler
import time 
import sbigCam as sbc
import time
import sys
import os
from csv import reader
from spotfinder import spotfinder
import numpy as np
import configparser
import xylib as xylib


sys.path.append('/data/common/software/products/tsmount-umich/python')
import cem120func as cf


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


def start_mount():
    """
    Instantiate the ts mount class
    returns:
        cem120 (serial obj): object for moving mount class

    """
    cem120 = cf.initialize_mount("/dev/ttyUSB0")
    return cem120



def movemount(mtpos):
    """Place holder for mount
    input:
        mtpos (): list (?) with the mount position
    """
    pass


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
    if len(row) == 0:
        print("No correction")
        return None
    for i, row in enumerate(movetable):
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
    mounttable = [ i for i in movelines]
    for i, row in enumerate(movetable):
        assert len(row)==1, \
               f"Error: mount row {i} with {len(row)} Columns; should be 1!"
    return mounttable


def send_posmove(mvargs, remote_script="fao_seq.py", verbose=False):
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


def write_db(session_label, imount, mvlabel, posid, imove, cent, 
             dbname="output/database.csv"):
    """
    cent = centroids from spotfinder 
    imove = [dir, speed, motor, angle ]
    """
    if not os.path.isfile(dbname):
        print("DB file not found. Initializing a new one at {dbname}")
        with open(dbname, 'w') as ndb:
            ndb.write("label mount move posid direction speed motor angle xpix ypix peaks fwhm\n")

    idir, ispeed, imotor, iangle = imove
    with open(dbname, 'a') as fdb:
        fdb.write(f"{session_label} {imount} ") #mount wise
        fdb.write(f"{mvlabel} {posid} {idir} {ispeed} {imotor} {iangle} ") # asked move
        #spotfinder points 
        fdb.write(f"{cent['x'][0]} {cent['y'][0]:} {cent['peaks'][0]:.4f} {cent['fwhm'][0]:.4f}\n")  
    return None




if __name__=='__main__':
    #TODO: receive config as parsed argument
    #      write logs at the end
    #      close connections to cam, mount


    # -------------------------------------------
    # configuration:
    import argparse

    # TODO: remove hardcoded stuff here
    pix2mm = 0.035337
    # todo, pass this as dictionary per positioner
    posid = '4852'
    hardstop_ang = {"4852": 166.270588} # in deg
    R1 = {"4852": 3.0747} # R theta
    R2 = {"4852": 2.9625} # R phi
    center = {"4852":[ 69.60399862271909, 31.40032983828514]}
    #todo: activate a flag for xy
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

    if hascam:# and (not dryrun):
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
    if xyflag:
        movetable = read_xytable(movetablefn)
    else:
        movetable = read_movetable(movetablefn)
    
    print(f" Loop positioner size: {len(movetable)}" )

    x_here = None
    y_here = None

    for i, imount in enumerate(mounttable):
        if imount !=0:  
            sys.exit("NotImplemented: Only position 0 for mount is allowed now")
        else: 
            print(f"starting positioners loop for MOUNT in {imount}")
            movemount(imount)        

        for j, imove in enumerate(movetable):

            #MOVE THIS BECAUSE OF ROWS
            if not xyflag:
                mvlabel = time.strftime("%Y%m%d-%H%M%S")
                ihash = np.random.randint(100000,1000000-1 )
                mvargs = f"{imove[0]} {imove[1]} {imove[2]} {imove[3]} {ihash}"
            else:
                x_tgt, y_tgt = imove[0], imove[1]

            # if not xyflag:
            #     first_xymove = False  

            if (x_here is None) or (y_here is None):
                print("Find current position XY")
                first_xymove = True

                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
                centroids = get_spot(f"{mvlabel}.fits", f"sbigpics/{session_label}", verbose=False)
                # todo: should be position-wise
                # TODO: assuming a single positioner    
                x_here = centroids['x'][0] 
                y_here = centroids['y'][0]                 

                xc, yc = center[posid] # todo this should be outside the loop
                # import pdb; pdb.set_trace()
                xpos, ypos = xylib.refpix2pos(pix2mm, xc,yc, x_here, y_here)
                print(f"curpos: {xpos} {ypos}")


            mvlabel = time.strftime("%Y%m%d-%H%M%S")
            ihash = np.random.randint(100000,1000000-1 )



            rows = xylib.calc_movetables(hardstop_ang[posid],
                                R1[posid],R2[posid], 
                                xpos,ypos, float(x_tgt),float(y_tgt))
            #H,R_theta,R_phi,x0,y0,x,y)
            for irow in rows: 
                mvlabel = time.strftime("%Y%m%d-%H%M%S")
                ihash = np.random.randint(100000,1000000-1 )

                _dir, _motor, _ang = irow
                ispeed = 'cruise' # todo : change if correction move
                mvargs = f"{_dir} {ispeed} {_motor} {_ang} {ihash}"
                print(mvargs)
                send_posmove(mvargs, verbose=True)    
                sys_status = confirm_move(ihash, get_remotehash(sh))


            # sys.exit(0)


            # if haspos: # and not xyflag: 
            #     # Send and Confirm
            #     send_posmove(mvargs, verbose=True)    
            #     sys_status = confirm_move(ihash, get_remotehash(sh))


            if not sys_status:
                sys.exit(1)
            elif not dryrun:
                # 1. get_pic
                # 2. __analyze pics
                # 3. __sanity checks related to position
                #       - there's a pic
                #       - the spot was detected
                #       - next move is not passing physical limits
                get_picture(cam, mvlabel, rootout=picpath, dryrun=dryrun)
                centroids = get_spot(f"{mvlabel}.fits", f"sbigpics/{session_label}", verbose=False)
                
                # todo: Adapt for multiple 
                x_here = centroids['x'][0] 
                y_here = centroids['y'][0] 
                xpos, ypos = xylib.refpix2pos(pix2mm, xc,yc, x_here, y_here)

                # TODO check_precision(pix2mm, x_here, y_here, x_tgt, y_tgt )


                print(x_here, y_here)
                # TODO
                if xyflag:
                    jmove = [_dir, ispeed, _motor, _ang]
                write_db(session_label, imount, mvlabel, posid, jmove, centroids, 
                         dbname=dbname)
                # placeholder: _last_position = []
                # sanity_check_for_phys_lim(_last_position, next_pos, arccenter_posid, returns:sys_status)


    if hascam and (not dryrun):
        cam.close_camera()
    # todo generate a log!!!


   