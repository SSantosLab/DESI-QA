"""Test Stand Library
"""
import os 

def write_fiddb(session_label, mvlabel, xfid, yfid, pix2mm, sigpix2mm, dbname="output/fiddb.csv"):
    """write fiducial database

    Args:
        session_label (float): _description_
        xfid (array or list): _description_
        yfid (array or list): _description_
        pix2mm (float): _description_
        sigpix2mm (float): _description_
        dbname (str, optional): _description_. Defaults to "output/fiddb.csv".

    """
    if not os.path.isfile(dbname):
        print(f"DB file not found. Initializing a new one at {dbname}")
        with open(dbname, 'w') as ndb:
            ndb.write("session,mvlabel,x0,y0,x1,y1,x2,y2,x3,y3,pix2mm,sigpix2mm\n")
            
    with open(dbname, 'a') as fdb:
        fdb.write(f"{session_label},{mvlabel}")
        for i in range(4):
            fdb.write(f"{xfid[i]:.6f},{yfid[i]:.6f},")
        fdb.write(f"{pix2mm:.6f},{sigpix2mm:.6f}\n")
    
    return None
        
        
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


def turnon_backlight(sh):
    """
    Turn on the backlight using paramiko ssh connection
    
    Args:
        sh (ssh): ssh connection
    Returns:    
        sin (stdin, str): stdin
        sout (stdout, str): stdout
        serr (stderr, str): stderr
    """
    _command = "python3 backlight.py"
    sin, sout, serr = sh.execute(_command)
    return sin, sout, serr