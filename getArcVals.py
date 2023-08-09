import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':120})

import sys
sys.path.append("/home/msdos/DESI-QA/desiarc-main/arc")
import find_center as fc
import sys
sys.path.append("/home/msdos/DESI-QA/")
import find_center as fc
from spotfinder import spotfinder

plt.rcParams.update({'figure.dpi':140})

def angle_between(c, p1, p2):
    # p1, p2 are points; c is center
    a = np.array(p1)
    b = np.array(c)
    c = np.array(p2)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def find_spot(fitsname, fitspath,  
              expected_spot_count=1, 
              regionsname='../regions.reg', 
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
    import os
    
    assert isinstance(fitsname, str)

    _ifn = f"{fitspath}/{fitsname}"
    if not (os.path.isfile(_ifn)):
        print(f"File not found \n{_ifn}")
#     if expected_spot_count != 1:
#         raise NotImplementedError("This mode wasn't tested here")
    try: 
        sf=spotfinder.SpotFinder(_ifn, expected_spot_count)
        centroids = sf.get_centroids(print_summary = verbose, 
                                     region_file=regionsname)
        if verbose: print(centroids)
    
    except Exception as err: #ignore photo if an error is raised
        print(f"{err}\nWarning: spot not found ")
        inval_number = np.nan
        return {  'peaks': [inval_number], 
                      'x': [inval_number], 
                      'y': [inval_number], 
                   'fwhm': [inval_number], 
                 'energy': [inval_number]} 
    return centroids


def collect_xy(files, picspath):
    # treating for single file
    if isinstance(files, str):
        files = [files]
    x2, y2 = [],[]
    for fullname in files:
        iname = fullname.split('/')[-1]
        _c = find_spot(iname, picspath)
        x2.append(_c['x'])
        y2.append(_c['y'])
    return x2, y2

def get_timecol(db):
    new = db.label.str.split("-", n=1, expand=True)
    new.columns = ['label', 'session']
    db['label'] = new['label']
    db.insert(1, "session", new['session'])
    db['session'] = pd.to_datetime(db['session'], format= '%Y%m%d-%H%M%S' )#.dt.time
    return 

def query_time(db, date=None, datemin=None, datemax=None):
    """
    First run get_timecol(database)
    datemin, datemax (str): e.g "2023-02-03 13:36:00"
    """
   
    if date is not None:
        return db['session'] == np.datetime64(date)
 
    dmin = [np.datetime64(datemin) if not None else None][0]
    dmax = [np.datetime64(datemax) if not None else None][0]
     
    cond1 = db["session"] >= dmin
    cond2 = db["session"] <= dmax
    if (datemin is not None) & (datemax is not None):
        return cond1 & cond2
    elif datemin is None:
        return cond2
    elif datemax is None: 
        return cond1 
    else:
        print("check datemin datemax fields")

def setplot(xc=0, yc=0, rmax=6.0, grid=True):
    """
    Setup for plot with args xc, yc, and Radius
    """
    plt.xlim(xc-rmax+.3, xc+rmax+.3)
    plt.ylim(yc+ rmax+.3,yc-rmax+.3)
    plt.ylim(yc- rmax , yc+rmax)
    plt.xlim(xc+rmax, xc-rmax)
    
    plt.plot(xc, yc, 'r+')
    plt.gca().set_aspect('equal')
    if grid:
        plt.grid(linestyle='--',linewidth=0.5)

def plot_circle(xc, yc, R, kwargs={}):
    _th = np.linspace(0, 2*np.pi)
    plt.plot(R* np.cos(_th)+xc, R*np.sin(_th)+yc, c='g', ls='--', lw=0.6)
    return 


def main(date1,date2):
    db = pd.read_csv("/home/msdos/DESI-QA/output/database.csv")
    get_timecol(db)

    fiddb = pd.read_csv("/home/msdos/DESI-QA/output/fiddb.csv");get_timecol(fiddb)

        # Add date1 and date2 as args
    

    dateStart = np.array([date1,date2],dtype='datetime64') # arcth time, arcph time, and arcth30small time
    dateEnd = dateStart+np.timedelta64(4,'m')

    m1 =  query_time(db, datemin=dateStart[0],datemax=dateEnd[0])
    m1 = (m1) & ( db['label'].str.contains('arcth') )  & (db['motor']=='theta')
    # m1 = (m1) & (db['direction']=='cw')

    m2 = query_time(db, datemin=dateStart[1],datemax=dateEnd[1])
    m2 = (m2) & ( db['label'].str.contains('arcph') )
    
    m1a =  query_time(fiddb, datemin=dateStart[0],datemax=dateEnd[0])
    m1a = (m1a) & ( fiddb['label'].str.contains('arcth') )  & (db['motor']=='theta')
    # m1 = (m1) & (db['direction']=='cw')

    m2a = query_time(fiddb, datemin=dateStart[1],datemax=dateEnd[1])
    m2a = (m2a) & ( fiddb['label'].str.contains('arcph') )

    # SM TODO - calculate pix2mm from fiducial database
    # Done, but want to do this in a more elegant way
    
    pix2mm = np.median(fiddb["pix2mm"][m1a])
    print("arcth pix2mm:",pix2mm)

    xc1, yc1, Rarc1 = [i*pix2mm for i in fc.get_circle(db[m1],)] # center and radius of theta arc

    pix2mm = np.median(fiddb["pix2mm"][m2a]) # Not reading properly - returning 
    print("arcph pix2mm:",pix2mm)
    xc2, yc2, R2 = [i*pix2mm for i in fc.get_circle(db[m2],)] # center and radius of phi arc

    # coordinates of center
    xc, yc= xc1, yc1
    R1 = np.hypot(xc2-xc1, yc2-yc1)

    # Capturing important pixels for hardstop angle - I think pix2mm is the incorrect naming convention, it should be millimeters to pixel? so long as we know moving forward - Sean
    hardStop = -angle_between(np.array([xc,yc]), np.array([xc+5,yc]), (xc2,yc2))

    return R1, R2, xc, yc, xc2, yc2, hardStop

def pix2mm(date1,date2):
    db = pd.read_csv("/home/msdos/DESI-QA/output/database.csv")
    get_timecol(db)

    fiddb = pd.read_csv("/home/msdos/DESI-QA/output/fiddb.csv");get_timecol(fiddb)

        # Add date1 and date2 as args
    

    dateStart = np.array([date1,date2],dtype='datetime64') # arcth time, arcph time, and arcth30small time
    dateEnd = dateStart+np.timedelta64(4,'m')

    m1 =  query_time(db, datemin=dateStart[0],datemax=dateEnd[0])
    m1 = (m1) & ( db['label'].str.contains('arcth') )  & (db['motor']=='theta')
    # m1 = (m1) & (db['direction']=='cw')

    m2 = query_time(db, datemin=dateStart[1],datemax=dateEnd[1])
    m2 = (m2) & ( db['label'].str.contains('arcph') )
    
    m1a =  query_time(fiddb, datemin=dateStart[0],datemax=dateEnd[0])
    m1a = (m1a) & ( fiddb['label'].str.contains('arcth') )  & (db['motor']=='theta')
    # m1 = (m1) & (db['direction']=='cw')

    m2a = query_time(fiddb, datemin=dateStart[1],datemax=dateEnd[1])
    m2a = (m2a) & ( fiddb['label'].str.contains('arcph') )

    # SM TODO - calculate pix2mm from fiducial database
    # Done, but want to do this in a more elegant way
    
    a,b = np.array(fiddb["pix2mm"][m1a]),np.array(fiddb["pix2mm"][m2a])
    a = np.concatenate((a,b))

    pix2mm = np.median(a)
    return pix2mm