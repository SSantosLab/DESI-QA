import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.dpi':180})

import sys
sys.path.append("/home/msdos/DESI-QA/desiarc-main/arc")
sys.path.append("/home/msdos/DESI-QA/")
sys.path.append("/home/msdos/DESI-QA/output/figures/")
import find_center as fc
from spotfinder import spotfinder
from xylib import calc_movetables as cm

def mount_to_label_raw(tupl):
    '''
    Function to take in a tuple of rowdata, and convert it to a string label representing the configuration
    Currently only configured for positioner up, positioner down, horizon
    '''
    if tupl[0]*tupl[1]==0:
        return "Horizon"
    elif tupl[0]*tupl[1]>0:
        return "Pos Down"
    else:
        return "Pos Up"
    
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
    plt.xlim(xc-rmax+.3, xc+rmax+.3)
    plt.ylim(yc+ rmax+.3,yc-rmax+.3)
    plt.plot(xc, yc, 'r+')
    plt.gca().set_aspect('equal')
    if grid:
        plt.grid(linestyle='--',linewidth=0.5)
        
        
def get_dbxy(db,  datemin, datemax, label):
    mxy = db["label"].str.contains(label)
    mxy = (mxy) & (query_time(db, datemin=datemin, datemax=datemax))
    print(db.session[mxy].unique())
    dbxy = db[['xpix','ypix']][mxy].reindex()
    print(dbxy.shape)
    dbxy['xpos'] = dbxy["xpix"].values * pix2mm -xc
    dbxy['ypos'] = dbxy["ypix"].values * pix2mm -yc
    dbxy.set_index(np.arange(dbxy.shape[0]), inplace=True)
    return dbxy


def plot_circle(xc, yc, R, kwargs={},axs = None):
    _th = np.linspace(0, 2*np.pi)
    if axs !=None:
        axs.plot(R* np.cos(_th)+xc, R*np.sin(_th)+yc, c='green', ls='--', lw=0.6)
    else:
        plt.plot(R* np.cos(_th)+xc, R*np.sin(_th)+yc, c='green', ls='--', lw=0.6)
    return 


def plot_xy(label, datemin, datemax, fig,ax,movefn, title='', show=True,save=False,pathname=None):
    dbxy = get_dbxy(db, datemin, datemax,label=label)
    xytgt = pd.read_csv(f"../movetables/{movefn}", sep=" ", header=None)
    xytgt.columns = ['xpos', 'ypos']

    ax.scatter('xpos', 'ypos', data=dbxy, c='black', marker ='+', s=30)#c=dbxy.index,cmap='Blues', edgecolors='b')
    # # plt.colorbar()
    ax.scatter('xpos', 'ypos', data=xytgt, facecolors='none',edgecolors='r', s=10, c='red', lw=0.5,label='target')
    ax.set_xlabel('xpos (mm)')
    ax.set_ylabel('ypos (mm)')
    ax.set_title(title, fontsize=8)
#     ax.set_xlim(-8,8)
    plt.gca().set_aspect('equal')
    plt.legend(loc='lower left', fontsize=8)
    if save:
        plt.savefig(pathname,dpi=180)
    if show:
        plt.show()
    return dbxy, xytgt
    # plt.legend(loc='best', fontsize=7)
    
def plot_formatting(num=1):
    fig,ax = plt.subplots(1,int(num),figsize=[6,6*int(num)])
    plot_circle(0,0, rPositioner)
    plt.grid(lw=0.6, ls='--')
    plt.plot(-np.linspace(0, rPositioner* np.cos(hardstopAngle)), np.linspace(0, rPositioner* np.sin(hardstopAngle)), lw=1 , c='g', label='hardstop'  )

    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    return fig,ax

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


def get_moveDirection(label,ABS=False):
    '''
    Function to take xy positions from a movetable and 
    return the theta and phi moves in separate arrays
    
    Here, CW is defined as positive and CCW is defined as negative
    '''

    df = pd.read_csv("/home/msdos/DESI-QA/movetables/"+label+".txt",sep=" ",names=["x","y"])
    
    theta_arr = np.array([0]) # set initial move direction to 0, since it can vary based on previous position - possible place to revise in the future, if necessary
    phi_arr = np.array([0])

    for j in range(len(df)-1):
        current = (j+1)
        old = j

        x_old,y_old = (df["x"][old],df["y"][old])
        x_current,y_current = (df["x"][current],df["y"][current])

        row = cm(hardstopAngle,R1,R2,x_old,y_old,x_current,y_current)

        if len(row)==1:
            if row[0][1]=='theta':
                phi_arr = np.append(phi_arr,0)
                if row[0][0]=="cw":
                    theta_arr = np.append(theta_arr,row[0][2])
                else:
                    theta_arr = np.append(theta_arr,-row[0][2])
            elif row[0][1]=='phi':
                theta_arr = np.append(theta_arr,0)
                if row[0][0]=="cw":
                    phi_arr = np.append(phi_arr,row[0][2])
                else:
                    phi_arr = np.append(phi_arr,-row[0][2])
            else:
                print("Rows from movetable are incompatible with storage settings - please check movetable!")
                print("Row "+str(j))
                break
        elif len(row)==2:

            if row[0][0]=="cw":
                theta_arr = np.append(theta_arr,row[0][2])
            else:
                theta_arr = np.append(theta_arr,-row[0][2])

            if row[1][0]=="cw":
                phi_arr = np.append(phi_arr,row[1][2])
            else:
                phi_arr = np.append(phi_arr,-row[1][2])
        else:
            print("Rows from movetable are incompatible with storage settings - please check movetable!")
            print("Row "+str(j))
            break
    if ABS:
        theta_arr = np.abs(theta_arr)
        phi_arr = np.abs(phi_arr)
    return theta_arr,phi_arr

def aligned(df1,df1_label,df2,df2_label):
    '''
    Function to test if df1 and df2 are aligned, by using a pre-defined axis label of identical keys
    Returns True if df1 and df2 are aligned, and False if not
    '''
    if len(df1)!=len(df2): # If mismatched length, obviously not aligned
        return False
    else:
        for entry in range(len(df1)):
            if df1.loc[entry][df1_label]!= df2.loc[entry][df2_label]:
                return False
        return True
    
def getSessions(df):
    '''
    Function to get indices of single arc sequences
    Returns session_ranges array, which contains all indices where a new arcsequence session either started or stopped
    '''
    session_ranges = np.array([-1],dtype=int)
    for i in range(len(df)-1):
        if (df.loc[i]['direction']!=df.loc[i+1]['direction']) and (df.loc[i]['direction']=='ccw') and (df.loc[i+1]['direction']=='cw'):
            session_ranges = np.append(session_ranges,i)
    session_ranges = np.append(session_ranges,len(df)-1)
    return session_ranges

def getOneSession(df,index,sessions_arr):
    '''
    Function to return a dataframe of a single arc-sequence session
    '''
    return df.loc[sessions_arr[index]+1:sessions_arr[index+1]]

def getDiff(sessions,k):
    '''
    Small helper function to get the difference in the total number of moves in a given session
    '''
    return sessions[k+1]-sessions[k]+1


def phi_centers(df,sessions):
    '''
    Function to calculate the center of a given arcsequence
    Uses the mean value of the pix2mm values from that session to calculate the center point
    '''
    xc2_arr,yc2_arr,R2_arr,xc2_pix_arr,yc2_pix_arr,pix2mm_arr = np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)
    for j in range(len(sessions)-1):
        singleDF = getOneSession(df,j,sessions)
        xc2, yc2, R2 = [i*singleDF['pix2mm'].mean() for i in fc.get_circle(singleDF)]
        for k in range(getDiff(sessions,j)-1):
            pix2mm_arr = np.append(pix2mm_arr,singleDF['pix2mm'].mean())
            xc2_arr,yc2_arr,R2_arr = np.append(xc2_arr ,xc2), np.append(yc2_arr ,yc2), np.append(R2_arr ,R2)
            xc2_pix_arr, yc2_pix_arr = np.append(xc2_pix_arr, xc2/pix2mm), np.append(yc2_pix_arr, yc2/pix2mm)
    return xc2_arr,yc2_arr,R2_arr,xc2_pix_arr,yc2_pix_arr

def set_MountConfig_String(df):
    '''
    Function to set the mount configuration in string form, based on the mount angles
    '''
    labels = np.array([],dtype=str)
    for j in range(len(df)):
        labels = np.append(labels,mount_to_label_raw((df.loc[j]['mtang1'],df.loc[j]['mtang2'])))
    return labels

def makeSessionLabels(sessions):
    '''
    Function to make session labels based on sessions and the dataframe
    '''
    testArr = np.array([],dtype=int)
    for j in range(len(sessions)-1):
        testArr = np.append(testArr,np.repeat(j, getDiff(sessions,j)-1))
    return testArr

def getAlphas(df,backlash = 1.9,ramp=1.995):
    '''
    Function to return arrays of alpha, required move [degrees], and observed move [degrees]
    '''

    alpha_arr = np.array([],dtype='float16')
    req_arr = np.array([],dtype='float16')
    obs_arr = np.array([],dtype='float16')

    for k in range(max(df['ArcSession'])+1):
        arcs = df.loc[df['ArcSession']==k].reset_index(drop=True)
        xc2_pix,yc2_pix = arcs['xc2 pix'].loc[0],arcs['yc2 pix'].loc[0]
        
        alpha_arr = np.append(alpha_arr , np.nan)
        req_arr = np.append(req_arr , np.nan)
        obs_arr = np.append(obs_arr , np.nan)
        
        for j in range(len(arcs)-1):
            prev = j
            current = j+1

            x_previous, y_previous,direction_previous = arcs.loc[prev]['xpix'],arcs.loc[prev]['ypix'],arcs.loc[prev]['direction']
            x_current, y_current,direction_current = arcs.loc[current]['xpix'],arcs.loc[current]['ypix'],arcs.loc[current]['direction']

            if direction_previous!=direction_current or (k==0 and j==0) or j==0:
                alpha_arr = np.append(alpha_arr,angle_between((xc2_pix,yc2_pix),(x_previous, y_previous),
                                                              (x_current, y_current))/(arcs.loc[current]['angle']+2*ramp-backlash))
                req_arr = np.append(req_arr,arcs.loc[current]['angle']+2*ramp-backlash)

            else:
                alpha_arr = np.append(alpha_arr,angle_between((xc2_pix,yc2_pix),(x_previous, y_previous),(x_current, y_current))/(arcs.loc[current]['angle']+2*ramp))
                req_arr = np.append(req_arr,arcs.loc[current]['angle']+2*ramp)

            obs_arr = np.append(obs_arr, angle_between((xc2_pix,yc2_pix),(x_previous, y_previous),(x_current, y_current)))
    return alpha_arr, req_arr, obs_arr

def getMeans(df,label='Alpha'):
    '''
    Function to return means of each session, repeated N times in an array
    N = number of datapoints in a given arc
    '''
    session_means = np.array([],dtype=float)
    session_stds = np.array([],dtype=float)
    means = np.array([],dtype=float)
    stds = np.array([],dtype=float)
    for sessionNum in np.unique(df['ArcSession']):
        maskedDF = df[df['ArcSession']==sessionNum]
        session_means = np.append(session_means, np.repeat(np.mean(maskedDF[label]),len(df[df['ArcSession']==sessionNum])))
        session_stds = np.append(session_stds, np.repeat(np.std(maskedDF[label]),len(df[df['ArcSession']==sessionNum])))
        means = np.append(means, np.mean(maskedDF[label]))
        stds = np.append(stds, np.std(maskedDF[label]))
    return session_means,session_stds,means,stds
