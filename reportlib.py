""" Report Library
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Flexible imports and pathing
parent = os.getcwd().split("/DESI-QA/")[0] 
root = f"{parent}/DESI-QA"
sys.path.append(f"{root}/")
sys.path.append(f"{root}/output/figures/")

from spotfinder import spotfinder
from xylib import calc_movetables as cm

try:
    sys.path.append(f"{parent}/desiarc/arc")
    import find_center as fc
except:
    sys.path.append(f"{root}/desiarc-main/arc")
    import find_center as fc
    

plt.rcParams.update({'figure.dpi':120})

figPath = f'{root}/output/figures/LinPhiTests/50ArcTest/'


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

def angle_between_extended(c, p1, p2):
    # p1, p2 are points; c is center
    # To be used for angles greater than 180 degrees. If less than, you can use either this function or angle_between
    a = np.array(p1)
    b = np.array(c)
    c = np.array(p2)
    ba = a - b
    bc = c - b
    angle = np.degrees(np.arctan2(bc[1], bc[0])-np.arctan2(ba[1], ba[0]))
    if angle<0:
        angle = angle+360
    return angle

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
    
def getSessionsArc(df):
    '''
    Function to get indices of single arc sequences
    Returns session_ranges array, which contains all indices where a new arcsequence session has started, and the termination of the full sequence (final move)
    '''
    session_ranges = np.array([-1],dtype=int)
    for i in range(len(df)-1):
        if (df.loc[i+1]['angle']==0) and (df.loc[i]['direction']=='ccw') and (df.loc[i+1]['direction']=='cw'):
            session_ranges = np.append(session_ranges,i)
    session_ranges = np.append(session_ranges,len(df)-1)
    return session_ranges

def getSessionsBacklash(df,motor):
    '''
    Function to get indices of single arc sequences during a backlash sequence
    Returns session_ranges array, which contains all indices where a new arcsequence session either started or stopped
    '''
    if motor=='phi':
        lims = [0,180]
    elif motor=='theta':
        lims = [0,360]
    else:
        print("Incorrect motor specifier")
        return -1
    session_ranges = np.array([-1],dtype=int)
    for i in range(len(df)-1):
        if (df.loc[i]['angle']==lims[1]) and (df.loc[i+1]['angle']==lims[0]):
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


def phi_centers(df,sessions,testType):
    '''
    Function to calculate the center of a given arcsequence
    Uses the mean value of the pix2mm values from that session to calculate the center point
    '''
    if testType=="RS":
        xc2, yc2, R2 = [i*df['pix2mm'].mean() for i in fc.get_circle(df)]
        pix2mm = df['pix2mm'].mean()
        length = len(df)
        xc2_arr,yc2_arr,R2_arr = np.repeat(xc2, length), np.repeat(yc2, length), np.repeat(R2, length)
        xc2_pix_arr, yc2_pix_arr = np.repeat(xc2/pix2mm, length), np.repeat(yc2/pix2mm, length)
    else:
        xc2_arr,yc2_arr,R2_arr,xc2_pix_arr,yc2_pix_arr,pix2mm_arr = np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)
        for j in range(len(sessions)-1):
            singleDF = getOneSession(df,j,sessions)
            xc2, yc2, R2 = [i*singleDF['pix2mm'].mean() for i in fc.get_circle(singleDF)]
            for k in range(getDiff(sessions,j)-1):
                pix2mm_arr = np.append(pix2mm_arr,singleDF['pix2mm'].mean())
                pix2mm = pix2mm_arr[-1] # FAO
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
        xc2_pix,yc2_pix = arcs['xc2pix'].loc[0],arcs['yc2pix'].loc[0]
        
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

    Args:

    '''
    session_means = np.array([],dtype=float)
    session_stds = np.array([],dtype=float)
    means = np.array([],dtype=float)
    stds = np.array([],dtype=float)
    for sessionNum in np.unique(df['ArcSession']):
        maskedDF = df[df['ArcSession']==sessionNum]
        session_means = np.append(session_means, np.repeat(np.mean(maskedDF[label]),len(df[df['ArcSession']==sessionNum])))
        session_stds = np.append(session_stds, np.repeat(np.std(maskedDF[label],ddof=1),len(df[df['ArcSession']==sessionNum])))
        means = np.append(means, np.mean(maskedDF[label]))
        stds = np.append(stds, np.std(maskedDF[label],ddof=1))
    return session_means,session_stds,means,stds

def toNumpy(series):
    '''
    function to return series in np.datetime64 format
    series is the df['move'] pandas series
    '''
    years = series.str[:4].to_numpy(dtype=int)
    months = series.str[4:6].to_numpy(dtype=int)
    days = series.str[6:8].to_numpy(dtype=int)
    hours = series.str[9:11].to_numpy(dtype=int)
    minutes = series.str[11:13].to_numpy(dtype=int)
    seconds = series.str[13:15].to_numpy(dtype=int)

    time=np.array([],dtype=np.datetime64)

    for j in range(len(years)):
        time = np.append(time,pd.Timestamp(year=years[j], month=months[j], day=days[j], hour=hours[j],minute=minutes[j],second=seconds[j]))
    return pd.Series(time)


def computeBacklash(df,sessions,testType=None,ramp=1.995):
    '''
    Function to compute backlash
    '''
    backlash = np.array([],dtype=float) # Initialize array
    if testType=="RS":
        # singleDF = getOneSession(df,j,sessions).reset_index(drop=True)
        backlash = np.append(backlash,np.nan)
        for k in range(len(df)-1):
            backlash = np.append(backlash,df.loc[k+1]['ObservedMove']-(df.loc[k+1]['angle']+2*ramp))
    else:
        for j in range(len(sessions)-1):
            singleDF = getOneSession(df,j,sessions).reset_index(drop=True)
            backlash = np.append(backlash,np.nan)
            for k in range(getDiff(sessions,j)-2):
                backlash = np.append(backlash,singleDF.loc[k+1]['ObservedMove']-(singleDF.loc[k+1]['angle']+2*ramp))
    return backlash

def getSessionsBacklashRS(df,motor):
    '''
    Function to get indices of single arc sequences using the RS theta sequence
    Returns session_ranges array, which contains all indices where a new arcsequence session has started, 
    and the termination of the full sequence (final move)
    '''
    session_ranges = np.array([-1,0],dtype=int)
    # for j in range(1,len(df)):
    #     if j%2==0:
    #         session_ranges = np.append(session_ranges,j)
    session_ranges = np.append(session_ranges,len(df)-1)
    return session_ranges

def importToDf(datapath,fidpath,testStart,testFinish,testType="arc",motor=None):
    
    '''
    Function to take path to database and fiducial database, and returned dataframe of combined data
    testype is one of 'arc', or 'backlash'
    '''
    
    # Importing different databases
    db = pd.read_csv(datapath)
    get_timecol(db)

    fiddb = pd.read_csv(fidpath)
    get_timecol(fiddb)

    # Because query_time uses label (not mvlabel or move), if initial selection is OK, then you can
    # join using db.insert(len(db.columns))
    mask1 = query_time(db, datemin=testStart,datemax=testFinish)
    mask2 = query_time(fiddb, datemin=testStart,datemax=testFinish)

    # Creating masks
    df = db[mask1].reset_index(drop=True)
    fiddf = fiddb[mask2].reset_index(drop=True)

    del db, fiddb

    # Inserting pix2mm and sigpix2mm
    if aligned(df,'move',fiddf,'mvlabel'):
        df.insert(len(df.columns),"fidx0",fiddf['x0'])
        df.insert(len(df.columns),"fidy0",fiddf['y0'])
        df.insert(len(df.columns),"fidx1",fiddf['x1'])
        df.insert(len(df.columns),"fidy1",fiddf['y1'])
        df.insert(len(df.columns),"fidx2",fiddf['x2'])
        df.insert(len(df.columns),"fidy2",fiddf['y2'])
        df.insert(len(df.columns),"fidx3",fiddf['x3'])
        df.insert(len(df.columns),"fidy3",fiddf['y3'])
        df.insert(len(df.columns),"pix2mm",fiddf['pix2mm'])
        df.insert(len(df.columns),"sigpix2mm",fiddf['sigpix2mm'])
        del fiddf
    else:
        print("Movelabels are not aligned - inspect your movelabels and try again")
        return -1

    # Add session labels
    print("Adding session labels for testType="+str(testType))
        
    if testType=='arc':
        # Find sessions for each arcsequence    
        sessions = getSessionsArc(df)

    elif testType=='backlasharc' and (motor=='theta' or motor=='phi'):
        #Find sessions for each arcsequence
        sessions = getSessionsBacklash(df,motor)

    elif testType=='RS' and (motor=='theta' or motor=='phi'):
        #Find sessions for each arcsequence
        sessions = getSessionsBacklashRS(df,motor)

    else:
        print("Incorrect specifier for testType or motor\n testType must be either \'arc\' or \'backlasharc\'\n motor must be either \'theta\' or \'phi\'")
        return -1

    # Make an arcnum session column and add it to the df
    sessionLabels = makeSessionLabels(sessions)

    print(sessionLabels)

    # Add session labels to the df
    df.insert(len(df.columns),'ArcSession',sessionLabels)

    #Find centers for each arcsequence
    xc2_arr,yc2_arr,R2_arr,xc2_pix_arr,yc2_pix_arr = phi_centers(df,sessions,testType)

    # Store centers in df
    df.insert(len(df.columns),'xc2mm',xc2_arr)
    df.insert(len(df.columns),'yc2mm',yc2_arr)
    df.insert(len(df.columns),'R2mm',R2_arr)
    df.insert(len(df.columns),'xc2pix',xc2_pix_arr)
    df.insert(len(df.columns),'yc2pix',yc2_pix_arr)

    # Change datatype of df['move'] column
    df['move'] = toNumpy(df['move'])

    # Calculate x and y positions in mm
    x_mm = df['xpix']*df['pix2mm']
    y_mm = df['ypix']*df['pix2mm']

    # Insert x and y positions to df
    df.insert(12,'x_mm',x_mm)
    df.insert(13,'y_mm',y_mm)

    # Insert mount config into df in string form
    df.insert(len(df.columns),'MountConfiguration',set_MountConfig_String(df))

    # Compute alpha individually
    alpha_arr, req_arr, obs_arr = getAlphas(df)

    # Insert alpha to df
    df.insert(len(df.columns),'Alpha',alpha_arr)
    df.insert(len(df.columns),'RequestedMove',req_arr)
    df.insert(len(df.columns),'ObservedMove',obs_arr)

    # Compute mean and std of alpha of each session
    mean_alpha_session,std_alpha_session, mean_alpha, std_alpha= getMeans(df)

    # Insert alpha session mean and std to df
    df.insert(len(df.columns),'MeanAlpha',mean_alpha_session)
    df.insert(len(df.columns),'StdAlpha',std_alpha_session)

    if testType=='backlash':
        # Calculate backlash
        backlash = computeBacklash(df,sessions)

        # Add backlash to df
        df.insert(len(df.columns),'Backlash',backlash)

        return df
    elif testType=='RS':
        # Calculate backlash
        backlash = computeBacklash(df,sessions,testType=testType)

        # Add backlash to df
        df.insert(len(df.columns),'Backlash',backlash)

        return df
    else:
       return df

