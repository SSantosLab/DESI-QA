'''
Program to take data for the fiducials and positioner, and return the important 
'''

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

if __name__=='__main__':

	import find_center as fc

	# Select data for fiducials



	# Fiducial calibration

	

	# Select data for positioner


	
	# Theta calculation

	xc1, yc1, Rarc1 = [i*pix2mm for i in fc.get_circle(db[m1],)] # center and radius of theta arc

	# Phi calculation

	xc2, yc2, R2 = [i*pix2mm for i in fc.get_circle(db[m2])] # center and radius of phi arc

	# R_theta calculation

	R1 = np.hypot(xc2-xc1, yc2-yc1)

	# Hardstop calculation

	hardStop = angle_between(np.array([xc,yc]), np.array([xc+5,yc]), (xc2,yc2))

	# Return important values

	return R1,R2,xc,yc,hardStop,pix2mm