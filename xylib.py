import numpy as np


def transform(H,R_theta,R_phi,x,y): 
    """
    Function to output theta and phi angles based on coordinates relative to center 
    """
    
    if x==0 and y==0:
        theta = 0
        phi = 0
    
    H = H*np.pi/180
    d = np.sqrt(x**2 + y**2)
    assert d <= R_theta+R_phi,\
         f"d> out of reach {d}>{R_theta}+{R_phi}"
    a = d*np.cos(H)
    b = d*np.sin(H)
    r = np.sqrt((x-a)**2 + (y-b)**2)
    alpha = 2*np.arcsin(r/(2*d))
    #print(alpha*180/np.pi)
    arg = (R_phi**2 + R_theta**2 - d**2)/(2*R_theta*R_phi)
    arg = np.round(arg, 8) # avoid numerical residuals for x,y -> Physical Length
    try:
        phi = np.arccos(arg)
    except Exception as err:
        print(err, f"arg={arg}")
    d_theta = np.arcsin(R_theta*np.sin(phi)/d)

    
    if abs(H) > np.pi/2:
        if y>(b/a)*x:
            theta = alpha + d_theta
        if y<=(b/a)*x:
            theta = ((2*np.pi)-alpha) + d_theta
    else:
        if y>(b/a)*x:
            theta = ((2*np.pi)-alpha) + d_theta
        if y<=(b/a)*x:
            theta = alpha + d_theta 
    
    theta = theta*180/np.pi
    phi = phi*180/np.pi
    
    if theta>360:
        theta = theta - 360
    
    return theta,phi

def refpix2pos(pix2mm, xc,yc, xpix, ypix):
    """
    Change of reference frame from pix (spotfinder) to
    positioners (mm in the center of theta arc)
    """
    xmm, ymm = xpix*pix2mm, ypix*pix2mm
    return xmm-xc, ymm-yc

def prepare2xy(x0,y0, x1,y1):
    """
    Simple coord change y-> -y
    for consistency in fiposcontroller
    """
    u0, v0 = x0, y0
    u1, v1 = x1, y1
    return u0, -v0, u1, -v1


#version of the function to give moves
def calc_moves(H,R_theta,R_phi, xc, yc, x0_inp,y0_inp,x_inp,y_inp):
    """
    Function to output moves in theta and phi 
    BEFORE YOU PASS: 
     - Change the center of coordinate system to the center of theta arc
     - Invert the y axis, keep the x axis intact
     - 
    TODO: Round the number for close to Zero!
    """
    x0,y0,x,y = prepare2xy(x0_inp,y0_inp,x_inp,y_inp)

    assert np.hypot( x, y) -0.1 <= R_theta+R_phi, \
            f'out of reach move {np.hypot(x,y)} >{R_theta+R_phi}'
    assert np.hypot(x0,y0) -0.1 <= R_theta+R_phi, \
            f'out of reach move {np.hypot(x0,y0)<=R_theta+R_phi}>{R_theta+R_phi}'
    
    
    #where you are
    theta0,phi0 = transform(H,R_theta,R_phi,x0,y0)
    #where you want to be 
    theta,phi = transform(H,R_theta,R_phi,x,y)
    #difference
    delta_theta = theta-theta0
    delta_phi = phi-phi0
    #logic
    if delta_theta >= 0:
        theta_cw = abs(delta_theta)
        theta_ccw = 0
    elif delta_theta < 0:
        theta_cw = 0
        theta_ccw = abs(delta_theta)
    if delta_phi >= 0:
        phi_cw = abs(delta_phi)
        phi_ccw = 0
    elif delta_phi < 0:
        phi_cw = abs(delta_phi)
        phi_ccw = 0
    
    theta_cw
    return theta_cw,theta_ccw,phi_cw,phi_ccw

#version that does move tables
def calc_movetables(H,R_theta,R_phi,x0,y0,x,y): 
    """
    Function to output moves in theta and phi (pos being cw)
    if speed=='cruise' and angle < 2* ramp, it is not possible to do a move
    """

    #fao 2
    H = H


    #where you are
    theta0,phi0 = transform(H,R_theta,R_phi,x0,-y0)
    #where you want to be 
    theta,phi = transform(H,R_theta,R_phi,x,-y) # returning nan?
    #difference
    delta_theta = theta-theta0
    delta_phi = phi-phi0
    
    # fao:
    angle1 = 0
    angle2 = 0


    #optional logic
    if delta_theta >= 0:
        direc1 = "cw"
        motor1 = "theta"
        angle1 = abs(delta_theta)
    elif delta_theta < 0:
        direc1 = "ccw"
        motor1 = "theta"
        angle1 = abs(delta_theta)
    if delta_phi >= 0:
        direc2 = "cw"
        motor2 = "phi"
        angle2 = abs(delta_phi)
    elif delta_phi < 0:
        direc2 = "ccw"
        motor2 = "phi"
        angle2 = abs(delta_phi)

    rows=[]
    if angle1 >= 1e-2:
        rows.append([direc1, motor1, np.round(angle1, 8)])
    if angle2 >= 1e-2:
        rows.append([direc2, motor2, np.round(angle2, 8)])

    return rows # direc1,motor1,angle1,direc2,motor2,angle2


def xy2move(calc):
    """
    Interface between calc_moves and command 
    to fiposcontroller
    Input:
    calc: (list):
        output from calc_moves with
        [th_cw, th_ccw, ph_cw, ph_ccw]
    Returns:
        list with direction(str), motor(str), angle (float)
    """

    _dir = ['cw','ccw']
    ith = [0 if calc[0] > calc[1] else 1][0]
    iph = [0 if calc[2] > calc[3] else 1][0]
    theta = calc[ith]
    phi  = calc[iph+2]
    row = []
    if abs(theta) >=1e-2:
        row.append([_dir[ith], 'theta', theta])
    if abs(phi) >=1e-2:
        row.append([_dir[iph], 'phi', phi])
    return row
    



if __name__=="__main__":

    Testme=True
    # DEPREACTED TRUTH
    TRUTH=[['ccw', 'theta', 77.99999999999974],
        ['ccw', 'theta', 78.00000000000028],
        ['ccw', 'theta', 77.99999999999999],
        ['ccw', 'theta', 78.00000000000004],
        ['cw', 'theta', 282.00000000000006]]

    xc, yc=-100000,1000
    R1 = 3.
    R2 = 3.
    hardstop = 167 #deg
    pix2mm = 0.035

    # Test_targets:
    # ALWAYS in mm, ALWAYS in POS reference frame
    aa = -np.linspace(-90, 300, 6) 
    xtst = 3*np.cos(np.deg2rad(aa))  
    ytst = 3*np.sin(np.deg2rad(aa))

    # simulate spotfinder result 
    xtstpix = (xtst + xc)/pix2mm
    ytstpix = (ytst + yc)/pix2mm 

    j_row = 0
    for i, xy in enumerate(np.c_[xtst[:], ytst[:]]):
        ix, iy = xy
        if i>0:
            xhere,yhere = refpix2pos(pix2mm, xc, yc,xtstpix[i-1], ytstpix[i-1])

            rows2 = calc_movetables(167,R1,R2, xhere,yhere,xtst[i],ytst[i])
            print(rows2)
            # print(xc, yc,xtstpix[i-1], ytstpix[i-1])
            calc = calc_moves(167, R1, R2, 
                              xc, yc,
                              xhere,yhere, 
                              xtst[i],   ytst[i])

            row = xy2move(calc)
            for j, j2 in zip(row, rows2):
                print(j)
                print(j2)
                if Testme:
                    assert j[:2] == TRUTH[j_row][:2],\
                           f"ERROR: {j} not equal {TRUTH[j_row]}"
                    assert np.isclose(j[2], TRUTH[j_row][2]), \
                           f"Not close: {j[2]}, {TRUTH[j_row][2]}"
                j_row+=1

    calc_movetables(hardstop_ang[posid], R1[posid], R2[posid], 6.0372, 0, 0, 6.0372)\
     == [['ccw', 'theta', 90]]