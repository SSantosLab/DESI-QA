import numpy as np


def transform(H,R_theta,R_phi,x,y,safe=True): 
    """
    Function to return theta and phi angles based on x and y coordinates relative to center, where center is defined as (0,0), with units in mm's
    
    INPUTS    
    H; The hardstop angle, given in degrees
    R_theta; Radius of theta, given in mm
    R_phi; Radius of phi, given in mm
    x; current x position, given in mm
    y; current x position, given in mm
    safe=True; A flag for testing. When set to False, will override patrol radius assertion. Only used for testing moves near R_theta+R_phi. DO NOT SET TO True IF PERFORMING DESI MOVES

    OUTPUTS
    theta; The angle of the theta move
    phi; the angle of the phi move
    
    NOTES
    I don't like this function name - it would be more intuitively named 'get_angle' or something similar
    """
    
    # For (0,0), we should home the positioner to theta=0 and phi=0
    if x==0 and y==0:
        theta,phi = 0,0
    
    # Redefining the hardstop angle in radians, used for numpy functions
    H = H*np.pi/180
    
    # Length of desired move from the center
    d = np.sqrt(x**2 + y**2) 
    
    # Assertion to protect against moves outside of the patrol radius
    # See documentation for 'safe' above
    if safe:
        assert d <= R_theta+R_phi,\
             f"d> out of reach {d}>{R_theta}+{R_phi}"
    
    # Defining components of the hardstop angle
    a = d*np.cos(H)
    b = d*np.sin(H)
    r = np.sqrt((x-a)**2 + (y-b)**2) # Magnitude of vector between (x,y) and (a,b)
    # FLAG
    alpha = 2*np.arcsin(r/(2*d)) # I don't now what this angle represents. I think it's supposed to be the angle between d and r/2, where r=(x,y)? Need to verify math
    
    # FLAG - I am fairly sure this sequence is to find the angle for phi - 
    arg = (R_phi**2 + R_theta**2 - d**2)/(2*R_theta*R_phi) # 
    arg = np.round(arg, 8) # avoid numerical residuals for x,y -> Physical Length
    try:
        phi = np.arccos(arg)
    except Exception as err:
        print(err, f"arg={arg}")
    d_theta = np.arcsin(R_theta*np.sin(phi)/d)

    # All of the below assumes the hardstop angle H is in the range [-180,180] - maybe should add while loop for abs(H)>180? to ensure we catch any edge cases?
    
    # FLAG - verify math
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
    
    # Converting theta and phi from radians to degrees
    theta = theta*180/np.pi
    phi = phi*180/np.pi
    
    # Reducing theta to a move within [0,360] 
    while theta>360:
        theta = theta - 360
    
    return theta,phi

def refpix2pos(pix2mm, xc,yc, xpix, ypix):
    """
    Change of reference frame from pix (spotfinder) to
    positioners (mm in the center of theta arc)
    
    INPUTS
    pix2mm; mm-pixel conversion factor
    xc; x coordinate of origin in mm
    yc; y coordinate of origin in mm
    xpix; x coordinate of position in pixels
    ypix; y coordinate of position in pixels
    
    OUTPUTS
    x2; New x coordinate in mm
    y2; New y coordinate in mm
    
    NOTES
    """
    xmm, ymm = xpix*pix2mm, ypix*pix2mm
    x2,y2 = xmm-xc, ymm-yc
    return x2,y2

def prepare2xy(x0,y0, x1,y1):
    """
    Simple coord change y-> -y, for consistency in fiposcontroller
    
    INPUTS
    x0; first x coordinate
    y0; first y coordinate
    x1; second x coordinate
    y1; second y coordinate
    
    OUTPUTS
    u0; first x coordinate
    v0; first y coordinate, flipped through axis
    u1; second x coordinate
    v1; second y coordinate, flipped through axis
    
    NOTES
    
    """
    u0, v0 = x0, -y0
    u1, v1 = x1, -y1
    return u0, v0, u1, v1


#version of the function to give moves
def calc_moves(H,R_theta,R_phi, xc, yc, x0_inp,y0_inp,x_inp,y_inp,buffer=0.1):
    """
    Function to output moves in theta and phi 
    
    INPUTS
    H; The hardstop angle, given in degrees
    R_theta; Radius of theta, given in mm
    R_phi; Radius of phi, given in mm
    xc; x coordinate of center of positioner, given in mm
    yc; y coordinate of center of positioner, given in mm 
    x0_inp; x coordinate of intial positioner position, given in mm
    y0_inp; y coordinate of intial positioner position, given in mm
    x_inp; x coordinate of final positioner position, given in mm
    y_inp; y coordinate of final positioner position, given in mm
    buffer=0.1; distance between R_theta+R_phi that is a 'no-go' zone, gives a
    
    OUTPUTS
    theta_cw; The theta cw move from the 
    theta_ccw;
    phi_cw;
    phi_ccw;

    NOTES
    BEFORE YOU PASS: 
     - Change the center of coordinate system to the center of theta arc, so that xc, yc, x0_inp, y0_inp, x_inp, and y_inp are in the same coordinate frame
     - Use mms for units, not pixels
     - Invert the y axis, keep the x axis intact - FLAG - this is done with prepare2xy function, so don't actually do this I think?
     
    TODO: Round the number for close to Zero!
    """
    x0,y0,x,y = prepare2xy(x0_inp,y0_inp,x_inp,y_inp)

    assert np.hypot( x, y) - buffer <= R_theta+R_phi, \
            f'out of reach move {np.hypot(x,y)} >{R_theta+R_phi}'
    assert np.hypot(x0,y0) - buffer <= R_theta+R_phi, \
            f'out of reach move {np.hypot(x0,y0)<=R_theta+R_phi}>{R_theta+R_phi}'
    
    # Using transform, compute where you are
    theta0,phi0 = transform(H,R_theta,R_phi,x0,y0,safe=True)
    #Using transform, compute where you want to be 
    theta,phi = transform(H,R_theta,R_phi,x,y,safe=True)
    # Take the difference
    delta_theta = theta-theta0
    delta_phi = phi-phi0
    
    # logic for theta
    if delta_theta >= 0:
        theta_cw = abs(delta_theta)
        theta_ccw = 0
    elif delta_theta < 0:
        theta_cw = 0
        theta_ccw = abs(delta_theta)

    # logic for theta        
    if delta_phi >= 0:
        phi_cw = abs(delta_phi)
        phi_ccw = 0
    elif delta_phi < 0:
        phi_cw = 0
        phi_ccw = abs(delta_phi)

    # I don't know why this is here -- testing reasons? I will remove
#     theta_cw

    return theta_cw,theta_ccw,phi_cw,phi_ccw

#version that does move tables
def calc_movetables(H,R_theta,R_phi,x0,y0,x,y): 
    """
    Function to output moves in theta and phi (pos being cw)
    if speed=='cruise' and angle < 2* ramp, it is not possible to do a move
    """

    x0,y0,x,y = prepare2xy(x0,y0,x,y)

    #where you are
    theta0,phi0 = transform(H,R_theta,R_phi,x0,y0,safe=False)
    #where you want to be 
    theta,phi = transform(H,R_theta,R_phi,x,y,safe=False) # returning nan?
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
