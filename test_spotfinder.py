import numpy as np
from spotfinder import spotfinder


def get_spot(fitsname, fitspath,  
              expected_spot_count=1, # number of positioners+ 3 fiducial spots
              regionsname='regions.reg', 
              verbose=False):
    """spotfinder handler

    Args:
        fitsname (str): _description_
        fitspath (str): _description_
        expected_spot_count (int, optional): Number of positioners + 3 fiducial 
            spots. Defaults to 1.
        verbose (bool, optional): Defaults to False.

    Returns:
        centroids (dict): raw output from spotfinder
    """
    assert isinstance(fitspath, str), "fitspath should be a string"
    assert isinstance(fitsname, str), "fitsname should be a string"

    _ifn = f"{fitspath}/{fitsname}"

    if expected_spot_count > 5:
        raise NotImplementedError("multiple spots to be implemented")
    try: 
        sf=spotfinder.SpotFinder(_ifn, expected_spot_count)
        centroids = sf.get_centroids(print_summary = verbose, 
                                     region_file=regionsname)
        if verbose: print(centroids)
    
    except Exception as err: #ignore photo if an error is raised
        print("Warning: spot not found ")
        print(f"\terr: {err}")
        inval_number = np.nan
        return {  'peaks': np.array([inval_number]*expected_spot_count), 
                      'x': np.array([inval_number]*expected_spot_count), 
                      'y': np.array([inval_number]*expected_spot_count), 
                   'fwhm': np.array([inval_number]*expected_spot_count), 
                 'energy': np.array([inval_number]*expected_spot_count)} 
    centroids = { k:np.array(v) for k,v in centroids.items()}
    return centroids

def select_fidregion(centroids, xmin=1036, xmax=1216, ymin=794, ymax=974):
    """_summary_
    Args:
        centroids (dict):
    Returns:
        mask  (np.array):         
    """
    fidregx = np.array([xmin, xmax])
    fidregy = np.array([ymin, ymax])
    mask = (centroids['x'] > fidregx[0]) & (centroids['x'] < fidregx[1]) & \
           (centroids['y'] > fidregy[0]) & (centroids['y'] < fidregy[1])
    return mask

def get_spotpos(posid, centroids, reg=reg ):
    mask=  (centroids['x'] >= reg[posid]['x'][0]) & \
    (centroids['x'] <= reg[posid]['x'][1]) &\
    (centroids['y'] >= reg[posid]['y'][0]) & \
    (centroids['y'] <= reg[posid]['y'][1])
    cent = {k:v[mask] for k,v in centroids.items()}

    return cent

def get_xyfid(centroids, mask, fidlabel=[1,0,2,3]):
    """
    In unit of PIX
    """
    yorder = np.argsort(centroids['y'][mask])
    centroids['y'][mask][yorder]
    fidlabel = [1,0,2,3] # order of y fiducials now, from ymin to ymax
    xfid = centroids['x'][mask][yorder][fidlabel]
    yfid = centroids['y'][mask][yorder][fidlabel]
    return xfid, yfid


def get_pix2mm(xfid, yfid):
    """
    Self calibration of fiducials
    Assuming: 
        Fiducials are on
        In the vblock region 
    """
    # Blueprint of fiducials
    dphys = { (3,2):1,(3,1):2, (3,0):1.6, (2,1):1, (2,0):1,  (1,0):1.2 }
    
    pix2mm_arr = np.array([])
    for k, v in dphys.items():
        dd = np.hypot(xfid[k[0]] - xfid[k[1]], yfid[k[0]] - yfid[k[1]] )
        pix2mm_arr = np.append(pix2mm_arr, dphys[k]/dd)
    return pix2mm_arr.mean(), pix2mm_arr.std(ddof=1)


def test_fiddetection(fitsname = "test_fidpos.fits", 
                      fitspath = "tests/"):
    """
    Test extraction of fiducial
    """
    xtst = np.array([1103.82688023, 1136.62107098, 1126.25603187, 1115.81365772])
    ytst = np.array([867.42444578, 858.47171232, 884.80090262, 911.07253092])
    
    import os
    assert os.path.isfile(f"{fitspath}/{fitsname}"), "Test File not found!"
    
    centroids = get_spot(fitsname, fitspath, 
                         expected_spot_count=5, 
                         verbose=False)
    
    fidmask = select_fidregion(centroids)

    xfid, yfid = get_xyfid(centroids, fidmask)
    
    assert np.allclose(xtst, xfid), 'Error in xfid'
    assert np.allclose(ytst, yfid), 'Error in yfid'
    print("Passed")

if __name__=="__main__":
    
    fitsname = "test_fidpos.fits"
    fitspath = "tests/"
    
    # Get fiducial positions:
    centroids = get_spot(fitsname, fitspath, 
                         expected_spot_count=5, 
                         verbose=False)
    fidmask = select_fidregion(centroids)

    xfid, yfid = get_xyfid(centroids, fidmask)
    print("\n\nOrdered Fiducial Reference [Jun 21st, 2023]:")
    print("    XPIX\t  YPIX")
    [ print(i, f"{px:.4f}\t{py:.4f}") for i, (px, py) in enumerate(zip(xfid, yfid))]
    print("")
    pix2mm, sigpix2mm = get_pix2mm(xfid, yfid)
    print(pix2mm, sigpix2mm, end='\n\n\n')
    
    # Unit test:
    test_fiddetection()
 
    
