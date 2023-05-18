# um_scripts
Repo of scripts and code for the DESI telescope simulator - now in the Soares Santos lab organization 2.1!

# Files
`connect4.py` : controls the execution of sequence (connection, remote_config, camera, mount, positioner, spotfinder, write_database) 
Usage:
```
[msdos@msslab01 ~/um_scripts]$ python3 connect4.py -c conf/getcurrent.ini
```

`fao_seq.py`: moves the positioners and reads args and remote_conf 
    installed @ PBOX: /home/msdos/pclight/trenz
    
`run_ang.py`: formerly connect5.py - main movement code (!)

`run_xytest2.py`: more functions to run an xy test - main movement code (!)
    
`XYtest.ipynb`: code to translate between angular arm positions and xy coords, includes functions to switch between coordinate systems and to calculate move tables
    also includes some code and plots to check that this math is working
    
`XYtestMovesFunction.ipynb`: essentially the same as XYtest, but with a lot of the excess code removed

`xylib.py`: library built off of XYtest, all the functions to translate coordinates

`fits look.ipynb`: look at a fits (image) file

`gen_movetable.ipynb`: create a movetable and check plot to make sure it is safe

`read_database.ipynb`: various examples of looking at data from the database

`fao_seq.py`: moves the positioners and reads args and remote_conf

`regions.reg`: used for spotfinder code, probably doesn't need to be here

`tsmount.py`: mount test move (?)

`write_data.ipynb`: for writing data to the database (?)

`Check_a_picture.ipynb` and `TakePicsAndPlot.ipynb`: exactly what they sound like

# Pipeline
1. Install positioners:
    After manual install, get calibrations for the positioner
     - home_theta
     - home_phi
     - center of theta arc (block's hole center)

2. For sequence
    a. Check the position of positioners and spotfinder, WITHOUT moving,  using*:
        `python3 run_ang.py -c conf/getcurrent.ini` 
        
    b.set: cruise_speed, spinramp, dev_bb

    *`run_ang.py` was called `connect5.py`

# Sync to remote 
$ scp remote/fao_seq.py msdos@141.211.99.73 (###141.211.96.72):/home/msdos/pclight/trenz
141.211.99.73

# Calibration 01-2023
2023-Feb-10
weighted: 28.2989 +/- 0.0271
weighted: 0.035337 +/- 0.000034

mm2pix_weighted: 28.2856 +/- 0.0498
pix2mm_weighted: 0.035354 +/- 0.000062

# Notes
1. dev_by_bus = {'can22':[4852]}

2. Estimated centers 01-2023
```xc, yc, rc = 1973, 885, 190
```
limit_phi= 180 +epsilon
limit_theta  = 360 + epsilon

2. a) estimated HOME:
 = 1978.803038, 882.274307 # 2023 -03-15
   = 1969.7230055849757 , 880.7183990108621  # 2023-02-15 late
    = 1962.1128441626051,  887.1141524237803 # 2023-02-15
xh, yh = 1983.056226337495 880.2326884811561 # 2023-02-07

    b)  opened phi arm hardstop = 1808.8754329 , 946.131032 
                                 #1806.361327104717 , 939.2073449651402


2. b RAMP angle correction 
gear_ratio = (46/14 + 1)**4
defcruise*(defcruise+1)* deframp / 20/ gear_ratio
defcruise =33, deframp=12 => ramp_angle = 1.995

3. Db header
 session mount movelabel 4852 cw cruise motor angle xpix ypix peaks fwhm

    
    # CAM UP

    cf.home(cem120)

    # Position 7 - R DEITA DE LADO com o vetor z apontando pro pc  eixo 0 = z up
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 

    # Position 8 - U   -> HOME->7->8 CAM UP  eixo 1  = x
    cf.move_90(cem120, 0., 1, 1)


    # cam DOWN
    cf.home(cem120)

    # Position 7 - R DEITA DE LADO com o vetor z apontando pro pc  eixo 0 = z up
    cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]-324000)%1296000, 0, 1) 

    # Position 8b  test HOME->7->8b CAM DOWN  eixo 1 = zbox; sign 0 = right hand
    cf.move_90(cem120, 0., 1, 0)

#  Turn light on
pbset FIDUCIALS{'can22':{4000:100}} # fiducial
pbset FIDUCIALS{'can22: {65535:100}} # positioner


# speed reference
https://docs.google.com/spreadsheets/d/1OwOr4Te70YioZWkdfp3y4ym-RlW-FjWO/edit#gid=1049833235

# mount codes for ICS: 
/data/common/software/products/tsmount-umich 
