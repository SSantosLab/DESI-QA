""" Working for a fiducial positioner + fiducial
$ python3 backlight.py
"""
import fiposcontrol20 as fiposcontrol 

_positioner = 65535
_fiducial = 4852 # 4852 is the positioner id for the fiducial 

devbb = {'can22':[_fiducial, _positioner]}
fipos =  fiposcontrol.FiposControl(devbb.keys())
fipos.set_fiducial_duty(devbb,100) 

# TODO : gracefully exit







