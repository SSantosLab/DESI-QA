"""
execute sequence from args 
remote directory:  /home/msdos/pclight/trenz/

felipeaoli@gmail.com
"""
import argparse
import yaml
import sys
import time
import fiposcontrol

print("PB connection")
try: 
    import fiposcontrol
except Exception as err:
    print("PB exception: ", err)
    dryrun=True

class DryRun():
    #fake class
    def __init__(self):
        print("fiposcontrol not found. Dry run")

    class FiposControl():
        def __init__(self, somelist):
            pass
        

        def move(self, devbb, direction, speed, motor, angle ):
            print(devbb, direction, speed, motor, angle)

        def execute_movetable(self):
            print("faking exec movetable")
            time.sleep(2)
            print("done")
            pass

def turnon_backlight(brightness, fipos):
    _devbb = {'can22': [65535]}
    fipos.set_fiducial_duty(_devbb, brightness)
    return None 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", help="cw or ccw", choices=['cw', 'ccw'])
    parser.add_argument("speed", help="cruise or creep", choices=['cruise', 'creep'])
    parser.add_argument("motor", help="theta or phi", choices=['theta', 'phi'])
    parser.add_argument("angle", help="angle in degrees", type=float)
    parser.add_argument("hash", help="hash cmd identifier", type=str)

    args = parser.parse_args()

    print(args)
    cfgfile = "conf/test.yml"
    cfgfile = "/home/msdos/pclight/trenz/umconf/conf.yml"
    with open(cfgfile, 'r') as file:
        cfg = yaml.safe_load(file)

    dryrun = cfg.get('dryrun')
    thmin, thmax = cfg['th_lim']
    phmin, phmax = cfg['ph_lim']
    devbb = cfg['dev_by_bus']

    print(f"Not implemented Phys. Limits: {(thmin, thmax)} {(phmin, phmax)}")

    if dryrun:
        print("dryrun")
        fiposcontrol = DryRun()

    
 

    direction = args.direction
    speed = args.speed
    motor = args.motor
    angle =  args.angle
    cmdhash = args.hash


try: 
    fipos = fiposcontrol.FiposControl(['can22'])
    turnon_backlight(100, fipos)
    fipos.move(devbb,direction, speed, motor, angle)
    fipos.execute_movetable()
except Exception as err:
    print("PB Exception:", err)
    sys.exit()

# returning the hash if everything runs well:
cmdhash ='#UM'+str(cmdhash)+'fff'
fipos.gentle_exit()
print(cmdhash)
open("umhash.txt", "w").write(str(cmdhash)+'\n')
sys.exit()


