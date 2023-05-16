import sys
sys.path.append('/home/msdos/pclight/bbb')

import fiposcontrol;fipos = fiposcontrol.FiposControl(['can22'])


dev_by_bus = {'can22':[1316]}; fipos.stay_alive(dev_by_bus)


print(fipos.get_fw_version(dev_by_bus))

# fipos.move(dev_by_bus)
# fipos.execute_movetable(dev_by_bus)

       
