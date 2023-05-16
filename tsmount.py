"""
tsmount.py"""
import sys
sys.path.append('/data/common/software/products/tsmount-umich/python')
import cem120func as cf
import time 


if __name__=="__main__":
	# Setup mount
	cem120 = cf.initialize_mount("/dev/ttyUSB0")
	cf.set_alt_lim(cem120, -89)
	cf.slew_rate(cem120, 9)
	cf.home(cem120)


	# Mount move
	# cf.move_90(cem120, (cf.get_ra_dec(cem120)[0]+324000)%1296000, 0, 0)
	# time.sleep(2)