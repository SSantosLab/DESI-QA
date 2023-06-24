import numpy as np
import sys
import run_ang as tst
sys.path.append('/data/common/software/products/tsmount-umich/python')
import cem120func as cf


# Starting the comm with the mount
cem120 = tst.start_mount()


input("Press Enter to continue or Ctrl+C to exit...")

print("Moving to 0,0")
cf.home(cem120)

print("Moving to 90,90")
input("Press Enter to continue or Ctrl+C to exit...")
tst.mount_posdown(cem120)

print("Moving to -90,90")
input("Press Enter to continue or Ctrl+C to exit...")
tst.mount_posup(cem120)

print("Exit")
sys.exit(0)