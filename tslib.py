"""Test Stand Library
"""


def turnon_backlight(sh):
    """
    Turn on the backlight using paramiko ssh connection
    
    Args:
        sh (ssh): ssh connection
    Returns:    
        sin (stdin, str): stdin
        sout (stdout, str): stdout
        serr (stderr, str): stderr
    """
    _command = "python3 backlight.py"
    sin, sout, serr = sh.execute(_command)
    return sin, sout, serr