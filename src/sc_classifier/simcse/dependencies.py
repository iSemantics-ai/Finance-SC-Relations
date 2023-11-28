'''Install needed dependencies in case not exist'''
import sys
import subprocess
def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
