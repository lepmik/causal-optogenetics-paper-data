import sys
import subprocess

for param in sys.argv[1:]:
    p = subprocess.Popen(['python', 'run_analysis.py', param])
