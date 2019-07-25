import sys
import subprocess
import os.path as op
import tempfile

params = sys.argv[1:]
################################################################################
################################# SIMULATION ###################################
# processes = []
# for param in params:
#     name = op.split(param)[-1].replace('.py', '')
#     f = tempfile.TemporaryFile()
#     p = subprocess.Popen(['python', 'simulator.py', param], stdout=f)
#     processes.append((p, f, name))
#
# for p, f, name in processes:
#     p.wait()
#     f.seek(0)
#     logname = op.join("log_sim_{}.txt".format(name))
#     with open(logname, "wb") as logfile:
#         logfile.write(f.read())
#     f.close()

################################################################################
################################# ANALYSIS #####################################
processes = []
for param in sys.argv[1:]:
    name = op.split(param)[-1].replace('.py', '')
    f = tempfile.TemporaryFile()
    p = subprocess.Popen(['python', 'run_analysis.py', param], stdout=f)
    processes.append((p, f, name))

for p, f, name in processes:
    p.wait()
    f.seek(0)
    logname = op.join("log_ana_{}.txt".format(name))
    with open(logname, "wb") as logfile:
        logfile.write(f.read())
    f.close()
