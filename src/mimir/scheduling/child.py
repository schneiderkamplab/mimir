"""Record child ownership durably before executing a scheduled command."""
import json
import os
from pathlib import Path
import sys
import psutil
from mimir.evaluation.metrics import write_record

receipt,parent_pid,parent_created,*argv=sys.argv[1:]
write_record(receipt,{'pid':os.getpid(),'created':psutil.Process().create_time()})
try:
    parent=psutil.Process(int(parent_pid))
    alive=parent.create_time()==float(parent_created) and parent.is_running()
except psutil.NoSuchProcess:alive=False
if not alive:sys.exit(76)
os.execvpe(argv[0],argv,os.environ)
