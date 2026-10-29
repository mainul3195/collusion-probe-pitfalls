"""List run ids whose transcript holds a failed-request placeholder ("[ERROR]") or an empty message.
Usage: python list_error_runs.py <raw run dirs...> > error_runs.txt"""
import glob
import json
import sys

for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/*/agent_prompts.json")):
        if any(r["response"].strip() in ("[ERROR]", "") for r in json.load(open(f))):
            print(f.split("/")[-2])
