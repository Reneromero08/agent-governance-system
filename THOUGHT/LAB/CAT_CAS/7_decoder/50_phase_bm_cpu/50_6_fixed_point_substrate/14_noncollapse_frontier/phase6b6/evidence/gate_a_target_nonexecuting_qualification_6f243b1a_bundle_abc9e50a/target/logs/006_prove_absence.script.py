
import os, json
def state(p):
    try:
        os.lstat(p); return "PRESENT"
    except FileNotFoundError:
        return "ABSENT"
    except OSError as e:
        return "UNOBSERVABLE:%s"%type(e).__name__
res={"execution_root":state(os.environ["ROOT"]),"transfer_stage":state(os.environ["STAGE"])}
print(json.dumps(res, sort_keys=True))
