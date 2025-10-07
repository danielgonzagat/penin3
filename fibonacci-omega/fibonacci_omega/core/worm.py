import os, time, hashlib, pickle
from typing import Dict, Any, Optional

class WORMLedger:
    def __init__(self, path: str = "data/worm"):
        self.path = path
        self.log_path = os.path.join(path, "ledger_log.csv")
        self.obj_path = os.path.join(path, "objects")
        os.makedirs(self.obj_path, exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("ts,event_type,hash,details_json\n")

    def append(self, event_type: str, details: Dict[str, Any], obj: Optional[Any] = None) -> str:
        import json
        ts = int(time.time())
        blob = json.dumps(details, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(f"{ts}|{event_type}|{blob}".encode("utf-8")).hexdigest()
        
        with open(self.log_path, "a") as f:
            f.write(f"{ts},{event_type},{h},{blob}\n")
        
        if obj:
            with open(os.path.join(self.obj_path, f"{h}.pkl"), "wb") as f:
                pickle.dump(obj, f)
        return h
