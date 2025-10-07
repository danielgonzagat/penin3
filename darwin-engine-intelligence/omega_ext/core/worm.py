import json, os, time, hashlib
from typing import Dict, Any
class WORMLedger:
    def __init__(self, path:str="omega_ext/data/worm_ledger.csv", last_hash_path:str="omega_ext/data/worm_last_hash.txt"):
        self.path=path; self.last_hash_path=last_hash_path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            with open(self.path,"w") as f: f.write("ts,prev_hash,hash,event_json\n")
        if not os.path.exists(self.last_hash_path):
            with open(self.last_hash_path,"w") as f: f.write("GENESIS")
    def _prev(self)->str:
        with open(self.last_hash_path,"r") as f: return f.read().strip()
    def _write(self,h:str):
        with open(self.last_hash_path,"w") as f: f.write(h)
    def append(self, event:Dict[str,Any])->str:
        prev=self._prev(); blob=json.dumps(event, sort_keys=True, ensure_ascii=False)
        h=hashlib.sha256((prev+"|"+blob).encode("utf-8")).hexdigest()
        with open(self.path,"a") as f: f.write(f"{int(time.time())},{prev},{h},{blob}\n")
        self._write(h); return h
