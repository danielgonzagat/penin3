import os, json, time, hashlib
import logging
logger = logging.getLogger(__name__)
try:
    import gzip
except Exception:
    gzip = None

class Worm:
    def __init__(self, path: str = "data/worm.csv", head: str = "data/worm_head.txt"):
        # Allow environment overrides for ledger locations
        env_path = os.getenv("DARWINACCI_WORM_PATH")
        env_head = os.getenv("DARWINACCI_WORM_HEAD")
        if isinstance(env_path, str) and env_path.strip():
            path = env_path.strip()
        if isinstance(env_head, str) and env_head.strip():
            head = env_head.strip()

        self.path = path
        self.head = head

        # Ensure parent directory exists
        parent_dir = os.path.dirname(path) or "."
        os.makedirs(parent_dir, exist_ok=True)

        # Initialize files if missing (support gzip header when using .gz)
        if not os.path.exists(self.path):
            if self.path.endswith('.gz') and gzip is not None:
                with gzip.open(self.path, 'wt') as f:
                    f.write("ts,prev,hash,json\n")
            else:
                with open(self.path, "w") as f:
                    f.write("ts,prev,hash,json\n")
        head_parent = os.path.dirname(self.head) or "."
        os.makedirs(head_parent, exist_ok=True)
        if not os.path.exists(self.head):
            tmp = self.head + '.tmp'
            with open(tmp, "w") as f:
                f.write("GENESIS")
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, self.head)

        # Durability toggle
        self._do_fsync = os.getenv("DARWINACCI_WORM_FSYNC", "0") == "1"

    def _prev(self)->str:
        try:
            return open(self.head).read().strip()
        except FileNotFoundError:
            # Initialize head on the fly if missing
            self._write_head('GENESIS')
            return 'GENESIS'

    def _write_head(self,h:str):
        # Ensure parent exists for head file as well
        try:
            head_parent = os.path.dirname(self.head) or "."
            os.makedirs(head_parent, exist_ok=True)
        except Exception:
            pass
        # Use unique tmp path to avoid races across threads/processes
        try:
            pid = os.getpid()
        except Exception:
            pid = 0
        try:
            now_us = int(time.time() * 1_000_000)
        except Exception:
            now_us = 0
        base = os.path.basename(self.head)
        tmp = os.path.join(os.path.dirname(self.head) or ".", f".{base}.{pid}.{now_us}.tmp")
        with open(tmp, "w") as f:
            f.write(h)
            try:
                f.flush(); os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, self.head)

    def _append_line(self, line: str) -> None:
        if self.path.endswith('.gz') and gzip is not None:
            with gzip.open(self.path, 'at') as f:
                f.write(line)
                # gzip lacks fileno/fsync portability; best-effort
        else:
            with open(self.path, "a") as f:
                f.write(line)
                if self._do_fsync:
                    f.flush(); os.fsync(f.fileno())

    def append(self, event:dict)->str:
        prev=self._prev()
        # Optional HMAC signature inside JSON payload
        ev = dict(event)
        try:
            key = os.getenv('DARWINACCI_HMAC_KEY')
            if key:
                base_blob = json.dumps(ev, sort_keys=True, ensure_ascii=False)
                sig = hashlib.sha256((key + "|" + prev + "|" + base_blob).encode()).hexdigest()
                ev['sig'] = sig
                ev['sig_alg'] = 'sha256(prev|json)'
        except Exception:
            pass
        blob=json.dumps(ev,sort_keys=True,ensure_ascii=False)
        h=hashlib.sha256((prev+"|"+blob).encode()).hexdigest()

        # Auto-rotate ledger if too large (>X MB) or by time window (days)
        try:
            rotate_mb = int(os.getenv('DARWINACCI_WORM_ROTATE_MB', '50'))
            rotate_days = int(os.getenv('DARWINACCI_WORM_ROTATE_DAYS', '0'))
            if os.path.exists(self.path) and not self.path.endswith('.gz') and os.path.getsize(self.path) > rotate_mb * 1024 * 1024:
                import shutil, gzip as _gzip
                ts = int(time.time())
                backup = f"{self.path}.{ts}.gz"
                with open(self.path, 'rb') as f_in, _gzip.open(backup, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                with open(self.path, "w") as f:
                    f.write("ts,prev,hash,json\n")
            # Time-based rotation: if header ts is older than N days, rotate
            if rotate_days > 0 and os.path.exists(self.path) and not self.path.endswith('.gz'):
                try:
                    # naive check: if file mtime older than days threshold
                    mtime = os.path.getmtime(self.path)
                    if (time.time() - mtime) > rotate_days * 86400:
                        import shutil, gzip as _gzip
                        ts = int(time.time())
                        backup = f"{self.path}.{ts}.gz"
                        with open(self.path, 'rb') as f_in, _gzip.open(backup, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        with open(self.path, "w") as f:
                            f.write("ts,prev,hash,json\n")
                except Exception:
                    pass
        except Exception:
            pass

        line = f"{int(time.time())},{prev},{h},{blob}\n"
        self._append_line(line)
        self._write_head(h)
        return h