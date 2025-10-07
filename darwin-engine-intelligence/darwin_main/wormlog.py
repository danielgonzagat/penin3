# wormlog.py — registro encadeado com hash (append-only)
import hashlib, json
from pathlib import Path

class WormLog:
    async def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    async def append(self, record: dict):
        prev_hash = self._last_hash()
        data = {"previous_hash": prev_hash} | record
        line = json.dumps(data, separators=(",", ":"))
        h = hashlib.sha256(line.encode()).hexdigest()
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.write(f"HASH:{h}\n")

    async def _last_hash(self):
        try:
            tail = self.path.read_text().strip().splitlines()
            for line in reversed(tail):
                if line.startswith("HASH:"):
                    return await line.split("HASH:",1)[1].strip()
        except Exception:
            pass
        return await "GENESIS"
    
    async def verify_chain(self):
        """Verifica integridade da cadeia hash"""
        try:
            lines = self.path.read_text().strip().splitlines()
            prev_expected = "GENESIS"
            
            i = 0
            while i < len(lines) - 1:
                if not lines[i+1].startswith("HASH:"):
                    i += 1
                    continue
                
                record_line = lines[i]
                hash_line = lines[i+1]
                
                # Verificar record
                record = json.loads(record_line)
                if record.get("previous_hash") != prev_expected:
                    return await False, f"Quebra de cadeia na linha {i}"
                
                # Verificar hash
                expected_hash = hashlib.sha256(record_line.encode()).hexdigest()
                actual_hash = hash_line.split("HASH:",1)[1].strip()
                
                if expected_hash != actual_hash:
                    return await False, f"Hash inválido na linha {i+1}"
                
                prev_expected = actual_hash
                i += 2
            
            return await True, "Cadeia íntegra"
        except Exception as e:
            return await False, f"Erro: {e}"