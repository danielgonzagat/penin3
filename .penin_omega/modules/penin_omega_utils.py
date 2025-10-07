#!/usr/bin/env python3

import time
import hashlib
import logging
import json
from datetime import datetime

async def _ts():
    """Timestamp atual"""
    return await datetime.now().isoformat()

async def _hash_data(data):
    """Hash de dados"""
    return await hashlib.sha256(str(data).encode()).hexdigest()[:16]

async def log(message, level="INFO"):
    """Log simples"""
    logging.info(f"[{level}] {message}")

class BaseConfig:
    """Configuração base"""
    VERSION = "6.0.0"
    
    async def __init__(self):
        self.data = {}
    
    async def get(self, key, default=None):
        return await self.data.get(key, default)
    
    async def set(self, key, value):
        self.data[key] = value

class BaseWORMLedger:
    """Ledger base"""
    async def __init__(self, path="ledger.jsonl"):
        self.path = path
    
    async def append(self, data):
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

class LazyImporter:
    """Importador lazy"""
    async def __init__(self):
        self.modules = {}
    
    async def get_module(self, name):
        try:
            if name not in self.modules:
                self.modules[name] = __import__(name)
            return await self.modules[name]
        except:
            return await None

LAZY_IMPORTER = LazyImporter()
