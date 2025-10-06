import argparse, json, os
from pathlib import Path


def inspect(path: str, head: str | None = None, limit: int = 20, verify: bool = False):
    if not os.path.exists(path):
        print(f"not found: {path}")
        return 1
    # Print last N lines
    try:
        import gzip
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as f:
            lines = f.readlines()
        # Skip header if present
        start = 1 if lines and lines[0].startswith('ts,prev,hash,json') else 0
        tail = lines[max(start, len(lines)-limit):]
        print("# ts,prev,hash,json (last entries)")
        import json as _json
        for ln in tail:
            ln = ln.strip()
            if not ln:
                continue
            print(ln)
            if verify:
                try:
                    # ts,prev,hash,json
                    parts = ln.split(',', 3)
                    if len(parts) == 4:
                        prev = parts[1]
                        h = parts[2]
                        blob = parts[3]
                        obj = _json.loads(blob)
                        sig = obj.get('sig')
                        if sig:
                            key = os.getenv('DARWINACCI_HMAC_KEY')
                            base = dict(obj); base.pop('sig', None); base.pop('sig_alg', None)
                            calc = __import__('hashlib').sha256((key + '|' + prev + '|' + _json.dumps(base, sort_keys=True, ensure_ascii=False)).encode()).hexdigest() if key else None
                            ok = (calc == sig)
                            print(f"# verify sig={ok}")
                except Exception:
                    print("# verify sig=error")
        if head and os.path.exists(head):
            print("\n# HEAD:", Path(head).read_text().strip())
    except Exception as e:
        print(f"inspect failed: {e}")
        return 2
    return 0


def main():
    ap = argparse.ArgumentParser(description="WORM CLI")
    sub = ap.add_subparsers(dest='cmd')
    ins = sub.add_parser('inspect', help='show last N entries of worm ledger')
    ins.add_argument('--path', required=True)
    ins.add_argument('--head', required=False)
    ins.add_argument('--limit', type=int, default=20)
    ins.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'inspect':
        raise SystemExit(inspect(args.path, args.head, args.limit, args.verify))
    ap.print_help()


if __name__ == '__main__':
    main()
