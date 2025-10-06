import os, argparse, glob, gzip, json
from pathlib import Path
from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.core.evaluator import EvaluatorPipeline
from darwinacci_omega.core.env_plugins import load_portfolio_preset
from darwinacci_omega.plugins import toy


def load_config(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def resume_latest(eng: DarwinacciEngine, ck_dir: str) -> int:
    p = Path(ck_dir)
    p.mkdir(parents=True, exist_ok=True)
    files = sorted(p.glob('cycle_*.json.gz'))
    if not files:
        return 0
    return eng.load_checkpoint_json(str(files[-1]))


def main():
    parser = argparse.ArgumentParser(description="Run Darwinacci-Ω")
    parser.add_argument("--from-config", dest="config", help="Path to JSON config file", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in DARWINACCI_CKPT_DIR or data/checkpoints")
    parser.add_argument("--cycles", type=int, default=7, help="Max cycles to run")
    parser.add_argument("--pop", type=int, default=48, help="Population size")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--preset", choices=["fast","default","exhaustive"], default=None, help="Preset for cycles/pop/trials")
    args = parser.parse_args()

    # Apply config if provided
    if args.config:
        os.environ['DARWINACCI_CONFIG_PATH'] = args.config
        cfg = load_config(args.config)
        # Allow overriding common parameters
        args.pop = int(cfg.get('pop_size', args.pop))
        args.cycles = int(cfg.get('max_cycles', args.cycles))
        args.seed = int(cfg.get('seed', args.seed))

    # Apply presets
    if args.preset:
        if args.preset == 'fast':
            os.environ['DARWINACCI_TRIALS'] = os.environ.get('DARWINACCI_TRIALS', '1')
            args.cycles = min(args.cycles, 2)
            args.pop = min(args.pop, 16)
        elif args.preset == 'exhaustive':
            os.environ['DARWINACCI_TRIALS'] = os.environ.get('DARWINACCI_TRIALS', '5')
            args.cycles = max(args.cycles, 12)
            args.pop = max(args.pop, 96)

    # Wrap eval with pipeline (intrinsics/portfolio) if toggled via env
    # Portfolio preset if provided via env
    portfolio_name = os.getenv('DARWINACCI_PORTFOLIO_PRESET')
    if portfolio_name:
        fns, names = load_portfolio_preset(portfolio_name)
        pipe = EvaluatorPipeline(toy.evaluate, portfolio=fns, task_names=names)
        pipe.use_portfolio = True
        # Optionally enable curriculum
        if os.getenv('DARWINACCI_CURRICULUM', '0') == '1':
            pipe.use_curriculum = True
        eval_fn = pipe.evaluate
    else:
        eval_fn = EvaluatorPipeline(toy.evaluate).evaluate
    eng = DarwinacciEngine(toy.init_genome, eval_fn, max_cycles=args.cycles, pop_size=args.pop, seed=args.seed)

    # Resume if requested
    if args.resume:
        ck_dir = os.getenv('DARWINACCI_CKPT_DIR', 'data/checkpoints')
        try:
            cycle = resume_latest(eng, ck_dir)
            if cycle:
                print(f"[Resume] Restored checkpoint from cycle {cycle}")
        except Exception as e:
            print(f"[Resume] Failed to restore: {e}")

    champ = eng.run(max_cycles=args.cycles)
    print("\n🏆 Campeão:", round(champ.score, 4) if champ else "None")


if __name__ == "__main__":
    main()