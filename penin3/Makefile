.PHONY: run-fast run-valid run-soak test-fast

run-fast:
	PENIN3_LOG_LEVEL=INFO V7_DRY_RUN=1 python -m penin3_system

run-valid:
	PENIN3_LOG_LEVEL=INFO V7_DRY_RUN=0 python -m penin3_system

run-soak:
	PENIN3_LOG_LEVEL=DEBUG V7_DRY_RUN=0 python -m penin3_system

test-fast:
	pytest -q penin3/tests -k "smoke or acfa or worm"
