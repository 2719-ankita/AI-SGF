
.PHONY: preprocess inject dbn ai_sgf evaluate reproduce clean

CONFIG?=config.yaml

preprocess:
	python scripts/preprocess.py --config $(CONFIG)

inject:
	python scripts/inject_attacks.py --config $(CONFIG) --attack-intensity medium

dbn:
	python scripts/train_baselines.py --config $(CONFIG) --model dbn

ai_sgf:
	python scripts/train_ai_sgf.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

reproduce: preprocess inject dbn ai_sgf evaluate

clean:
	rm -rf artifacts data/*.parquet data/*.log
