# ProtoNet (V6)

ProtoNet consumes V6 benchmark artifacts from `dataset_builder/output/benchmark/ambiguity_openworld`.

## Required Input Files

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

## Train

```powershell
python protonet\code\cli.py train --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld
```

## Evaluate

```powershell
python protonet\code\cli.py eval --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld --split test --checkpoint protonet\output\checkpoints\best.pt
```

## Export

```powershell
python protonet\code\cli.py export --input-type benchmark --input-dir dataset_builder\output\benchmark\ambiguity_openworld --checkpoint protonet\output\checkpoints\best.pt
```

## See Training/Eval Artifacts

```powershell
Get-ChildItem protonet\output\checkpoints
Get-ChildItem protonet\output -Recurse
```

## Full V6 Flow

1. Build dataset in `dataset_builder`.
2. Train with `protonet\code\cli.py train`.
3. Evaluate best checkpoint.
4. Export model bundle.
