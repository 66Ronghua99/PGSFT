# Paper-to-code mapping

| Paper component | Implementation | Recipe |
|---|---|---|
| Standard supervised objective | `src/pgsft/methods/sft.py` | `sft.yaml` |
| Detached demonstrated-token weighting | `src/pgsft/methods/dft.py` | `dft.yaml` |
| Base forward-KL regularization | `src/pgsft/methods/sft_base_kl.py` | `sft_base_kl.yaml` |
| Proximal SFT baseline | `src/pgsft/methods/psft.py`, `training/snapshot.py` | `psft.yaml` |
| Privileged trace construction | `src/pgsft/data/privileged.py`, `data/tokenization.py` | all PG-SFT data |
| Path projection and turn allocation | `src/pgsft/methods/pg_sft.py`, `methods/full_vocab.py` | `pg_sft.yaml` |
| Fixed Base/privileged teachers | `src/pgsft/training/teacher.py` | Base-KL, PG-SFT |
| Global token normalization | `src/pgsft/training/normalization.py` | all methods |
| Checkpoint selection and merge | `src/pgsft/training/checkpoint.py`, `training/merge.py` | all methods |

The Qwen3 and Qwen3.5 templates are separate checked-in files. Training and
inference templates are bound explicitly by every recipe; “ninja” in the paper
artifact discussion refers to these two backbone-specific template variants,
not an additional model or method.
