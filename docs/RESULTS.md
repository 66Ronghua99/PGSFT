# Results and reporting

This package provides the paper's implementation and frozen protocols; it does
not bundle model checkpoints or generated benchmark outputs. Use each recipe's
`evidence.protocol_status` field to distinguish paper-reported configurations
from supported but unreported baseline configurations.

For a new run, report the selected checkpoint, merged-model manifest, exact
recipe, input manifest, GPU/software environment, and the complete benchmark
summary. GPQA/BFCL/LCB results produced here use the uniform 64K protocol and
must not be mixed with older selective-context results. SWE-bench results must
state that the 89-task set is the easy-50/frozen-50 union and is not the full
300-task SWE-bench Lite score.
