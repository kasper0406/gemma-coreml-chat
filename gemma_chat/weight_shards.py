"""Serialize the exported weights into blob files smaller than 2 GiB.

An ``.mlpackage`` keeps every weight in ``Data/com.apple.CoreML/weights/``, and
coremltools writes all of them into a single ``weight.bin``.  Once that one file
crosses **2 GiB** (``2**31`` bytes), Core ML stops offering the model to the
Neural Engine — not op by op, but *entirely*:

===============  ==========================  ===========================
weight.bin       ops reporting ANE support   ops the planner puts on ANE
===============  ==========================  ===========================
1.61 GB          96 / 96                     96
2.15 GB          **0** / 128                 0
2.50 GB          **0** / 160                 0
3.22 GB          **0** / 192                 0
3.22 GB, sharded 192 / 192                   192
===============  ==========================  ===========================

Measured with ``MLComputePlan`` on ``.cpuAndNeuralEngine`` over a synthetic
chain of int4 per-channel ``matmul``s (one weight per matmul, so the only thing
that changes across rows is how many bytes land in the blob).  The last row is
the *same 3.22 GB model* with its weights spread over four ~1 GB files: every op
is ANE-eligible again, and the planner schedules all of them there.  Nothing
about the ops, shapes, dtypes or quantization scheme differs between the rows —
only the size of the file the weights sit in.

That is what this module removes.  ``MILProtoExporter.get_weight_path`` is the
one coremltools hook that decides which blob a constant is written to; the
default returns ``weight.bin`` for everything.  Overriding it to roll over to
``weight_1.bin``, ``weight_2.bin``, … before the budget is spent keeps every
file comfortably under the cliff, and costs nothing else: the file names are
recorded per-constant in the MIL proto, ``.mlmodelc`` compilation copies them
across unchanged, and weight deduplication is untouched (the exporter caches by
``weight_id`` *around* this hook, so a constant shared by all 16 functions is
still written once).

Gemma-4 E2B cannot avoid the cliff by quantizing harder: at int4 the two
embedding tables alone are 1.37 GB and the matmul weights another 0.94 GB, so
even a hypothetical zero-cost logit projection leaves the model over 2 GiB.
Sharding is the only fix that does not change a single number in the weights.

**This is necessary, not sufficient.**  With the blob capped (2.03 GB +
1.26 GB), the 35-layer model still reports zero ANE-eligible ops, because the
ANE compiler also rejects the graph itself — a 15-layer export fails with
``(ANECompiler) Error: live input tensor <private> not used in network`` and the
35-layer one crashes ``ANECompilerService`` outright.  That second gate is open
work; this module removes the one that is unambiguously ours to remove, and
costs nothing when the model is small enough not to need it.
"""

from __future__ import annotations

import os

from coremltools.converters.mil.backend.mil import load as _mil_load
from coremltools.converters.mil.mil.types.type_mapping import (
    SUB_BYTE_DTYPE_METADATA_KEY as _SUB_BYTE_KEY,
)
from coremltools.models.utils import _WEIGHTS_FILE_NAME

# The cliff itself.  A blob file must stay strictly below this.
_HARD_LIMIT = 1 << 31

# What one shard is allowed to hold.  The ~110 MB of slack against the hard
# limit absorbs the blob format's per-record header and padding, which this
# module estimates rather than measures.
_SHARD_BUDGET = 2_040_000_000


def _record_bytes(val) -> int:
    """On-disk size of one weight record, including header and padding.

    Sub-byte weights (this project's int4) are carried in an ``int8`` numpy
    array tagged with the real MIL dtype, so ``nbytes`` overstates them by 2x.
    """
    metadata = getattr(val.dtype, "metadata", None)
    if metadata is not None and _SUB_BYTE_KEY in metadata:
        bits = metadata[_SUB_BYTE_KEY].get_bitwidth()
        data = (val.size * bits + 7) // 8
    else:
        data = val.nbytes
    return -(-data // 64) * 64 + 64


def _sharded_get_weight_path(self, op) -> str:
    """``MILProtoExporter.get_weight_path`` that rolls over between blob files.

    Called once per *distinct* constant (the exporter's ``weight_id`` cache sits
    in front of it), in serialization order, so a running byte count is all the
    bookkeeping a rollover needs.
    """
    state = self.__dict__.setdefault("_shard_state", {"index": 0, "used": 0})
    size = _record_bytes(op.outputs[0].val)
    if size >= _HARD_LIMIT:
        raise ValueError(
            f"const {op.name} is {size / 1e9:.2f} GB on its own, which no blob "
            f"file can hold (limit {_HARD_LIMIT / 1e9:.2f} GB).  It has to be "
            "split or quantized further before the model can run on the ANE."
        )
    if state["used"] and state["used"] + size > _SHARD_BUDGET:
        state["index"] += 1
        state["used"] = 0
    state["used"] += size

    index = state["index"]
    name = _WEIGHTS_FILE_NAME if index == 0 else f"weight_{index}.bin"
    return os.path.join(self.weights_dir, name)


_mil_load.MILProtoExporter.get_weight_path = _sharded_get_weight_path
