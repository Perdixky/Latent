import re
from typing import Any


CHUNK_RE = re.compile(r"C\d{3,6}")
SLOT_RE = re.compile(
    r'<slot\s+id="(?P<slot_id>S\d+)">\s*\[(?P<chunks>[^\]]*)\]\s*</slot>',
    re.DOTALL,
)


def parse_prediction(text: str, valid_chunk_ids: set[str]) -> dict[str, Any]:
    slots = []
    all_pred_chunks = []
    invalid_ids = []
    slot_matches = list(SLOT_RE.finditer(text))

    for match in slot_matches:
        slot_id = match.group("slot_id")
        chunk_ids = CHUNK_RE.findall(match.group("chunks"))
        valid = []
        invalid = []

        for cid in chunk_ids:
            if cid in valid_chunk_ids:
                valid.append(cid)
            else:
                invalid.append(cid)

        slots.append(
            {
                "slot_id": slot_id,
                "pivot_chunks": valid,
                "invalid_chunks": invalid,
            }
        )
        all_pred_chunks.extend(valid)
        invalid_ids.extend(invalid)

    return {
        "slots": slots,
        "pred_chunks": list(dict.fromkeys(all_pred_chunks)),
        "pred_chunks_with_duplicates": all_pred_chunks,
        "invalid_ids": invalid_ids,
        "exact_format": len(slot_matches) > 0 and "</answer>" in text,
    }
