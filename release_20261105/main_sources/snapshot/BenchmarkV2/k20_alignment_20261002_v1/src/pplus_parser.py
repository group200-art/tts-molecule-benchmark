import re
from typing import Dict, Any, Tuple
SMILES_TAG_RE = re.compile(r"<\s*SMILES\s*>\s*(.*?)\s*<\s*/\s*SMILES\s*>", re.I|re.S)

def parse_smiles(
    raw_response: str,
) -> Tuple[str, str, int]:
    matches = [
        value.strip()
        for value in SMILES_TAG_RE.findall(
            raw_response
            or ""
        )
        if value.strip()
    ]

    if not matches:
        return (
            "",
            "no_nonempty_smiles_tag",
            0,
        )

    return (
        matches[-1],
        "tag_last_nonempty",
        len(
            matches
        ),
    )
