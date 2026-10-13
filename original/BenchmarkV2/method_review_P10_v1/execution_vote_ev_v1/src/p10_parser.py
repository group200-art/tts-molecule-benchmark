"""Unchanged P10 last-complete-tag parser."""
import re
from typing import Dict,Any

def extract_candidate(
    raw_response: str,
) -> Dict[str, Any]:
    raw = str(raw_response or "")

    tag_matches = re.findall(
        r"<SMILES>\s*(.*?)\s*</SMILES>",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )

    nonempty_tag_matches = [
        match.strip()
        for match in tag_matches
        if match.strip()
    ]

    if nonempty_tag_matches:
        return {
            "parsed_candidate": (
                nonempty_tag_matches[-1]
            ),
            "parse_method": "tag_last_nonempty",
            "n_tag_matches": len(tag_matches),
        }

    for line in raw.splitlines():
        candidate = (
            line.strip()
            .strip("`")
            .strip()
        )

        for prefix in (
            "%%% Response:",
            "Response:",
            "Candidate:",
            "SMILES:",
        ):
            if candidate.lower().startswith(
                prefix.lower()
            ):
                candidate = candidate[
                    len(prefix):
                ].strip()

        if candidate:
            return {
                "parsed_candidate": candidate,
                "parse_method": (
                    "first_nonempty_line_fallback"
                ),
                "n_tag_matches": len(tag_matches),
            }

    return {
        "parsed_candidate": "",
        "parse_method": "empty",
        "n_tag_matches": len(tag_matches),
    }

def normalize_raw_text(text: str) -> str:
    normalized = re.sub(
        r"\s+",
        " ",
        str(text or "").strip(),
    )

    if not normalized:
        return "<EMPTY_RAW_RESPONSE>"

    return normalized
