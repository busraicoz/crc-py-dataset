from __future__ import annotations
from jsonschema import validate

RECORD_SCHEMA = {
    "type": "object",
    "properties": {
        "comment": {"type": "string"},
        "code": {"type": "string"},
        "file_path": {"type": "string"},
        "pr_number": {"type": ["number", "integer"]},
        "repo": {"type": "string"},
        "owner": {"type": "string"},
        "comment_id": {"type": ["number", "integer"]},
        "comment_created_at": {"type": "string"},
        "category": {"type": "string"},
        "subcategory": {"type": "string"},
        "enriched": {"type": "string"},
        "line_number": {"type": ["number", "integer", "null"]},
    },
    "required": [
        "comment", "code", "file_path", "pr_number", "repo", "owner",
        "comment_id", "comment_created_at", "category", "subcategory",
        "enriched"
    ]
}

def validate_record(obj: dict) -> None:
    validate(instance=obj, schema=RECORD_SCHEMA)
