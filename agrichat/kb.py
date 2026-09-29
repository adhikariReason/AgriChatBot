# -*- coding: utf-8 -*-
"""Knowledge-base loading and validation."""

from __future__ import annotations

import glob
import json
import os

KB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "kb")


class Entry:
    __slots__ = ("id", "domain", "crops", "patterns", "answer", "followups", "sources")

    def __init__(self, raw: dict, domain: str):
        self.id = raw["id"]
        self.domain = domain
        self.crops = raw.get("crops", [])
        self.patterns = raw["patterns"]
        self.answer = raw["answer"]
        self.followups = raw.get("followups", [])
        self.sources = raw.get("sources", [])

    def __repr__(self):
        return f"<Entry {self.id} ({len(self.patterns)} patterns)>"


def load(kb_dir: str = KB_DIR) -> list:
    """Load every data/kb/*.json file into a flat list of entries."""
    entries = []
    seen = {}
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        domain = data.get("domain", os.path.basename(path))
        for raw in data["entries"]:
            entry = Entry(raw, domain)
            if entry.id in seen:
                raise ValueError(
                    f"duplicate intent id {entry.id!r} in {path} and {seen[entry.id]}"
                )
            if not entry.patterns:
                raise ValueError(f"intent {entry.id!r} has no patterns")
            if not entry.answer.strip():
                raise ValueError(f"intent {entry.id!r} has an empty answer")
            seen[entry.id] = path
            entries.append(entry)

    known = {e.id for e in entries}
    for entry in entries:
        for ref in entry.followups:
            if ref not in known:
                raise ValueError(f"intent {entry.id!r} points at unknown followup {ref!r}")
    return entries
