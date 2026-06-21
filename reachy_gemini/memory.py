"""Long-term memory via mem0 (hosted) -- so Reachy actually remembers you.

mem0 stores facts about a person and retrieves the relevant ones semantically,
across sessions. We use it three ways:
  - remember(fact) / recall(query): explicit tools the brain can call
  - auto-recall: before each reply we fetch memories relevant to what you said and
    hand them to the brain, so it answers as someone who knows you
  - auto-save: after each turn we give the exchange to mem0, which extracts what's
    worth keeping on its own

No Upstash needed -- mem0 hosted brings its own vector store. (Upstash was only ever
a maybe for face/voice embeddings later, and even those can live locally on the robot.)

`user_id` is a single default for now; once face/voice recognition lands, the
recognized person's name becomes the user_id so memories are per-person.
"""
from __future__ import annotations

import threading


class Memory:
    def __init__(self, cfg: dict):
        m = (cfg or {}).get("mem0", {})
        self.api_key = m.get("api_key", "")
        self.enabled = bool(self.api_key)
        self.auto_recall = bool(m.get("auto_recall", True))
        self.auto_save = bool(m.get("auto_save", True))
        self.user_id = m.get("user_id", "default_user")
        self._client = None
        self._lock = threading.Lock()

    def _client_or_none(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    from mem0 import MemoryClient
                    self._client = MemoryClient(api_key=self.api_key)
                    print("[memory] mem0 connected", flush=True)
                except Exception as e:
                    print(f"[memory] mem0 unavailable ({e}) -- memory disabled", flush=True)
                    self.enabled = False
        return self._client

    @staticmethod
    def _texts(res) -> list:
        """Normalise mem0's search/get result (list, or {'results': [...]}) to strings."""
        items = res.get("results", res) if isinstance(res, dict) else res
        out = []
        for it in items or []:
            if isinstance(it, dict):
                t = it.get("memory") or it.get("text") or it.get("data")
            else:
                t = str(it)
            if t:
                out.append(t)
        return out

    def search(self, query: str, limit: int = 5) -> list:
        c = self._client_or_none()
        if c is None or not query:
            return []
        # mem0 v2 wants filters={'user_id': ...}; older versions took user_id=. Try v2, fall back.
        try:
            return self._texts(c.search(query, version="v2",
                                        filters={"user_id": self.user_id}, limit=limit))
        except Exception as e:
            try:
                return self._texts(c.search(query, user_id=self.user_id, limit=limit))
            except Exception:
                print(f"[memory] search failed: {e}", flush=True)
                return []

    def remember(self, fact: str) -> bool:
        c = self._client_or_none()
        if c is None or not (fact or "").strip():
            return False
        try:
            c.add([{"role": "user", "content": fact}], user_id=self.user_id)
            return True
        except Exception as e:
            print(f"[memory] remember failed: {e}", flush=True)
            return False

    def add_turn(self, user_text: str, reply_text: str) -> None:
        """Hand the exchange to mem0 in the background; it extracts what to keep."""
        c = self._client_or_none()
        if c is None or not (user_text and reply_text):
            return

        def _run():
            try:
                c.add([{"role": "user", "content": user_text},
                       {"role": "assistant", "content": reply_text}], user_id=self.user_id)
            except Exception as e:
                print(f"[memory] add_turn failed: {e}", flush=True)

        threading.Thread(target=_run, daemon=True).start()
