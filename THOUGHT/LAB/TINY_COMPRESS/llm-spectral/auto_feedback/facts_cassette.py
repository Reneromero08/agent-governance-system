"""Facts Cassette — Targeted knowledge graph + domain docs for model blind spots.

Stores structured facts as (entity, predicate, value) triples AND
rich domain documents with embedding-based retrieval. Targets the
specific factual errors observed in GPT-2 and TraDo-4B generation.

Lil Q integration: domain docs from THOUGHT/LAB/LIL_Q/test_sandbox/docs/
provide step-by-step reasoning (math, code, logic, chemistry) alongside
flat facts. E-gating via Born rule: E = <query|doc> = dot(query_vec, doc_vec).

Schema:
    triples(entity TEXT, predicate TEXT, value TEXT, embedding BLOB)
    docs(doc_id TEXT, domain TEXT, title TEXT, content TEXT, embedding BLOB)

Usage:
    fc = FactsCassette()
    fc.query("capital of France")  -> ["Paris"]
    fc.retrieve_docs("quadratic equation") -> ["# Expanding Squares...", ...]
"""

import json, math, sqlite3, struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- The actual facts both models get wrong ----

FACTS = [
    # Geography
    ("Burkina Faso", "capital", "Ouagadougou"),
    ("France", "capital", "Paris"),
    ("Japan", "capital", "Tokyo"),
    # Chemistry
    ("water", "chemical_formula", "H2O"),
    ("iron", "chemical_symbol", "Fe"),
    ("gold", "chemical_symbol", "Au"),
    ("carbon dioxide", "chemical_formula", "CO2"),
    ("salt", "chemical_formula", "NaCl"),
    # Physics
    ("light", "speed_km_per_s", "299792"),
    ("water", "boiling_point_celsius", "100"),
    ("water", "freezing_point_celsius", "0"),
    ("sound", "speed_m_per_s", "343"),
    # Biology
    ("human body", "bone_count", "206"),
    ("human", "chromosome_count", "46"),
    ("human body", "largest_organ", "skin"),
    ("human heart", "chamber_count", "4"),
    ("human", "senses_count", "5"),
    ("blue whale", "is_largest", "mammal"),
    ("photosynthesis", "absorbs_gas", "carbon dioxide"),
    # History
    ("World War II", "end_year", "1945"),
    ("Berlin Wall", "fell_year", "1989"),
    ("Soviet Union", "dissolved_year", "1991"),
    ("moon landing", "first_year", "1969"),
    ("French Revolution", "began_year", "1789"),
    # Astronomy
    ("Mars", "is_red", "planet"),
    ("Earth", "orbit_days", "365"),
    ("Moon", "gravity_ratio", "0.166"),
    ("Jupiter", "is_largest", "planet"),
    ("Mercury", "closest_to", "Sun"),
    # Math
    ("144", "square_root", "12"),
    ("17 times 24", "equals", "408"),
    ("hexagon", "side_count", "6"),
    ("circle", "degrees", "360"),
    ("triangle", "angle_sum_degrees", "180"),
    # People
    ("1984 novel", "author", "George Orwell"),
    ("Mona Lisa", "painter", "Leonardo da Vinci"),
    ("theory of relativity", "developed_by", "Albert Einstein"),
    ("light bulb", "inventor", "Thomas Edison"),
    # General
    ("Earth", "continent_count", "7"),
    ("Earth", "ocean_count", "5"),
    ("Pacific", "is_largest", "ocean"),
    ("Asia", "is_largest", "continent"),
    ("Everest", "is_tallest", "mountain"),
    # Money / pop
    ("Earth", "population_2024", "8 billion"),
    ("bat and ball total 1.10 bat costs 1.00 more", "ball_cost", "5 cents"),
    ("coin flip 3 times at least 2 heads", "probability", "50 percent"),
    ("lily pad doubles daily covers lake in 48 days", "half_lake_days", "47"),
    ("5 machines 5 minutes 5 widgets 100 machines 100 widgets", "time_minutes", "5"),
]


class FactsCassette:
    """Lightweight knowledge graph for model blind spots.

    Stores (entity, predicate, value) triples with embedding-based
    retrieval. Not a full cassette — uses SQLite directly for simplicity.
    """

    def __init__(self, db_path: str = None, model_name: str = "all-MiniLM-L6-v2"):
        if db_path is None:
            db_path = str(Path(__file__).resolve().parent / "facts.db")
        self.db_path = db_path
        self._model = None
        self._model_name = model_name
        self._dim = 384  # MiniLM embedding dimension

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _ensure_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                embedding BLOB
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON triples(entity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON triples(entity, predicate)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_domain ON docs(domain)")
        conn.commit()
        return conn

    def build(self):
        """Populate the database with all facts and domain docs."""
        print("Building facts cassette with {} triples + domain docs...".format(len(FACTS)), flush=True)
        model = self._get_model()
        conn = self._ensure_db()
        conn.execute("DELETE FROM triples")
        conn.execute("DELETE FROM docs")

        # Index triples
        texts = ["{} {}".format(e, p) for e, p, v in FACTS]
        embeddings = model.encode(texts, normalize_embeddings=True)
        for (entity, predicate, value), emb in zip(FACTS, embeddings):
            emb_bytes = emb.astype(np.float32).tobytes()
            conn.execute(
                "INSERT INTO triples (entity, predicate, value, embedding) VALUES (?, ?, ?, ?)",
                (entity, predicate, value, emb_bytes))

        # Index domain docs from Lil Q
        docs_dir = Path(__file__).resolve().parent / "docs"
        doc_count = 0
        if docs_dir.exists():
            for md_file in sorted(docs_dir.glob("**/*.md")):
                domain = md_file.parent.name
                content = md_file.read_text(encoding="utf-8")
                title = content.split("\n")[0].lstrip("# ").strip() if content.startswith("#") else md_file.stem
                doc_id = "{}/{}".format(domain, md_file.stem)
                # Embed the title for retrieval
                emb = model.encode([title], normalize_embeddings=True)[0]
                emb_bytes = emb.astype(np.float32).tobytes()
                conn.execute(
                    "INSERT OR REPLACE INTO docs (doc_id, domain, title, content, embedding) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, domain, title, content, emb_bytes))
                doc_count += 1

        conn.commit()
        conn.close()
        print("  Done. {} triples + {} docs indexed.".format(len(FACTS), doc_count), flush=True)

    def query(self, query_text: str, predicate: str = None, top_k: int = 3) -> List[Dict]:
        """Semantic search for relevant facts.

        Args:
            query_text: Natural language query (e.g. "capital of France")
            predicate: Optional filter (e.g. "capital")
            top_k: Number of results

        Returns:
            List of {entity, predicate, value, similarity} dicts
        """
        model = self._get_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT id, entity, predicate, value, embedding FROM triples").fetchall()

        results = []
        for row in rows:
            rid, entity, pred, value, emb_bytes = row
            if predicate and pred != predicate:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = float(np.dot(query_emb, emb))  # Cosine since normalized
            results.append({
                "entity": entity, "predicate": pred, "value": value,
                "similarity": round(sim, 4),
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        conn.close()
        return results[:top_k]

    def get(self, entity: str, predicate: str) -> Optional[str]:
        """Direct lookup by entity + predicate."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT value FROM triples WHERE entity = ? AND predicate = ?",
            (entity, predicate)).fetchone()
        conn.close()
        return row[0] if row else None

    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        triples = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        domains = conn.execute(
            "SELECT domain, COUNT(*) as n FROM docs GROUP BY domain ORDER BY n DESC"
        ).fetchall()
        conn.close()
        return {
            "triples": triples,
            "docs": docs,
            "domains": [{"domain": d, "count": n} for d, n in domains],
        }

    def retrieve_docs(self, query_text: str, domain: str = None,
                       top_k: int = 3, threshold: float = 0.3) -> List[str]:
        """E-gated document retrieval (Lil Q Born rule).

        Args:
            query_text: Natural language query
            domain: Optional domain filter (math, code, logic, chemistry)
            top_k: Number of documents
            threshold: E threshold (Born rule, E = <query|doc>)

        Returns:
            List of document content strings
        """
        model = self._get_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT doc_id, domain, title, content, embedding FROM docs").fetchall()

        results = []
        for row in rows:
            doc_id, doc_domain, title, content, emb_bytes = row
            if domain and doc_domain != domain:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            E = float(np.dot(query_emb, emb))  # Born rule: E = <psi|phi>
            if E >= threshold:
                results.append((E, doc_domain, title, content))

        conn.close()
        results.sort(key=lambda r: r[0], reverse=True)
        return [content for _, _, _, content in results[:top_k]]

    def retrieve_docs_with_scores(self, query_text: str, domain: str = None,
                                   top_k: int = 3, threshold: float = 0.3) -> List[dict]:
        """E-gated retrieval with scores for debugging."""
        model = self._get_model()
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT doc_id, domain, title, content, embedding FROM docs").fetchall()

        results = []
        for row in rows:
            doc_id, doc_domain, title, content, emb_bytes = row
            if domain and doc_domain != domain:
                continue
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            E = float(np.dot(query_emb, emb))
            if E >= threshold:
                results.append({"E": round(E, 3), "domain": doc_domain,
                                "title": title, "content": content[:200]})

        conn.close()
        results.sort(key=lambda r: r["E"], reverse=True)
        return results[:top_k]

    def retrieve_fact(self, prompt: str) -> List[str]:
        """Retrieve fact values relevant to a prompt. Returns value strings."""
        results = self.query(prompt, top_k=3)
        return [r["value"] for r in results if r["similarity"] > 0.3]

    def correct(self, prompt: str) -> Optional[str]:
        """Get the correct answer for a factual prompt, if known."""
        facts = self.retrieve_fact(prompt)
        return facts[0] if facts else None


# ============================================================================
# CLI + Test
# ============================================================================

if __name__ == "__main__":
    import sys
    fc = FactsCassette()

    if "--build" in sys.argv or len(sys.argv) == 1:
        fc.build()

    print("\nStats:", json.dumps(fc.get_stats(), indent=2))

    tests = [
        "capital of France",
        "chemical formula for water",
        "how many bones in human body",
        "who wrote 1984",
        "speed of light",
        "largest organ in human body",
        "what is 17 times 24",
        "who developed theory of relativity",
        "when did World War II end",
        "bat and ball cost 1.10",
    ]
    print("\n--- Retrieval Tests ---")
    for q in tests:
        facts = fc.retrieve_fact(q)
        direct = fc.correct(q)
        print(f"  {q[:50]:50s} -> {facts[:2]}  (direct: {direct})")
