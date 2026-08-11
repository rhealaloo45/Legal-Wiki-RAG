"""Offline checks for content-ambiguity detection — no DB, no LLM, no network.

The failure (Q85, 6/10 through v3 and v4): a question identifies its document by
text several documents share, and the system answers from one of them with no
sign the others exist.

  "What is the execution date of the services agreement entered into by Tata
   Sons Private Limited having its registered office at Bombay House, 24 Homi
   Mody Street, Mumbai?"

Service Agreement 2 answers 18 July 2025 and Service Agreement 4 answers
28 August 2025. Both are correct; the question cannot choose between them.

WHICH SIGNAL. Ground truth blames the registered-office block, and the first
three attempts at this were built on it. A diagnostic against the real corpus
showed the office block is reachable by phrase search in Service Agreement 2
ONLY — it lives in that document's "Parties" page and simply is not present in
Service Agreement 4's compiled pages. Building on it produced a confident answer
from SA 2, swapping one single answer for the other. The signal that does
discriminate is the pairing the question also supplies:

  "Tata Sons Private Limited"  → 10 documents corpus-wide
  Service Agreement family     → 62 documents
  intersection                 → exactly SA 2 and SA 4

which is _resolve_party_within_family, already in the codebase and already used
to RESOLVE when the intersection is one document. This change carries the case
where it is more than one. The office block is still read, but only to word the
disclosure — never to filter, since filtering on it is what lost SA 4.

Usage: cd app && python eval/check_description_ambiguity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import intent_agent, wiki

SA2 = "sess_Service Agreement 2_redacted.pdf"
SA4 = "sess_Service Agreement 4_redacted.pdf"
SA7 = "sess_Service Agreement 7_redacted.pdf"
NDA1 = "sess_NDA 1_redacted.pdf"

Q85 = ("What is the execution date of the services agreement entered into by "
       "Tata Sons Private Limited having its registered office at Bombay House, "
       "24 Homi Mody Street, Mumbai?")

# Q21 from the v4 report — the same mechanism resolving to ONE document, which
# must keep resolving cleanly and silently.
Q21 = "What is the termination clause of the Service Agreement of Tata Steel Limited?"

ADDRESS = "Bombay House, 24 Homi Mody Street, Mumbai"

PAGES = {
    "Parties – SA-Tata (Service Agreement)": {"source_doc": SA2},
    "Term and Termination – SA-Tata (Service Agreement)": {"source_doc": SA2},
    "Parties and Recitals – SA-Redwood (Service Agreement)": {"source_doc": SA4},
    "Execution and Signature – SA-Redwood (Service Agreement)": {"source_doc": SA4},
    "Confidentiality – Zephyr-Solaris (NDA 1)": {"source_doc": NDA1},
}

# Counts measured against the real corpus by eval/diagnose_q85.py.
CORPUS = {
    "Tata Sons": [SA2, SA4] + [f"other{i}" for i in range(8)],
    "Tata Steel": [SA7] + [f"steel{i}" for i in range(6)],
}
FAMILIES = {"Service Agreement": [SA2, SA4, SA7] + [f"sa{i}" for i in range(59)],
            "NDA": [NDA1]}


class _FakeDB:
    """The db helpers these paths touch, backed by dicts.

    find_source_docs_mentioning_phrase mimics phraseto_tsquery's behaviour that
    matters here: punctuation-insensitive matching on an adjacent phrase.
    """

    def __init__(self, corpus=None, families=None):
        self.corpus = CORPUS if corpus is None else corpus
        self.families = FAMILIES if families is None else families

    @staticmethod
    def _norm(s):
        return " ".join(s.lower().replace(",", " ").split())

    def find_source_docs_mentioning_phrase(self, sid, phrase, cap=25):
        key = self._norm(phrase)
        for probe, docs in self.corpus.items():
            p = self._norm(probe)
            if key.startswith(p) or p.startswith(key):
                return docs[:cap]
        return []

    def get_source_docs(self, sid):
        seen = []
        for docs in list(self.corpus.values()) + list(self.families.values()):
            for d in docs:
                if d not in seen:
                    seen.append(d)
        return seen

    def list_doc_families(self, sid):
        return list(self.families)

    def get_documents_by_family(self, sid, family):
        return list(self.families.get(family, []))


def _install(corpus=None, families=None):
    wiki.config.USE_DATABASE = True
    wiki._db = _FakeDB(corpus, families)
    wiki._load_index = lambda sid: {"pages": PAGES}


_results = []


def check(name, cond, detail=""):
    _results.append(bool(cond))
    print(("PASS  " if cond else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------------------
# 1. Q85 — the party/instrument pair fits two documents, and both must surface.
# ---------------------------------------------------------------------------
_install()

_pif = wiki._resolve_party_within_family(Q85, "s", set(FAMILIES["Service Agreement"]))
check("the party/instrument intersection is exactly the two agreements",
      _pif == {SA2, SA4}, str(sorted(_pif)))

_s = wiki.resolve_scope(Q85, "s", pages=PAGES)
check("Q85 pins BOTH candidates for retrieval",
      set(_s.get("target_docs") or []) == {SA2, SA4}, str(_s.get("method")))
_m = _s.get("ambiguous_match") or {}
check("Q85 is flagged ambiguous, not silently resolved",
      set(_m.get("docs") or []) == {SA2, SA4}, str(_m.get("docs")))
check("the disclosure names the party and the instrument",
      "Tata Sons" in _m.get("description", "")
      and "Service Agreement" in _m.get("description", ""),
      _m.get("description", ""))
check("the disclosure explains why the address the user gave did not help",
      "Bombay House" in _m.get("description", ""), _m.get("description", ""))
check("the weaker unresolved-party warning still rides along as a fallback",
      _s.get("unresolved_party"))

# The gate in front of it must let the question through, or none of the above
# ever runs — confirmed live twice: "Vague 'services agreement' reference →
# disambiguate among 68 service agreement docs" claimed Q85 on the type word
# alone, and the harness's reply pinned whichever document it happened to name.
check("the disambiguation gate lets the pair through",
      wiki.classify_query(Q85, "s").get("needs_disambiguation") is False)

# ---------------------------------------------------------------------------
# 2. A one-document intersection still resolves cleanly and says nothing.
# ---------------------------------------------------------------------------
_s21 = wiki.resolve_scope(Q21, "s", pages=PAGES)
check("a party/instrument pair fitting ONE document resolves silently",
      _s21.get("target_docs") == [SA7] and not _s21.get("ambiguous_match"),
      str(_s21.get("method")))
check("the gate lets a uniquely-resolving pair through too",
      wiki.classify_query(Q21, "s").get("needs_disambiguation") is False)

# ---------------------------------------------------------------------------
# 3. The branch stays inert where it should.
# ---------------------------------------------------------------------------
# An explicitly named document outranks everything.
_named = wiki.resolve_scope(
    "What is the execution date of Service Agreement 4, entered into by Tata "
    "Sons Private Limited having its registered office at Bombay House, 24 "
    "Homi Mody Street, Mumbai?", "s", pages=PAGES)
check("an explicit document number still wins", _named["method"] == "file",
      _named["method"])

# Too many documents to answer for each, one by one → widen to the family as before.
_wide = [f"sa{i}" for i in range(wiki._PARTY_IN_FAMILY_MAX_DOCS + 3)]
_install(corpus={"Tata Sons": _wide},
         families={"Service Agreement": _wide, "NDA": [NDA1]})
_s = wiki.resolve_scope(Q85, "s", pages=PAGES)
check("an intersection too large to enumerate widens to the family",
      not _s.get("ambiguous_match") and _s.get("method") == "default-family",
      str(_s.get("method")))

# A party so common its document list came back truncated is not intersected at
# all (_PARTY_FAMILY_SCAN_CAP) — an intersection against a truncated list is
# arbitrary, so scope widens rather than guesses.
_huge = [f"sa{i}" for i in range(wiki._PARTY_FAMILY_SCAN_CAP + 5)]
_install(corpus={"Tata Sons": _huge}, families={"Service Agreement": _huge})
check("a truncated party list is not intersected",
      not (wiki.resolve_scope(Q85, "s", pages=PAGES).get("ambiguous_match")))

_install()
wiki.config.USE_DATABASE = False
check("inert without a database",
      not (wiki.resolve_scope(Q85, "s", pages=PAGES).get("ambiguous_match")))
wiki.config.USE_DATABASE = True

# ---------------------------------------------------------------------------
# 4. The office-block extractor — still read, for wording only.
# ---------------------------------------------------------------------------
check("the office block is extracted from Q85",
      wiki._extract_descriptive_identifier(Q85) == ADDRESS,
      wiki._extract_descriptive_identifier(Q85))
check("a trailing clause is trimmed off it",
      wiki._extract_descriptive_identifier(
          "the SA with Acme having its registered office at Bombay House, 24 Homi "
          "Mody Street, Mumbai, and when does it expire?") == ADDRESS)
for negative in [
    # Asking ABOUT an office, not naming a document by one.
    "What is the registered office at which notices must be served under NDA 3?",
    "Where is the registered office that the seller nominated for service of process?",
    # A location inside the document — no party-office noun at all.
    "What does the lease say about the premises located at 5 Main Street, Pune?",
    "What is the execution date of Service Agreement 4?",
]:
    check("no office block: " + negative[:50],
          not wiki._extract_descriptive_identifier(negative))

# ---------------------------------------------------------------------------
# 5. Answer-side wiring — directive, banner and note; and silence otherwise.
# ---------------------------------------------------------------------------
_captured = {}


def _fake_generate_answer(*a, **kw):
    _captured.clear()
    _captured.update(kw)
    return {"answer": "", "scope_method": "", "scope_docs": [],
            "pages_used": [], "files_used": [], "confidence_score": 0}


wiki.generate_answer = _fake_generate_answer
intent_agent._emit = lambda *a, **kw: None

_BASE = {"question": Q85, "intent": "factual", "session_id": "s",
         "wiki_context": "ctx", "selected_titles": [], "retrieval_meta": {},
         "conversation_context": ""}

intent_agent.generate_answer_node({**_BASE, "scope_decision": {
    "scope": "single_doc", "target_docs": [SA2, SA4], "method": "party-in-family",
    "unresolved_party": "Tata Sons",
    "ambiguous_match": {"description": '"Tata Sons" in the Service Agreement family',
                        "docs": [SA2, SA4]}}})
_d = _captured.get("ambiguity_directive", "")
check("directive reaches the prompt", _d)
check("directive names both documents",
      "service agreement 2" in _d and "service agreement 4" in _d)
check("directive requires a per-document answer", "SEPARATELY FOR EACH ONE" in _d)
check("directive forbids picking a single winner", "Do NOT pick" in _d)
check("deterministic warning banner emitted",
      "fits 2 documents equally" in _captured.get("scope_warning", ""),
      _captured.get("scope_warning", "")[:60])
check("the banner replaces the weaker unresolved-party warning",
      "may reflect a different document" not in _captured.get("scope_warning", ""))
check("display note emitted",
      "each is answered separately" in _captured.get("scope_note", ""))

intent_agent.generate_answer_node({**_BASE, "scope_decision": {
    "scope": "single_doc", "target_docs": [SA7], "method": "party-in-family"}})
check("a one-document resolution adds no directive/warning/note",
      not _captured.get("ambiguity_directive")
      and not _captured.get("scope_warning") and not _captured.get("scope_note"))

# The disclosures this sits alongside must be unaffected.
intent_agent.generate_answer_node({**_BASE, "scope_decision": {
    "scope": "corpus", "target_docs": [], "method": "default",
    "unresolved_party": "Tata Sons"}})
check("pre-existing unresolved-party warning intact",
      "Tata Sons" in _captured.get("scope_warning", ""))

intent_agent.generate_answer_node({**_BASE, "scope_decision": {
    "scope": "single_doc", "target_docs": [SA7], "method": "carryover"}})
check("pre-existing carryover note intact",
      "already under discussion" in _captured.get("scope_note", ""))

print()
_failed = _results.count(False)
print(f"{_results.count(True)}/{len(_results)} passed")
sys.exit(1 if _failed else 0)
