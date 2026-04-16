import hashlib
from dataclasses import dataclass, field


@dataclass
class Parameter:
    name: str
    description: str
    sample_values: str


@dataclass
class Event:
    event_name: str
    event_definition: str
    screen_name: str
    parameters: list[Parameter] = field(default_factory=list)
    key_event: str = "No"
    detailed_event_definition: str = ""

    # Deterministic ID used as the Pinecone vector ID.
    # Auto-populated in __post_init__; do not set manually.
    doc_id: str = ""

    # Set by reranker after scoring
    rerank_score: float = 0.0

    # Intents (metric names) this event matched across retrieval passes
    matched_intents: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raw = f"{self.event_name}_{self.screen_name}"
            self.doc_id = hashlib.md5(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Text representations
    # ------------------------------------------------------------------

    def to_index_text(self) -> str:
        """Compact concatenation used for both Gemini embedding and BM25 indexing.

        Format: {event_name} | {screen_name} | {event_definition} | {detailed_event_definition} | {params}
        Keeps technical identifiers, semantic richness, and parameter details in one string.
        """
        parts = [self.event_name, self.screen_name, self.event_definition]
        if self.detailed_event_definition:
            parts.append(self.detailed_event_definition)
        if self.parameters:
            params_text = "; ".join(
                f"{p.name}: {p.description} (e.g. {p.sample_values})"
                for p in self.parameters
            )
            parts.append(params_text)
        return " | ".join(parts)

    def to_document(self) -> str:
        """Full prose passage fed to the reranker.

        More structured than to_index_text() — prose format with labelled fields
        and key_event flag.
        """
        params_text = "; ".join(
            f"{p.name}: {p.description} (e.g. {p.sample_values})"
            for p in self.parameters
        )
        doc = (
            f"Event: {self.event_name}. "
            f"Definition: {self.event_definition} "
            f"Screen: {self.screen_name}. "
            f"Parameters: {params_text}. "
        )
        if self.detailed_event_definition:
            doc += f"Detailed definition: {self.detailed_event_definition}. "
        doc += f"Key event: {self.key_event}."
        return doc
