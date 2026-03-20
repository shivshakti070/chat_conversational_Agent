"""
data_parser.py
--------------
Parses XYZ chatbot conversation transcripts from input.txt into structured
Interaction objects and two levels of chunks:
  1. Interaction-level chunks  (full conversation per interaction)
  2. Turn-level chunks         (sliding window of consecutive QA turns)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    speaker: str          # "customer" | "chatbot"
    text: str


@dataclass
class Interaction:
    id: int
    channel: str
    start_time: str
    end_time: str
    turns: List[Turn] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Human-readable concatenation of all dialogue turns."""
        return "\n".join(f"{t.speaker.upper()}: {t.text}" for t in self.turns)

    @property
    def customer_turns(self) -> List[str]:
        return [t.text for t in self.turns if t.speaker == "customer"]

    @property
    def chatbot_turns(self) -> List[str]:
        return [t.text for t in self.turns if t.speaker == "chatbot"]


@dataclass
class Chunk:
    chunk_id: str
    interaction_id: int
    chunk_type: str           # "interaction" | "turn_window"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Matches both "Interaction N" and "interaction - N" header formats
INTERACTION_HEADER_RE = re.compile(
    r"(?:interaction\s*-?\s*(\d+)|interaction\s+(\d+))",
    re.IGNORECASE,
)

# Inline metadata line: "channel - X, interaction starttime - Y, ..."
META_LINE_RE = re.compile(
    r"channel\s*-\s*(\S+),\s*"
    r"interaction\s+starttime\s*-\s*([\d\- :\.]+),\s*"
    r"interaction\s+endtime\s*-\s*([\d\- :\.]+)",
    re.IGNORECASE,
)

TURN_RE = re.compile(r"^(customer|chatbot)\s*:\s*(.*)", re.IGNORECASE)


def parse_interactions(filepath: Union[str, Path]) -> List[Interaction]:
    """
    Parse the raw chatbot transcript file into a list of Interaction objects.
    Handles both formatting styles present in the file:
      - "Interaction N \n channel - ..."
      - "interaction - N, channel - X, ..."
    """
    text = Path(filepath).read_text(encoding="utf-8")
    lines = text.splitlines()

    interactions: List[Interaction] = []
    current: Optional[Interaction] = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # ---- Detect interaction header ----
        header_match = INTERACTION_HEADER_RE.match(line)
        if header_match:
            # Save previous interaction
            if current:
                interactions.append(current)

            iid = int(header_match.group(1) or header_match.group(2))

            # Metadata may be on same line (inline) or next non-empty line
            meta_match: Optional[re.Match] = META_LINE_RE.search(line)
            if not meta_match and i + 1 < len(lines):
                meta_match = META_LINE_RE.search(lines[i + 1])
                if meta_match:
                    i += 1   # consume meta line

            if meta_match:
                channel = meta_match.group(1).lower()
                start_time = meta_match.group(2).strip()
                end_time = meta_match.group(3).strip()
            else:
                channel, start_time, end_time = "unknown", "", ""

            current = Interaction(
                id=iid,
                channel=channel,
                start_time=start_time,
                end_time=end_time,
            )
            i += 1
            continue

        # ---- Detect dialogue turns ----
        if current is not None:
            turn_match = TURN_RE.match(line)
            if turn_match:
                speaker = turn_match.group(1).lower()
                text = turn_match.group(2).strip()
                # Skip system tokens like /start, /close, prompt_survey, close
                if text and not text.startswith("/") and text not in {
                    "close", "prompt_survey", "authenticated"
                }:
                    current.turns.append(Turn(speaker=speaker, text=text))

        i += 1

    # Don't forget the last interaction
    if current:
        interactions.append(current)

    return interactions


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def build_chunks(
    interactions: List[Interaction],
    window_size: int = 2,
) -> List[Chunk]:
    """
    Build two kinds of chunks from a list of interactions.

    1. Interaction-level chunk:
       Full conversation text for the interaction. One chunk per interaction.

    2. Turn-window chunks (sliding window of `window_size` turns):
       Each window is a consecutive group of customer+chatbot exchanges.
       Preserves local conversational context for fine-grained retrieval.

    Parameters
    ----------
    interactions : list of Interaction
    window_size  : number of QA pairs per turn-window chunk (default 2)

    Returns
    -------
    list of Chunk
    """
    chunks: List[Chunk] = []

    for interaction in interactions:
        # 1. Interaction-level chunk
        interaction_text = (
            f"[Channel: {interaction.channel}] "
            f"[Start: {interaction.start_time}]\n"
            + interaction.full_text
        )

        chunks.append(Chunk(
            chunk_id=f"int_{interaction.id}_full",
            interaction_id=interaction.id,
            chunk_type="interaction",
            text=interaction_text,
            metadata={
                "interaction_id": interaction.id,
                "channel": interaction.channel,
                "start_time": interaction.start_time,
                "end_time": interaction.end_time,
                "num_turns": len(interaction.turns),
            },
        ))

        # 2. Turn-window chunks
        # Build QA pairs: group consecutive (customer, chatbot) turn pairs
        qa_pairs: List[Tuple[Turn, Optional[Turn]]] = []
        turn_list = interaction.turns
        idx = 0
        while idx < len(turn_list):
            if turn_list[idx].speaker == "customer":
                bot_resp = (
                    turn_list[idx + 1]
                    if idx + 1 < len(turn_list) and turn_list[idx + 1].speaker == "chatbot"
                    else None
                )
                qa_pairs.append((turn_list[idx], bot_resp))
                idx += 2 if bot_resp else 1
            else:
                idx += 1

        # Sliding window over QA pairs
        for w_start in range(0, max(1, len(qa_pairs) - window_size + 1)):
            window = qa_pairs[w_start: w_start + window_size]
            window_lines = []
            for customer_turn, bot_turn in window:
                window_lines.append(f"CUSTOMER: {customer_turn.text}")
                if bot_turn:
                    window_lines.append(f"CHATBOT: {bot_turn.text}")
            window_text = "\n".join(window_lines)

            chunks.append(Chunk(
                chunk_id=f"int_{interaction.id}_win_{w_start}",
                interaction_id=interaction.id,
                chunk_type="turn_window",
                text=window_text,
                metadata={
                    "interaction_id": interaction.id,
                    "channel": interaction.channel,
                    "window_start": w_start,
                    "window_size": len(window),
                },
            ))

    return chunks


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    input_path = Path(__file__).parent / "Input Decs" / "input.txt"
    interactions = parse_interactions(input_path)
    print(f"Parsed {len(interactions)} interactions")
    for ia in interactions:
        print(f"  [{ia.id}] channel={ia.channel}  turns={len(ia.turns)}")

    chunks = build_chunks(interactions, window_size=2)
    interaction_chunks = [c for c in chunks if c.chunk_type == "interaction"]
    window_chunks = [c for c in chunks if c.chunk_type == "turn_window"]
    print(f"\nTotal chunks : {len(chunks)}")
    print(f"  Interaction-level : {len(interaction_chunks)}")
    print(f"  Turn-window       : {len(window_chunks)}")

    print("\nSample interaction chunk text:\n")
    print(interaction_chunks[0].text[:400])
