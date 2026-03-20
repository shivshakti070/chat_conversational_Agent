"""
rag_chain.py
------------
Conversational RAG chain for the XYZ improved banking assistant.

Architecture:
  1. Condensation step   — collapses multi-turn chat history + follow-up
                           into a standalone query (fixes chatbot context-loss)
  2. Retrieval step      — FAISS top-k with intent re-ranking
  3. Generation step     — LLM answering with retrieved context + improved prompts

LLM backends supported (in priority order):
  a. OpenAI GPT-4o / GPT-3.5-turbo   (if OPENAI_API_KEY is set)
  b. HuggingFace  google/flan-t5-base (local fallback, no API key needed)
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np

from vector_store import FAISSVectorStore
from embedder import EmbeddingPipeline
from improved_prompts import (
    CONDENSE_QUESTION_TEMPLATE,
    build_rag_prompt,
    format_chat_history,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# LLM backend loader
# ---------------------------------------------------------------------------

def _load_llm():
    """
    Try to load OpenAI first; fall back to a local HuggingFace model.
    Returns a callable: (prompt: str) -> str
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            def openai_generate(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system",  "content": SYSTEM_PROMPT},
                        {"role": "user",    "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=512,
                )
                return response.choices[0].message.content.strip()

            print("✓ LLM backend: OpenAI GPT-3.5-turbo")
            return openai_generate
        except Exception as e:
            print(f"⚠ OpenAI failed ({e}), falling back to local model.")

    # Local HuggingFace fallback
    try:
        from transformers import pipeline as hf_pipeline
        generator = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
        )

        def hf_generate(prompt: str) -> str:
            # Flan-T5 context window is small — truncate prompt
            truncated = prompt[-2000:]
            result = generator(truncated, do_sample=False)
            return result[0]["generated_text"].strip()

        print("✓ LLM backend: google/flan-t5-base (local)")
        return hf_generate
    except Exception as e:
        print(f"⚠ HuggingFace also failed ({e}). Using echo mode.")

        def echo_generate(prompt: str) -> str:
            # Absolute fallback: return the retrieved context as response
            return (
                "[LLM unavailable — showing retrieved context]\n\n"
                + prompt.split("ASSISTANT RESPONSE:")[-1].strip()
            )

        return echo_generate


# ---------------------------------------------------------------------------
# Conversational RAG Chain
# ---------------------------------------------------------------------------

class ConversationalRAGChain:
    """
    Multi-turn conversational RAG over XYZ chatbot interactions.

    Parameters
    ----------
    vector_store    : FAISSVectorStore with indexed chunks
    embedding_pipeline : fitted EmbeddingPipeline (for query encoding)
    k               : number of context chunks to retrieve (default 3)
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_pipeline: EmbeddingPipeline,
        k: int = 3,
    ):
        self.store = vector_store
        self.pipeline = embedding_pipeline
        self.k = k
        self.chat_history: List[Tuple[str, str]] = []   # (human, ai) pairs
        self._llm = _load_llm()

    # ------------------------------------------------------------------
    def _condense_question(self, question: str) -> str:
        """
        If there's chat history, condense history + question into a
        standalone query for retrieval.
        """
        if not self.chat_history:
            return question

        condensation_prompt = CONDENSE_QUESTION_TEMPLATE.format(
            chat_history=format_chat_history(self.chat_history),
            question=question,
        )
        try:
            return self._llm(condensation_prompt).strip()
        except Exception:
            return question   # graceful degradation

    # ------------------------------------------------------------------
    def _retrieve_context(self, standalone_query: str) -> str:
        """Encode query and retrieve diverse context chunks."""
        query_vec = self.pipeline.encode_query(standalone_query)
        return self.store.get_diverse_context(query_vec, k=self.k)

    # ------------------------------------------------------------------
    def ask(self, question: str, verbose: bool = False) -> str:
        """
        Process one customer turn.

        Parameters
        ----------
        question : raw customer message
        verbose  : if True, print retrieved context and condensed query

        Returns
        -------
        str : assistant response
        """
        # Step 1: Condense multi-turn history into standalone query
        standalone = self._condense_question(question)
        if verbose and standalone != question:
            print(f"\n  [Condensed query]: {standalone}")

        # Step 2: Retrieve context from vector store
        context = self._retrieve_context(standalone)
        if verbose:
            print(f"\n  [Retrieved context]\n{context[:500]}...\n")

        # Step 3: Build full RAG prompt
        full_prompt = build_rag_prompt(
            context=context,
            chat_history=self.chat_history,
            question=question,
        )

        # Step 4: Generate response
        response = self._llm(full_prompt)

        # Step 5: Save to history
        self.chat_history.append((question, response))

        return response

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear conversation history for a new session."""
        self.chat_history = []

    # ------------------------------------------------------------------
    def run_demo(self, test_queries: List[str]) -> None:
        """
        Run a series of test queries and print the conversation.
        """
        print("\n" + "=" * 70)
        print("  XYZ Improved Assistant — Demo Session")
        print("=" * 70)

        for query in test_queries:
            print(f"\n  CUSTOMER: {query}")
            response = self.ask(query, verbose=False)
            print(f"  ASSISTANT: {response}")
            print("-" * 70)

        print("\n  [End of demo session]\n")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    from data_parser import parse_interactions, build_chunks
    from metadata_tagger import tag_chunks

    input_path = Path(__file__).parent / "Input Decs" / "input.txt"
    interactions = parse_interactions(input_path)
    chunks = build_chunks(interactions, window_size=2)
    chunks = tag_chunks(chunks, interactions)

    pipeline = EmbeddingPipeline()
    pipeline.fit(chunks)

    store = FAISSVectorStore()
    store.add_chunks(chunks, pipeline.embeddings)

    chain = ConversationalRAGChain(store, pipeline, k=3)

    # Test the fixed flows
    test_queries = [
        "I lost my debit card",
        "and I'm travelling to Spain next week",       # follow-up (tests condensation)
        "how do I change my address to 16 Sycamore Tree, Haswell DH6 2AG",
        "I want to open a savings account",
        "I got a suspicious text saying my account is locked",
    ]

    chain.run_demo(test_queries)
