from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from enum import Enum


IGNORED_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]+")
LEADING_SEPARATOR_PATTERN = re.compile(r"^[\s,.;:!?，。！？]+")
TOKEN_PATTERN = re.compile(r"\S+")
FUZZY_SIMILARITY_THRESHOLD = 0.88
MINIMUM_RELAXED_BOUNDARY_CHARACTERS = 3
DEFAULT_FINALIZE_SILENCE_MS = 2000
CONSENSUS_HISTORY_SIZE = 3


class CandidateProvenance(str, Enum):
    NONE = "NONE"
    TENTATIVE = "TENTATIVE"
    VALIDATED = "VALIDATED"


def normalize_for_matching(text: str) -> str:
    return IGNORED_PATTERN.sub("", text).lower()


def normalize_with_positions(text: str) -> tuple[str, list[int]]:
    normalized: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text):
        lowered = character.lower()
        if IGNORED_PATTERN.fullmatch(lowered):
            continue
        normalized.append(lowered)
        positions.append(index)
    return "".join(normalized), positions


def is_boundary_before(text: str, index: int) -> bool:
    return index == 0 or bool(IGNORED_PATTERN.fullmatch(text[index - 1]))


def is_boundary_after(text: str, index: int) -> bool:
    return index >= len(text) or bool(IGNORED_PATTERN.fullmatch(text[index]))


def longest_raw_suffix_prefix(previous: str, current: str, minimum_characters: int) -> int:
    previous = previous.strip()
    current = current.strip()
    for length in range(min(len(previous), len(current)), 0, -1):
        if previous[-length:] != current[:length]:
            continue
        previous_start = len(previous) - length
        if (
            len(normalize_for_matching(current[:length])) >= minimum_characters
            and is_boundary_before(previous, previous_start)
            and is_boundary_after(current, length)
        ):
            return length
    return 0


def longest_token_suffix_prefix(previous: str, current: str, minimum_characters: int) -> int:
    previous_tokens = list(TOKEN_PATTERN.finditer(previous))
    current_tokens = list(TOKEN_PATTERN.finditer(current))
    for count in range(min(len(previous_tokens), len(current_tokens)), 0, -1):
        if [match.group() for match in previous_tokens[-count:]] != [
            match.group() for match in current_tokens[:count]
        ]:
            continue
        current_end = current_tokens[count - 1].end()
        if len(normalize_for_matching(current[:current_end])) >= minimum_characters:
            return current_end
    return 0


def longest_normalized_suffix_prefix(
    previous: str, current: str, minimum_characters: int
) -> tuple[int, int]:
    normalized_previous, previous_positions = normalize_with_positions(previous)
    normalized_current, current_positions = normalize_with_positions(current)
    for length in range(min(len(normalized_previous), len(normalized_current)), minimum_characters - 1, -1):
        if normalized_previous[-length:] != normalized_current[:length]:
            continue
        previous_start = previous_positions[len(normalized_previous) - length]
        current_cut = current_positions[length - 1] + 1
        if is_boundary_before(previous, previous_start) and is_boundary_after(current, current_cut):
            return length, current_cut
    return 0, 0


def longest_relaxed_boundary_suffix_prefix(previous: str, current: str) -> int:
    """Match a short exact seam when only one side is an explicit word boundary.

    Korean window seams commonly split a particle or begin inside an eojeol, so
    requiring boundaries on both sides misses clear repetitions such as
    ``아침``/``아침에`` and ``만들어봤는데``/``봤는데``. This remains limited
    to an exact suffix-prefix seam and never searches or replaces global text.
    """
    previous = previous.strip()
    current = current.strip()
    for length in range(min(len(previous), len(current)), 0, -1):
        if previous[-length:] != current[:length]:
            continue
        normalized_length = len(normalize_for_matching(current[:length]))
        if normalized_length < MINIMUM_RELAXED_BOUNDARY_CHARACTERS:
            continue
        previous_start = len(previous) - length
        if is_boundary_before(previous, previous_start) or is_boundary_after(current, length):
            return length
    return 0


def fuzzy_suffix_prefix(
    previous: str,
    current: str,
    minimum_characters: int,
    minimum_similarity: float = FUZZY_SIMILARITY_THRESHOLD,
) -> tuple[float, int]:
    """Return a conservative approximate previous-suffix/current-prefix match."""
    normalized_previous, _ = normalize_with_positions(previous)
    normalized_current, current_positions = normalize_with_positions(current)
    if min(len(normalized_previous), len(normalized_current)) < minimum_characters:
        return 0.0, 0

    blocks = [block for block in SequenceMatcher(None, normalized_previous, normalized_current).get_matching_blocks() if block.size]
    best_similarity = 0.0
    best_cut = 0
    for first in blocks:
        if first.b > 1:
            continue
        for last in blocks:
            if last.a < first.a or last.b < first.b or last.a + last.size < len(normalized_previous) - 1:
                continue
            current_length = last.b + last.size
            previous_suffix = normalized_previous[first.a:]
            current_prefix = normalized_current[:current_length]
            if min(len(previous_suffix), len(current_prefix)) < minimum_characters:
                continue
            similarity = SequenceMatcher(None, previous_suffix, current_prefix).ratio()
            if similarity >= minimum_similarity and current_length > best_cut:
                best_similarity = similarity
                best_cut = current_positions[current_length - 1] + 1
    return best_similarity, best_cut


def complete_fuzzy_overlap_token(current: str, cut: int) -> int:
    """Do not expose the unmatched tail of a partially matched current token.

    Approximate Korean seams can align through the first syllable of a changed
    eojeol.  Appending the remainder would manufacture a hybrid such as
    ``도로상환 황을`` from ``도로상환`` / ``도로 상황을``.  The old hypothesis is
    retained; this only moves the start of genuinely new text to the next token.
    """
    for token in TOKEN_PATTERN.finditer(current):
        if token.start() < cut < token.end():
            return token.end()
    return cut


def corrected_suffix_prefix(previous: str, current: str) -> tuple[int, int, float] | None:
    """Align a bounded, word-delimited seam, allowing Korean syllable errors.

    Return old normalized suffix length, new display cut, and similarity.
    Only adjacent windows may use this; it never searches session history.
    Short matches need close decomposed Hangul (jamo) agreement as well as
    syllable agreement. A new hypothesis is evidence, not ground truth.
    """
    left, left_positions = normalize_with_positions(previous)
    right, right_positions = normalize_with_positions(current)
    best = None
    best_rank = (0.0, 0)
    for length in range(4, min(48, len(left)) + 1):
        start = left_positions[len(left) - length]
        if not is_boundary_before(previous, start):
            continue
        suffix = left[-length:]
        for count in range(max(4, length - 2), min(len(right), length + 2) + 1):
            cut = right_positions[count - 1] + 1
            if not is_boundary_after(current, cut):
                continue
            prefix = right[:count]
            syllable_score = SequenceMatcher(None, suffix, prefix, autojunk=False).ratio()
            jamo_score = SequenceMatcher(
                None, unicodedata.normalize("NFD", suffix),
                unicodedata.normalize("NFD", prefix), autojunk=False,
            ).ratio()
            # Do not merge different short phrases solely because their endings
            # look similar. Longer seams permit modest spelling corrections.
            required = 0.82 if min(length, count) <= 6 else 0.88
            if min(length, count) <= 6 and suffix[0] != prefix[0]:
                continue
            common_prefix = normalized_common_prefix_length(suffix, prefix)
            if (common_prefix >= 6 and max(length, count) - common_prefix <= 4
                    and syllable_score >= 0.80):
                # A shared multiword stem with a revised Korean ending, e.g.
                # "공원에는 운동을 하거든요" -> "공원에는 운동을 하거나".
                required = min(required, 0.86)
            if syllable_score < 0.60 or jamo_score < required:
                continue
            rank = (jamo_score, min(length, count))
            if rank > best_rank:
                best_rank = rank
                best = (length, cut, jamo_score)
    return best


def anchored_prefix_revision(previous: str, current: str) -> tuple[int, int, float] | None:
    """Revise a short modifier before a shared word, with external timing support.

    E.g. '룰렛 돌려서' / '제가 돌려서 ...'. This is lexical alignment, not
    semantic equivalence. The caller must establish that the previous speech
    tail occupies the shared audio. Never generalize to numbers, repeated words,
    arbitrary common endings, or a whole-phrase duplicate without new content.
    """
    old_tokens = list(TOKEN_PATTERN.finditer(previous))
    new_tokens = list(TOKEN_PATTERN.finditer(current))
    if len(old_tokens) < 2 or len(new_tokens) < 3:
        return None
    old_modifier, old_anchor = [normalize_for_matching(m.group()) for m in old_tokens[-2:]]
    new_modifier, new_anchor = [normalize_for_matching(m.group()) for m in new_tokens[:2]]
    if not (
        old_anchor == new_anchor and len(old_anchor) >= MINIMUM_RELAXED_BOUNDARY_CHARACTERS
        and 1 <= len(old_modifier) <= 2 and 1 <= len(new_modifier) <= 2
        and old_modifier != new_modifier
        and all(re.fullmatch(r"[가-힣]+", token) for token in (old_modifier, new_modifier, old_anchor))
        and old_modifier not in old_anchor and new_modifier not in old_anchor
    ):
        return None
    old_seam = old_modifier + old_anchor
    new_seam = new_modifier + new_anchor
    return len(old_seam), new_tokens[1].end(), SequenceMatcher(None, old_seam, new_seam).ratio()


def supported_overlap_tail_anchor(
    previous: str,
    current: str,
    minimum_characters: int,
) -> tuple[int, int, float] | None:
    """Locate a bounded previous-tail/current-head anchor for acoustic revision.

    This only identifies lexical correspondence. The caller must separately
    prove that adjacent VAD intervals cover the same overlap audio. A five-char
    minimum avoids treating common short endings as enough evidence to rewrite
    a partial, while still covering Korean phrase revisions such as
    ``그냥과와 높아지는`` -> ``분양가와 높아지는``.
    """
    left = normalize_for_matching(previous)
    right, right_positions = normalize_with_positions(current)
    if not left or not right:
        return None
    minimum_anchor = max(5, minimum_characters - 1)
    allowed_left_tail = max(1, round(len(left) * 0.15))
    allowed_right_head = max(3, round(len(right) * 0.20))
    best: tuple[int, int, float] | None = None
    best_rank = (0, 0.0)
    for block in SequenceMatcher(None, left, right, autojunk=False).get_matching_blocks():
        if block.size < minimum_anchor:
            continue
        left_tail = len(left) - block.a - block.size
        if left_tail > allowed_left_tail or block.b > allowed_right_head:
            continue
        compared_span = (len(left) - block.a) + block.b + block.size
        similarity = 2 * block.size / compared_span
        rank = (block.size, similarity)
        if rank <= best_rank:
            continue
        start = right_positions[block.b]
        end = right_positions[block.b + block.size - 1] + 1
        best = start, end, similarity
        best_rank = rank
    return best


def append_preserving_text(assembled: str, new_text: str) -> str:
    if not assembled:
        return new_text.strip()
    new_text = new_text.strip()
    if not new_text:
        return assembled
    separator = "" if assembled[-1].isspace() or new_text[0] in ",.;:!?，。！？" else " "
    return assembled + separator + new_text


@dataclass(frozen=True)
class AssemblyEvent:
    window: int
    speaker: int
    raw: str
    previous: str
    match_type: str
    overlap: str
    new: str
    assembled: str
    raw_length: int
    overlap_length: int
    new_length: int
    duplicate_only: bool
    suspicious_deletion: bool
    review_reasons: tuple[str, ...]
    matching_seconds: float
    similarity: float | None
    raw_fragment: str
    new_fragment: str
    utterance_hypothesis: str
    session_text: str
    confirmed_prefix_length: int = 0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["review_reasons"] = list(self.review_reasons)
        return result


class SubtitleAssembler:
    def __init__(self, speaker: int, minimum_characters: int = 6) -> None:
        if minimum_characters < 2:
            raise ValueError("minimum_characters must be at least 2")
        self.speaker = speaker
        self.minimum_characters = minimum_characters
        self.previous = ""
        self.assembled = ""
        self.utterance_hypothesis = ""
        self.last_window = -1
        self._finalized_session = ""
        self.confirmed_prefix_length = 0

    def reset_utterance(self, *, final_text: str | None = None) -> None:
        """Start a new utterance without discarding finalized session history."""
        self._finalized_session = (self.assembled if final_text is None else
                                   append_preserving_text(self._finalized_session, final_text))
        self.assembled = self._finalized_session
        self.confirmed_prefix_length = 0
        self.previous = ""
        self.utterance_hypothesis = ""

    def process(
        self, window: int, raw: str, *,
        shared_speech: bool | None = None,
        supported_tail_revision: bool = False,
    ) -> AssemblyEvent:
        started = time.perf_counter()
        if window <= self.last_window:
            raise ValueError(
                f"Out-of-order assembler input for speaker {self.speaker}: "
                f"window {window} after {self.last_window}"
            )

        raw = raw.strip()
        before_hypothesis = self.utterance_hypothesis
        previous = self.previous if window == self.last_window + 1 and shared_speech is not False else ""
        match_type = "none"
        overlap = ""
        new_text = raw
        normalized_overlap_length = 0
        similarity: float | None = None

        if previous and raw:
            token_overlap_length = longest_token_suffix_prefix(
                previous, raw, self.minimum_characters
            )
            if token_overlap_length:
                match_type = "token_exact"
                overlap = raw[:token_overlap_length].strip()
                normalized_overlap_length = len(normalize_for_matching(overlap))
                new_text = raw[token_overlap_length:]
            else:
                raw_overlap_length = longest_raw_suffix_prefix(
                    previous, raw, self.minimum_characters
                )
                if raw_overlap_length:
                    match_type = "raw_exact"
                    overlap = raw[:raw_overlap_length].strip()
                    normalized_overlap_length = len(normalize_for_matching(overlap))
                    new_text = raw[raw_overlap_length:]
                else:
                    normalized_overlap_length, raw_cut = longest_normalized_suffix_prefix(
                        previous, raw, self.minimum_characters
                    )
                    if normalized_overlap_length:
                        match_type = "normalized_exact"
                        overlap = raw[:raw_cut].strip()
                        new_text = raw[raw_cut:]
                    else:
                        relaxed_cut = longest_relaxed_boundary_suffix_prefix(previous, raw)
                        if relaxed_cut:
                            match_type = "boundary_exact"
                            overlap = raw[:relaxed_cut].strip()
                            normalized_overlap_length = len(normalize_for_matching(overlap))
                            new_text = raw[relaxed_cut:]
                        else:
                            similarity, fuzzy_cut = fuzzy_suffix_prefix(
                                previous, raw, self.minimum_characters
                            )
                            if fuzzy_cut:
                                fuzzy_cut = complete_fuzzy_overlap_token(raw, fuzzy_cut)
                                match_type = "fuzzy"
                                overlap = raw[:fuzzy_cut].strip()
                                normalized_overlap_length = len(normalize_for_matching(overlap))
                                new_text = raw[fuzzy_cut:]

        correction = corrected_suffix_prefix(previous, raw) if previous and raw else None
        if correction is None and previous and raw and supported_tail_revision:
            correction = anchored_prefix_revision(previous, raw)
        tail_revision = None
        correction_covers_previous = bool(
            correction
            and correction[0] == len(normalize_for_matching(previous))
        )
        correction_beats_fuzzy_seam = bool(
            correction
            and similarity is not None
            and correction[2] > similarity
        )
        # When fuzzy alignment already separated an overlapping head from a
        # genuinely new suffix, keep an established prefix outside that seam and
        # append only the suffix. A single noisy adjacent window must not turn a
        # bounded tail append into a destructive correction. A full re-observation
        # of the previous fragment may still correct it, and correction-only
        # hypotheses continue to use the normal branch.
        if correction and (
            match_type == "none"
            or (
                match_type == "fuzzy"
                and (correction_covers_previous or correction_beats_fuzzy_seam)
            )
        ):
            old_length, new_cut, similarity = correction
            old_suffix = normalize_for_matching(previous)[-old_length:]
            # The old seam must actually be the end of the active hypothesis.
            # A finalized utterance is never revised.
            if normalize_for_matching(self.utterance_hypothesis).endswith(old_suffix):
                _, positions = normalize_with_positions(self.utterance_hypothesis)
                cut = positions[len(positions) - old_length]
                self.utterance_hypothesis = self.utterance_hypothesis[:cut] + raw
                match_type = "fuzzy_replace"
                overlap = raw[:new_cut].strip()
                normalized_overlap_length = len(normalize_for_matching(overlap))
                new_text = raw[new_cut:]
            else:
                correction = None
        else:
            correction = None
        if correction is None and match_type == "none" and previous and raw and supported_tail_revision:
            tail_revision = supported_overlap_tail_anchor(
                previous, raw, self.minimum_characters
            )
            normalized_previous = normalize_for_matching(previous)
            normalized_active, positions = normalize_with_positions(self.utterance_hypothesis)
            if not (
                tail_revision
                and normalized_active.endswith(normalized_previous)
            ):
                tail_revision = None
            else:
                protected_length = min(self.confirmed_prefix_length, len(normalized_active))
                protected_cut = (
                    positions[protected_length - 1] + 1 if protected_length else 0
                )
                protected_prefix = self.utterance_hypothesis[:protected_cut].rstrip()
                self.utterance_hypothesis = append_preserving_text(protected_prefix, raw)
                anchor_start, anchor_end, similarity = tail_revision
                match_type = "supported_tail_replace"
                overlap = raw[anchor_start:anchor_end].strip()
                normalized_overlap_length = len(normalize_for_matching(overlap))
                new_text = raw
        new_text = LEADING_SEPARATOR_PATTERN.sub("", new_text).strip()
        if correction is None and tail_revision is None:
            # Rebuild exact seams from the current display text. Appending a
            # fragment beginning inside an eojeol would create "아침 에".
            normalized_active, positions = normalize_with_positions(self.utterance_hypothesis)
            exact = match_type in {"token_exact", "raw_exact", "normalized_exact", "boundary_exact"}
            normalized_overlap = normalize_for_matching(overlap)
            if exact and normalized_active.endswith(normalized_overlap):
                seam_start = len(positions) - len(normalized_overlap)
                cut = positions[seam_start]
                if seam_start <= self.confirmed_prefix_length:
                    self.confirmed_prefix_length = seam_start + len(normalized_overlap)
                self.utterance_hypothesis = self.utterance_hypothesis[:cut] + raw
            else:
                self.utterance_hypothesis = append_preserving_text(
                    self.utterance_hypothesis, new_text
                )
        if correction is not None or tail_revision is not None:
            self.confirmed_prefix_length = min(
                self.confirmed_prefix_length,
                normalized_common_prefix_length(before_hypothesis, self.utterance_hypothesis),
            )
        self.assembled = append_preserving_text(
            self._finalized_session, self.utterance_hypothesis
        )
        self.previous = raw
        self.last_window = window

        normalized_raw_length = len(normalize_for_matching(raw))
        normalized_new_length = len(normalize_for_matching(new_text))
        review_reasons: list[str] = []
        if normalized_overlap_length == self.minimum_characters and overlap:
            review_reasons.append("overlap_at_minimum_threshold")
        if normalized_raw_length and normalized_overlap_length / normalized_raw_length >= 0.8:
            review_reasons.append("overlap_consumes_at_least_80_percent")
        duplicate_only = bool(
            raw and not new_text and overlap
            and normalize_for_matching(before_hypothesis) == normalize_for_matching(self.utterance_hypothesis)
        )
        elapsed = time.perf_counter() - started
        return AssemblyEvent(
            window=window,
            speaker=self.speaker,
            raw=raw,
            previous=previous,
            match_type=match_type,
            overlap=overlap,
            new=new_text,
            assembled=self.assembled,
            raw_length=normalized_raw_length,
            overlap_length=normalized_overlap_length,
            new_length=normalized_new_length,
            duplicate_only=duplicate_only,
            suspicious_deletion=bool(review_reasons),
            review_reasons=tuple(review_reasons),
            matching_seconds=elapsed,
            similarity=similarity,
            raw_fragment=raw,
            new_fragment=new_text,
            utterance_hypothesis=self.utterance_hypothesis,
            session_text=self.assembled,
            confirmed_prefix_length=self.confirmed_prefix_length,
        )


@dataclass(frozen=True)
class SubtitleStateEvent:
    window: int
    speaker: int
    utterance_id: int
    status: str
    action: str
    raw_text: str
    before: str
    text: str
    previous_raw: str
    match_type: str
    overlap_text: str
    similarity: float | None
    stream_time_seconds: float
    utterance_start_seconds: float
    finalize_reason: str | None
    matching_seconds: float
    stable_text: str
    tentative_text: str
    history_size: int
    stability_action: str
    publication_text: str
    source_supported_text: str
    source_supported: bool
    require_final_support: bool
    support_update: dict[str, object] = field(default_factory=dict)
    candidate_provenance: str = "NONE"
    candidate_transition: str = "keep"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def prefix_hypothesis_similarity(previous: str, current: str) -> float:
    """Compare an older hypothesis with the equally sized prefix of a newer one."""
    normalized_previous = normalize_for_matching(previous)
    normalized_current = normalize_for_matching(current)
    if not normalized_previous or len(normalized_current) < len(normalized_previous):
        return 0.0
    return SequenceMatcher(
        None,
        normalized_previous,
        normalized_current[: len(normalized_previous)],
    ).ratio()


def reconcile_partial_tail(
    partial: str,
    current: str,
    minimum_characters: int,
) -> tuple[str, float] | None:
    """Replace only a short unstable partial tail using a strongly aligned new window."""
    normalized_partial, partial_positions = normalize_with_positions(partial)
    normalized_current, current_positions = normalize_with_positions(current)
    if not normalized_partial or not normalized_current:
        return None
    match = SequenceMatcher(None, normalized_partial, normalized_current).find_longest_match()
    partial_tail = len(normalized_partial) - (match.a + match.size)
    current_coverage = match.size / len(normalized_current)
    allowed_tail = max(8, round(len(normalized_partial) * 0.2))
    if (
        match.a == 0
        and match.size == len(normalized_partial)
        and normalized_current != normalized_partial
    ):
        return None
    if (
        match.size < minimum_characters
        or partial_tail > allowed_tail
        or current_coverage < 0.5
    ):
        return None
    partial_cut = partial_positions[match.a + match.size - 1] + 1
    current_cut = current_positions[match.b + match.size - 1] + 1
    corrected = append_preserving_text(partial[:partial_cut], current[current_cut:])
    if normalize_for_matching(corrected) == normalized_partial:
        return None
    similarity = match.size * 2 / (len(normalized_partial) + len(normalized_current))
    return corrected, similarity


def display_cut_for_normalized_prefix(text: str, normalized_length: int) -> int:
    if normalized_length <= 0:
        return 0
    normalized, positions = normalize_with_positions(text)
    if not normalized:
        return 0
    return positions[min(normalized_length, len(normalized)) - 1] + 1


def normalized_common_prefix_length(left: str, right: str) -> int:
    normalized_left = normalize_for_matching(left)
    normalized_right = normalize_for_matching(right)
    length = 0
    for left_character, right_character in zip(normalized_left, normalized_right):
        if left_character != right_character:
            break
        length += 1
    return length


def split_hypothesis_at_normalized_prefix(
    hypothesis: str, normalized_prefix_length: int
) -> tuple[str, str]:
    """Split display text at a normalized prefix while preserving punctuation."""
    if normalized_prefix_length <= 0:
        return "", hypothesis.strip()
    normalized, positions = normalize_with_positions(hypothesis)
    if not normalized:
        return "", hypothesis.strip()
    cut = positions[min(normalized_prefix_length, len(normalized)) - 1] + 1
    while cut < len(hypothesis) and IGNORED_PATTERN.fullmatch(hypothesis[cut]):
        cut += 1
    stable = hypothesis[:cut].rstrip()
    tentative = LEADING_SEPARATOR_PATTERN.sub("", hypothesis[cut:]).strip()
    return stable, tentative


def prefix_before_suffix_overlap(text: str, normalized_overlap_length: int) -> str:
    normalized, positions = normalize_with_positions(text)
    prefix_length = len(normalized) - normalized_overlap_length
    if prefix_length <= 0:
        return ""
    return text[: positions[prefix_length - 1] + 1].rstrip()


def trim_repeated_consensus_boundary(stable: str, tentative: str) -> tuple[str, int]:
    """Trim a tiny duplicated seam caused by an approximate overlap boundary."""
    normalized_stable = normalize_for_matching(stable)
    normalized_tentative, tentative_positions = normalize_with_positions(tentative)
    for length in range(min(4, len(normalized_stable), len(normalized_tentative)), 0, -1):
        if normalized_stable[-length:] != normalized_tentative[:length]:
            continue
        cut = tentative_positions[length - 1] + 1
        return LEADING_SEPARATOR_PATTERN.sub("", tentative[cut:]).strip(), cut
    return tentative, 0


def supported_prefix_from_history(
    current: str,
    history: list[str],
    speaker: int,
    minimum_characters: int,
) -> int:
    """Return the longest current prefix supported by any recent hypothesis."""
    supported_cut = 0
    for previous in history:
        probe = SubtitleAssembler(speaker, minimum_characters)
        probe.previous = previous
        probe.last_window = 0
        event = probe.process(1, current)
        if event.match_type != "none":
            supported_cut = max(supported_cut, len(event.overlap))
            continue
        similarity = prefix_hypothesis_similarity(previous, current)
        if similarity >= 0.82:
            supported_cut = max(
                supported_cut,
                display_cut_for_normalized_prefix(
                    current, len(normalize_for_matching(previous))
                ),
            )
    return supported_cut


class SpeakerSubtitleState:
    """Per-speaker state for complete hypotheses of the current utterance.

    Boundary overlap removal belongs to ``SubtitleAssembler``. Each non-empty
    value observed here must be the complete hypothesis for only the current
    utterance, never cumulative session history and never a raw STT window.
    """

    def __init__(
        self,
        speaker: int,
        minimum_characters: int = 6,
        finalize_silence_ms: int = DEFAULT_FINALIZE_SILENCE_MS,
        require_final_support: bool = False,
    ) -> None:
        if minimum_characters < 2:
            raise ValueError("minimum_characters must be at least 2")
        if finalize_silence_ms < 1:
            raise ValueError("finalize_silence_ms must be positive")
        self.speaker = speaker
        self.minimum_characters = minimum_characters
        self.finalize_silence_seconds = finalize_silence_ms / 1000
        # Live VAD callers supply acoustic evidence. Text-only replay callers
        # retain their existing contract; production enables this explicitly.
        self.require_final_support = require_final_support
        self._source_supported_text = ""
        self._candidate_provenance = CandidateProvenance.NONE
        self._candidate_window: int | None = None
        self._candidate_independent_support = False
        self._candidate_owner_conflict = False
        # Independently evidenced text withheld for confirmation, NOT generic
        # unsupported text. This private buffer is never publication text.
        self._candidate_supported_text = ""
        self.final_segments: list[str] = []
        self.stable_text = ""
        self.tentative_text = ""
        self.tentative_supported_cut = 0
        self.hypothesis_history: list[str] = []
        self.partial_text = ""
        self.previous_raw = ""
        self.last_speech_time: float | None = None
        self.last_update_time: float | None = None
        self.last_window = -1
        self.utterance_id = 0
        self.utterance_start_seconds = 0.0

    def _observe(self, raw: str) -> None:
        self.hypothesis_history.append(raw)
        if len(self.hypothesis_history) > CONSENSUS_HISTORY_SIZE:
            del self.hypothesis_history[0]

    @property
    def candidate_context(self) -> dict[str, object]:
        """Admission receives a snapshot; subtitle state alone owns its lifetime."""
        return {
            "provenance": self._candidate_provenance.value,
            "window": self._candidate_window,
            "independent_support": self._candidate_independent_support,
            "owner_contained_conflict": self._candidate_owner_conflict,
            "pending_text_supported": bool(
                self._candidate_supported_text
                and normalize_for_matching(self._candidate_supported_text) == normalize_for_matching(self.partial_text)
            ),
        }

    def _display_text(self) -> str:
        return append_preserving_text(self.stable_text, self.tentative_text)

    def _publication_text(self) -> str:
        if not self.require_final_support:
            return self.partial_text
        # ``stable_text`` is textual consensus from overlapping STT windows.
        # Repeated ghost text can create it without independent acoustic
        # evidence, so production VAD publication must remain grounded in the
        # retained source-supported prefix.
        return self._source_supported_text

    def process(
        self,
        window: int,
        hypothesis: str,
        speech_detected: bool,
        stream_time_seconds: float,
        *,
        confirmed_prefix_length: int | None = None,
        source_supported: bool = False,
        source_text: str | None = None,
        candidate_transition: dict[str, object] | None = None,
    ) -> list[SubtitleStateEvent]:
        if window <= self.last_window:
            raise ValueError(
                f"Out-of-order subtitle input for speaker {self.speaker}: "
                f"window {window} after {self.last_window}"
            )
        self.last_window = window
        hypothesis = hypothesis.strip()
        transition = candidate_transition or {}
        candidate_action = transition.get("action", "keep")
        if candidate_action == "discard" and self._candidate_provenance == CandidateProvenance.TENTATIVE:
            return [self.finalize(window, stream_time_seconds, "secondary_candidate_rejected")]

        if not speech_detected:
            self.previous_raw = ""
            if (
                self.partial_text
                and self.last_speech_time is not None
                and stream_time_seconds - self.last_speech_time
                >= self.finalize_silence_seconds
            ):
                return [self.finalize(window, stream_time_seconds, "vad_silence")]
            return []

        self.last_speech_time = stream_time_seconds
        if not hypothesis:
            return []

        started = time.perf_counter()
        before = self.partial_text
        previous_raw = self.previous_raw
        match_type = "none"
        overlap = ""
        similarity: float | None = None

        if not before:
            self.utterance_id += 1
            self.utterance_start_seconds = stream_time_seconds
            self.stable_text = ""
            self.tentative_text = hypothesis
            self.tentative_supported_cut = 0
            self.hypothesis_history = []
            self._observe(hypothesis)
            action = "start"
            stability_action = "first_observation"
        else:
            self._observe(hypothesis)
            normalized_previous = normalize_for_matching(previous_raw)
            normalized_current = normalize_for_matching(hypothesis)
            if normalized_current == normalized_previous:
                stable_length = len(normalize_for_matching(self.stable_text))
                self.stable_text, self.tentative_text = (
                    split_hypothesis_at_normalized_prefix(hypothesis, stable_length)
                )
                self.tentative_supported_cut = len(self.tentative_text)
                action = "retain"
                match_type = "same_hypothesis"
                similarity = 1.0
                stability_action = "tentative_retained"
            elif normalized_previous and normalized_current.startswith(normalized_previous):
                old_stable_length = len(normalize_for_matching(self.stable_text))
                promoted_length = len(normalized_previous)
                self.stable_text, self.tentative_text = (
                    split_hypothesis_at_normalized_prefix(hypothesis, promoted_length)
                )
                self.tentative_supported_cut = len(self.tentative_text)
                action = "extend"
                match_type = "hypothesis_extension"
                similarity = 1.0
                stability_action = (
                    "promote_stable"
                    if promoted_length > old_stable_length
                    else "tentative_replace"
                )
            else:
                common_length = normalized_common_prefix_length(
                    previous_raw, hypothesis
                )
                self.stable_text, self.tentative_text = (
                    split_hypothesis_at_normalized_prefix(hypothesis, common_length)
                )
                self.tentative_supported_cut = len(self.tentative_text)
                action = "replace"
                match_type = "hypothesis_correction"
                similarity = SequenceMatcher(
                    None, normalized_previous, normalized_current
                ).ratio()
                stability_action = "consensus_correction"

        if confirmed_prefix_length is not None:
            # Assembled prefixes copied from the last hypothesis are not an
            # independent observation. Only the assembler knows that provenance.
            self.stable_text, self.tentative_text = split_hypothesis_at_normalized_prefix(
                hypothesis, max(0, confirmed_prefix_length)
            )
            stability_action = (
                "independent_window_support" if self.stable_text
                else "awaiting_independent_support"
            )
        self.partial_text = self._display_text()
        # Only an explicit adjacent-window acoustic confirmation can release a
        # deferred candidate. Both observations must be independently supported,
        # and the ENTIRE previous hypothesis must be in the withheld buffer.
        # Generic unsupported prefixes/gaps never qualify for this transition.
        candidate_confirmed = bool(
            candidate_action == "validate"
            and self._candidate_provenance == CandidateProvenance.TENTATIVE
            and self._candidate_window == window - 1
            and self._candidate_independent_support
            and transition.get("independent_support")
            and source_supported and source_text and before
            and normalize_for_matching(self._candidate_supported_text) == normalize_for_matching(before)
        )
        # Acoustic evidence belongs to this raw window, not to unobserved
        # prefixes the assembler copied from older hypotheses.
        retained_length = normalized_common_prefix_length(self.partial_text, self._source_supported_text)
        self._source_supported_text, _ = split_hypothesis_at_normalized_prefix(
            self.partial_text, retained_length)
        support_update: dict[str, object] = {
            "reason": "no_current_source_support",
            "source_text": hypothesis if source_text is None else source_text,
            "retained_prefix_length": retained_length,
        }
        if source_supported:
            normalized = normalize_for_matching(self.partial_text)
            observed = normalize_for_matching(hypothesis if source_text is None else source_text)
            # A copied assembler prefix is not made source-supported merely
            # because the current raw window supports a later suffix. Only a
            # prefix that was already acoustically supported may bridge them.
            known_prefix = retained_length
            suffix_matches = bool(observed and normalized.endswith(observed))
            unobserved_prefix_length = len(normalized) - len(observed)
            support_update.update(
                observed_suffix_matches=suffix_matches,
                unobserved_prefix_length=unobserved_prefix_length,
                unsupported_gap_length=(max(0, unobserved_prefix_length - known_prefix)
                                        if suffix_matches else None),
                reason=("empty_source_text" if not observed else
                        "source_not_hypothesis_suffix" if not suffix_matches else
                        "unsupported_prefix_gap"),
            )
            if observed and normalized.endswith(observed) and len(normalized) - len(observed) <= known_prefix:
                self._source_supported_text = self.partial_text
                support_update["reason"] = "current_source_covers_unretained_text"
        if candidate_confirmed:
            # The assembler may revise the seam, so raw source_text need not be
            # a literal suffix. The two complete contributing observations were
            # independently evidenced; this exception lasts only this transition.
            self._source_supported_text = self.partial_text
            support_update["coverage_reason_before_confirmation"] = support_update["reason"]
            support_update["reason"] = "secondary_candidate_confirmed"
        if candidate_action == "hold":
            self._candidate_provenance = CandidateProvenance.TENTATIVE
            observed = normalize_for_matching(source_text or "")
            # Do not remember copied or unsupported prefixes as pending evidence.
            self._candidate_supported_text = (
                self.partial_text if transition.get("independent_support")
                and observed and observed == normalize_for_matching(self.partial_text) else ""
            )
        elif self._source_supported_text:
            self._candidate_provenance = CandidateProvenance.VALIDATED
            self._candidate_supported_text = ""
        self._candidate_window = window
        self._candidate_independent_support = bool(transition.get("independent_support"))
        self._candidate_owner_conflict = bool(transition.get("owner_contained_conflict"))
        support_update.update(candidate_provenance=self._candidate_provenance.value,
                              candidate_transition=candidate_action)
        self.previous_raw = hypothesis
        self.last_update_time = stream_time_seconds
        return [
            SubtitleStateEvent(
                window=window,
                speaker=self.speaker,
                utterance_id=self.utterance_id,
                status="partial",
                action=action,
                raw_text=hypothesis,
                before=before,
                text=self.partial_text,
                previous_raw=previous_raw,
                match_type=match_type,
                overlap_text=overlap,
                similarity=similarity,
                stream_time_seconds=stream_time_seconds,
                utterance_start_seconds=self.utterance_start_seconds,
                finalize_reason=None,
                matching_seconds=time.perf_counter() - started,
                stable_text=self.stable_text,
                tentative_text=self.tentative_text,
                history_size=len(self.hypothesis_history),
                stability_action=stability_action,
                publication_text=self._publication_text(),
                source_supported_text=self._source_supported_text,
                source_supported=source_supported,
                require_final_support=self.require_final_support,
                support_update=support_update,
                candidate_provenance=self._candidate_provenance.value,
                candidate_transition=str(candidate_action),
            )
        ]

    def finalize(
        self,
        window: int,
        stream_time_seconds: float,
        reason: str,
    ) -> SubtitleStateEvent:
        if not self.partial_text:
            raise ValueError("Cannot finalize an empty subtitle partial")
        tentative_before = self.tentative_text
        before = self.partial_text
        text = before
        if self.require_final_support:
            # Silence and pipeline_end add no acoustic evidence. Textual
            # repetition alone cannot turn an unsupported hypothesis into a
            # user-visible FINAL.
            text = self._source_supported_text
        event = SubtitleStateEvent(
            window=window,
            speaker=self.speaker,
            utterance_id=self.utterance_id,
            status="final",
            action="finalize" if text else "discard",
            raw_text="",
            before=before,
            text=text,
            previous_raw=self.previous_raw,
            match_type="none",
            overlap_text="",
            similarity=None,
            stream_time_seconds=stream_time_seconds,
            utterance_start_seconds=self.utterance_start_seconds,
            finalize_reason=reason,
            matching_seconds=0.0,
            stable_text=self.stable_text,
            tentative_text=tentative_before,
            history_size=len(self.hypothesis_history),
            stability_action=("tentative_retained_at_final" if text == before
                              else "tentative_discarded_at_final"),
            publication_text=text,
            source_supported_text=self._source_supported_text,
            source_supported=False,
            require_final_support=self.require_final_support,
            support_update={"reason": "finalize_without_new_evidence"},
            candidate_provenance=self._candidate_provenance.value,
            candidate_transition="reset",
        )
        if text:
            self.final_segments.append(text)
        self._source_supported_text = ""
        self._candidate_provenance = CandidateProvenance.NONE
        self._candidate_window = None
        self._candidate_independent_support = False
        self._candidate_owner_conflict = False
        self._candidate_supported_text = ""
        self.stable_text = ""
        self.tentative_text = ""
        self.tentative_supported_cut = 0
        self.hypothesis_history = []
        self.partial_text = ""
        self.previous_raw = ""
        self.last_speech_time = None
        self.last_update_time = stream_time_seconds
        return event

    def flush(self, window: int, stream_time_seconds: float) -> SubtitleStateEvent | None:
        if not self.partial_text:
            return None
        return self.finalize(window, stream_time_seconds, "pipeline_end")
