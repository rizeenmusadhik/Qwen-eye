"""What a gradient map explains: one scalar score computed from the full model's logits.

A target builds its input (tower.EvalPreprocessor), says how many final positions of logits it
needs (`keep`, passed as logits_to_keep), and turns those logits into the score. `maps` lists
the maps one backward pass of that score gives: (label, sign); a signed map s becomes
ReLU(sign * s). An optional `resolve(logits)` sees the plain forward's logits before the
backward pass (LetterContrast picks its wrong option there).

  LetterContrast  a multiple-choice question (your own, or a VITATECS / MotionBench item with
                  the exact eval prompt): logit(correct) - logit(wrong)
                  at the answer position. Raw scores, as Grad-CAM uses the class score before
                  softmax: a confident model's probabilities barely move, so their gradient
                  vanishes. Content both options share raises both logits alike and cancels.
                  + side: evidence for the correct option; - side: for the wrong one (the
                  Grad-CAM paper's "counterfactual explanation").
  Phrase          "Is there <phrase> in the video? Answer yes or no.": at the answer position,
                  logsumexp(logit Yes, logit yes) - logsumexp(logit No, logit no), i.e.
                  log(p(Yes)+p(yes)) - log(p(No)+p(no)) from raw logits. The model answers in
                  lowercase (Qwen3-VL-8B: p(yes) ~ 1, p(Yes) ~ 0), so a capitalised-only
                  contrast would be about tokens it does not say. + side: evidence the thing
                  is there.
                  (Scoring log p(phrase | video, "Describe the video.") instead fails: the
                  shared first token "a" costs 11-13 nats and dominates every phrase, so
                  "a bonfire" and "a car" give the same maps.)

A new score (e.g. caption contrast) is a new class with the same members.
"""

from __future__ import annotations

import torch

DEFAULT_PROMPT = "Is there {phrase} in the video? Answer yes or no."


def _single_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens, expected 1")
    return ids[0]


def _answer_report(logits: torch.Tensor, ids: dict, tokenizer) -> dict:
    last = logits[0, -1].float()
    p = torch.softmax(last, -1)
    top = int(last.argmax())
    return {"logits": {k: float(last[i]) for k, i in ids.items()}, "p": {k: float(p[i]) for k, i in ids.items()},
            "argmax_token": tokenizer.decode([top]), "argmax_p": float(p[top])}


class LetterContrast:
    keep = 1

    def __init__(self, item):
        self.item = item
        self.name = "answer"
        self.wrong = item.wrong

    @property
    def maps(self):
        return [(f"for correct ({self.item.correct})", 1), (f"for wrong ({self.wrong})", -1)]

    def build(self, pre):
        self.tokenizer = pre.processor.tokenizer
        self.ids = {k: _single_token(self.tokenizer, k) for k in self.item.options}
        return pre.encode(self.item.prompt)

    def resolve(self, logits: torch.Tensor) -> None:
        """No wrong letter given (a correctly answered 3-4 option item): the strongest other option."""
        if self.wrong is None:
            last = logits[0, -1]
            self.wrong = max((k for k in self.ids if k != self.item.correct), key=lambda k: float(last[self.ids[k]]))

    def score(self, logits: torch.Tensor) -> torch.Tensor:
        last = logits[0, -1]
        return last[self.ids[self.item.correct]] - last[self.ids[self.wrong]]

    def report(self, logits: torch.Tensor) -> dict:
        return {"wrong": self.wrong} | _answer_report(logits, self.ids, self.tokenizer)


class Phrase:
    keep = 1

    def __init__(self, text: str, prompt: str = DEFAULT_PROMPT):
        if "{phrase}" not in prompt:
            raise ValueError(f"phrase prompt {prompt!r} has no {{phrase}}")
        self.text, self.prompt = text, prompt
        self.question = prompt.replace("{phrase}", text)
        self.name = f"phrase: {text}"
        self.maps = [(f'"{text}": for yes', 1), (f'"{text}": for no', -1)]

    def build(self, pre):
        self.tokenizer = pre.processor.tokenizer
        self.ids = {k: _single_token(self.tokenizer, k) for k in ("Yes", "No", "yes", "no")}
        return pre.encode(self.question)

    def score(self, logits: torch.Tensor) -> torch.Tensor:
        last = logits[0, -1].float()
        side = lambda *ks: torch.logsumexp(last[[self.ids[k] for k in ks]], 0)  # noqa: E731
        return side("Yes", "yes") - side("No", "no")

    def report(self, logits: torch.Tensor) -> dict:
        return {"question": self.question} | _answer_report(logits, self.ids, self.tokenizer)
