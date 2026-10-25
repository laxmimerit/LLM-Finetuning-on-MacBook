"""Mini-Clef: Qwen 3.5 0.8B + a small joint schema head → typed decisions, no text generation.

Same design as Cloudflare's Clef (Apache-2.0, huggingface.co/Cloudflare/clef, `joint_schema_model.py`),
scaled down and text-only:
1. One prompt holds the state AND the questions with their allowed options, written as text.
2. The LLM reads it once (one forward pass). Its word head (248k words) is not used.
3. The joint schema head reads the hidden states: each option looks for its evidence in the state,
   the questions are decided together, and every allowed option gets one logit.
4. A softmax per question gives probabilities; plain code writes the Jev / SystemOne JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
HEAD_CONFIG = {"hidden_size": 1024, "width": 512, "routing_layers": 1, "layers": 2, "heads": 8, "feedforward": 2048}


def render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def question_options(question: dict) -> list[tuple[str, Any]]:
    if question["type"] == "noul":
        crit = {"true": "The proposition is true or the answer is yes.", "false": "The proposition is false or the answer is no."}
        crit.update(question.get("criteria") or {})
        return [(k, crit[k]) for k in ("true", "false")]
    if question["type"] == "choice":
        return sorted((str(k), v) for k, v in question["criteria"].items())
    return [(str(i), v) for i, v in enumerate(question["criteria"])]


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]  # token range of the instruction
    option_spans: tuple[tuple[int, int], ...]  # token range of each option's text
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]


def encode_record(tok, record: dict, max_length: int = 4096) -> EncodedRecord:
    """STATE first, then the SCHEMA: the model reads left to right, so every option token has already seen the state."""
    ids = lambda text: tok(text, add_special_tokens=False).input_ids
    schema, questions = ids("\n\nSCHEMA FIELDS:\n"), []
    for qi, (qid, q) in enumerate(record["questions"].items()):
        schema += ids(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        q0 = len(schema)
        schema += ids(render(q.get("instructions") or qid))
        q1 = len(schema)
        schema += ids("\nALLOWED OPTIONS:\n")
        spans, oids = [], []
        for oi, (oid, desc) in enumerate(question_options(q)):
            schema += ids(f"OPTION {oi + 1}: ")
            o0 = len(schema)
            schema += ids(render({"option_id": oid, **({"description": desc} if desc is not None else {})}))
            spans.append((o0, len(schema)))
            oids.append(oid)
            schema += ids("\n")
        schema += ids("END FIELD\n")
        questions.append((qid, QUESTION_TYPES[q["type"]], (q0, q1), spans, oids))
    prefix = ids(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n")
    suffix = ids("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:")
    state = ids(render(record["state"]))[: max_length - len(prefix) - len(schema) - len(suffix)]
    off = len(prefix) + len(state)
    shift = lambda s: (s[0] + off, s[1] + off)
    return EncodedRecord(
        tuple(prefix + state + schema + suffix),
        tuple(EncodedQuestion(qid, t, shift(qs), tuple(shift(s) for s in sp), tuple(oi)) for qid, t, qs, sp, oi in questions),
    )


def collate(records: list[EncodedRecord], pad_id: int, device, multiple: int = 128) -> dict:
    # pad to a multiple of 128 tokens: PyTorch on Apple GPUs builds a new graph for every new shape
    n = -(-max(len(r.input_ids) for r in records) // multiple) * multiple
    input_ids = torch.full((len(records), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(records), n), dtype=torch.long)
    for i, r in enumerate(records):
        input_ids[i, : len(r.input_ids)] = torch.tensor(r.input_ids)
        mask[i, : len(r.input_ids)] = 1
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "records": records}


class EvidenceRoutingLayer(torch.nn.Module):
    """Each option (a query) attends over every token of the prompt (the memory) to collect its evidence."""

    def __init__(self, width, heads, feedforward):
        super().__init__()
        self.query_norm, self.memory_norm = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward_norm = torch.nn.LayerNorm(width)
        self.feedforward = torch.nn.Sequential(torch.nn.Linear(width, feedforward), torch.nn.GELU(), torch.nn.Linear(feedforward, width))

    def forward(self, queries, memory, memory_mask=None):
        mem = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), mem, mem, key_padding_mask=memory_mask, need_weights=False)[0]
        return queries + self.feedforward(self.feedforward_norm(queries))


class JointSchemaHead(torch.nn.Module):
    """Clef's joint schema head, same parts, smaller (width 512, 1 routing + 2 field layers ≈ 16M parameters)."""

    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward):
        super().__init__()
        lin = lambda: torch.nn.Linear(hidden_size, width, bias=False)
        self.hidden_norm = torch.nn.LayerNorm(hidden_size)
        self.memory_projection, self.question_projection, self.option_question_projection = lin(), lin(), lin()
        self.global_projection, self.option_context_projection, self.option_lexical_projection = lin(), lin(), lin()
        self.type_embedding = torch.nn.Embedding(3, width)
        self.evidence_layers = torch.nn.ModuleList([EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)])
        self.option_summary_norm = torch.nn.LayerNorm(width)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerDecoderLayer(width, heads, feedforward, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.field_norm, self.option_norm = torch.nn.LayerNorm(width), torch.nn.LayerNorm(width)
        self.residual_scorer = torch.nn.Sequential(torch.nn.Linear(width * 4, width), torch.nn.GELU(), torch.nn.Linear(width, 1))
        self.prior_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = torch.nn.Parameter(torch.zeros(()))
        self.residual_gate = torch.nn.Parameter(torch.zeros(()))

    def forward(self, hidden_states, input_ids, attention_mask, records, output_embedding_weight):
        hidden = self.hidden_norm(hidden_states.float())
        L = hidden.shape[1]
        results = []
        for b, rec in enumerate(records):
            h = hidden[b]  # padded length, padding masked out below (fixed shapes keep the GPU fast)
            pad = (attention_mask[b] == 0).unsqueeze(0)
            memory = self.memory_projection(h).unsqueeze(0)
            global_vector = h[int(attention_mask[b].sum()) - 1]  # the last token has read the whole prompt
            spans = [q.question_span for q in rec.questions] + [s for q in rec.questions for s in q.option_spans]
            pool = torch.zeros(len(spans), L)
            for i, (s0, s1) in enumerate(spans):
                pool[i, s0:s1] = 1.0 / (s1 - s0)
            pool = pool.to(h.device)  # span means as one matmul
            nq, counts = len(rec.questions), [len(q.option_spans) for q in rec.questions]
            means = pool @ h
            qvec = means[:nq]
            ctx = list(torch.split(means[nq:], counts))  # what the LLM thinks of each option, in context
            lex = list(torch.split(pool[nq:] @ output_embedding_weight[input_ids[b]].float(), counts))  # the option's own words
            types = torch.tensor([q.question_type for q in rec.questions], device=h.device)

            opts = torch.cat([self.option_context_projection(c) + self.option_lexical_projection(l) + self.option_question_projection(qvec[i]).unsqueeze(0)
                              for i, (c, l) in enumerate(zip(ctx, lex))]).unsqueeze(0)
            for layer in self.evidence_layers:
                opts = layer(opts, memory, pad)
            opts = list(torch.split(opts[0], counts))

            base = self.question_projection(qvec)
            summaries = [(torch.softmax(o @ f / math.sqrt(o.shape[-1]), 0).unsqueeze(-1) * o).sum(0) for f, o in zip(base, opts)]
            fields = base + self.option_summary_norm(torch.stack(summaries)) + self.global_projection(global_vector).unsqueeze(0) + self.type_embedding(types)
            fields = fields.unsqueeze(0)
            for layer in self.layers:  # the questions look at each other and at the prompt: decided jointly
                fields = layer(fields, memory, memory_key_padding_mask=pad)
            fields = self.field_norm(fields[0])

            logits = []
            for i, (field, l, o) in enumerate(zip(fields, lex, opts)):
                anchor = F.normalize(qvec[i] + global_vector, dim=-1)
                prior = self.prior_logit_scale.clamp(max=math.log(100.0)).exp() * (F.normalize(l, dim=-1) @ anchor)
                o = self.option_norm(o)
                f = field.unsqueeze(0).expand_as(o)
                joint = self.joint_logit_scale.clamp(max=math.log(100.0)).exp() * F.cosine_similarity(f, o, dim=-1)
                joint = joint + self.residual_scorer(torch.cat([f, o, f * o, (f - o).abs()], -1)).squeeze(-1)
                logits.append(prior + torch.sigmoid(self.residual_gate) * joint)  # one logit per allowed option
            results.append(logits)
        return results


class MiniClef(torch.nn.Module):
    def __init__(self, language_model, head: JointSchemaHead):
        super().__init__()
        self.language_model, self.head = language_model, head

    def _base(self):
        lm = self.language_model
        return lm.get_base_model() if hasattr(lm, "get_base_model") else lm

    def forward(self, batch):
        base = self._base()
        out = base.model.language_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, return_dict=True)
        return self.head(out.last_hidden_state, batch["input_ids"], batch["attention_mask"], batch["records"], base.get_output_embeddings().weight)


def load_release_model(path: str | Path, device="mps", dtype=torch.bfloat16):
    """Load a release folder (merged backbone + joint_head.safetensors) or a Hub repo id."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    path = Path(path) if Path(path).is_dir() else Path(snapshot_download(str(path)))
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype).to(device)
    head = JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(path / "joint_head.safetensors"))
    return MiniClef(backbone, head.to(device)).eval(), AutoTokenizer.from_pretrained(path)


def systemone_answer(question, probs):
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probs["true"], 4)}
    if question["type"] == "choice":
        options = [str(o) for o in question["criteria"]]
        choice = max(options, key=probs.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": round(probs[choice], 4), "probabilities": {o: round(probs[o], 4) for o in options}}
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {"type": "score", "score": round(sum(i * probs[l] for i, l in enumerate(levels)), 4),
            "confidence": round(max(probs[l] for l in levels), 4), "probabilities": {l: round(probs[l], 4) for l in levels}}


@torch.inference_mode()
def systemone(model: MiniClef, tok, request: dict) -> dict:
    """A Jev / SystemOne request body ({model, state, questions}) in, the same response body out."""
    enc = encode_record(tok, request)
    device = next(model.parameters()).device
    logits = model(collate([enc], tok.pad_token_id, device))[0]
    answers = {q.question_id: systemone_answer(request["questions"][q.question_id], dict(zip(q.option_ids, lg.float().softmax(-1).tolist())))
               for q, lg in zip(enc.questions, logits)}
    return {"model": request.get("model", "miniclef"), "answers": answers, "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0}}


def save_release(folder, backbone, head, tokenizer):
    """Clef's release layout: the LoRA merged into the backbone, the head next to it, and this file to load both."""
    import shutil
    from safetensors.torch import save_file

    folder = Path(folder)
    backbone.merge_and_unload().save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, folder / "joint_head.safetensors")
    (folder / "joint_head_config.json").write_text(json.dumps(HEAD_CONFIG, indent=2))
    shutil.copy(__file__, folder / "miniclef.py")
    return sorted(p.name for p in folder.iterdir())
