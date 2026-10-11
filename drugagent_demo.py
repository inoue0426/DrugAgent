"""Browse saved DrugAgent evidence and predictions without API credentials."""

import csv
import json
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "output" / "summary_v0_ml_kg_rag_nomodel_none.csv"
PAYLOAD_DIR = ROOT / "output" / "input_payloads"


def load_examples():
    """Read only the checked-in examples and their local evidence payloads."""
    examples = []
    with SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            # Restrict payload access to the bundled directory; the CSV is not
            # allowed to point the demo at arbitrary files on the host machine.
            name = Path(row.get("input_payload", "")).name
            payload_path = PAYLOAD_DIR / name
            if not name or not payload_path.is_file():
                continue
            with payload_path.open(encoding="utf-8") as payload_file:
                payload = json.load(payload_file)
            examples.append((row, payload))
    if not examples:
        raise ValueError("No saved DrugAgent examples with evidence payloads found")
    return examples


EXAMPLES = load_examples()
CHOICES = [
    (f"{row['drug']} · {row['target']}", str(index))
    for index, (row, _) in enumerate(EXAMPLES)
]


def show_example(index):
    row, payload = EXAMPLES[int(index)]
    evidence = [payload.get(f"{name}_evidence") or {} for name in ("ml", "kg", "rag")]
    ml, kg, rag = evidence
    return (
        row.get("fusion_label", ""),
        row.get("fusion_conf", ""),
        row.get("fusion_reason", ""),
        row.get("summary_reasoning", ""),
        ml.get("label", row.get("ml_label", "")),
        str(ml.get("pKd", "")),
        ml.get("reason", ""),
        kg.get("label", row.get("kg_label", "")),
        str(kg.get("kg_direct", "")),
        kg.get("reason", ""),
        rag.get("label", row.get("rag_label", "")),
        ", ".join(map(str, rag.get("pmc_ids") or [])),
        rag.get("reason", ""),
    )


def build_demo():
    with gr.Blocks(title="DrugAgent · Saved results demo") as demo:
        gr.Markdown(
            "# DrugAgent · Evidence explorer\n"
            "Explore **saved examples** of drug–target predictions and the ML, "
            "knowledge-graph, and literature evidence used in their summaries. "
            "This viewer does not run a new prediction or contact an LLM."
        )
        pair = gr.Dropdown(choices=CHOICES, value=CHOICES[0][1], label="Drug · target")
        with gr.Row():
            decision = gr.Textbox(label="Combined label", interactive=False)
            confidence = gr.Textbox(label="Reported confidence", interactive=False)
        fusion = gr.Textbox(label="Combination rationale", lines=5, interactive=False)
        reasoning = gr.Textbox(
            label="Reasoning and uncertainties", lines=6, interactive=False
        )
        with gr.Accordion("ML evidence", open=False):
            ml_label = gr.Textbox(label="ML label", interactive=False)
            pkd = gr.Textbox(label="Predicted pKd", interactive=False)
            ml_reason = gr.Textbox(label="ML explanation", lines=6, interactive=False)
        with gr.Accordion("Knowledge-graph evidence", open=False):
            kg_label = gr.Textbox(label="KG label", interactive=False)
            direct = gr.Textbox(label="Direct graph edge reported", interactive=False)
            kg_reason = gr.Textbox(label="KG explanation", lines=6, interactive=False)
        with gr.Accordion("Literature evidence", open=False):
            rag_label = gr.Textbox(label="RAG label", interactive=False)
            pmc = gr.Textbox(label="Reported PMC IDs", interactive=False)
            rag_reason = gr.Textbox(
                label="Literature explanation", lines=6, interactive=False
            )
        outputs = [
            decision,
            confidence,
            fusion,
            reasoning,
            ml_label,
            pkd,
            ml_reason,
            kg_label,
            direct,
            kg_reason,
            rag_label,
            pmc,
            rag_reason,
        ]
        pair.change(show_example, inputs=pair, outputs=outputs)
        demo.load(show_example, inputs=pair, outputs=outputs)
        gr.Markdown(
            "**Scope:** These are previously generated outputs committed to the "
            "repository. Labels, confidence, and text are displayed as recorded, "
            "without independent verification. They do not establish clinical "
            "validity or performance on new drug–target pairs."
        )
    return demo


if __name__ == "__main__":
    build_demo().launch(share=False)
