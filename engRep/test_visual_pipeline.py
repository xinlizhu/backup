#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from extract_visual_records import add_token_usage, contains_cjk, has_token_usage, select_source_footnote
from visual_recall.evidence import build_evidence
from visual_recall.llm_client import validate_model_output
from visual_recall.render import render_labeled_image
from visual_recall.schema import FINAL_FIELDS, build_final_record, validate_final_record


SAMPLE = Path(os.environ.get("ENGREP_TEST_REPORT_DIR", "external_data/reference_report"))


@unittest.skipUnless(os.environ.get("ENGREP_TEST_REPORT_DIR"), "External reference report not supplied")
class VisualPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="visual_pipeline_test_")
        cls.output = Path(cls.temp.name)
        cls.evidences, cls.build_messages = build_evidence(SAMPLE, cls.output, context_blocks=10)
        cls.by_id = {item.visual_id: item for item in cls.evidences}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_sample_mapping(self) -> None:
        self.assertEqual(len([x for x in self.evidences if x.visual_type == "figure"]), 10)
        self.assertEqual(len([x for x in self.evidences if x.visual_type == "table"]), 7)
        figure = self.by_id["main_figure1"]
        self.assertEqual(figure.page_idx, 1)
        self.assertEqual(figure.bbox, [0.54, 0.335, 0.926, 0.455])
        self.assertIn("Risk vs. Reward", figure.original_caption)
        self.assertTrue(any("Downside Scenario" in text for text in figure.context_after))
        payload = figure.prompt_payload()
        self.assertNotIn("page_text", payload)
        self.assertIn("nearby_page_text", payload)
        self.assertLessEqual(len(payload["nearby_page_text"]), 8)
        local_context = {
            text.casefold()
            for text in payload["context_before"] + payload["context_after"]
        }
        self.assertFalse(local_context.intersection(text.casefold() for text in payload["nearby_page_text"]))

    def test_split_table_gets_neighbor_headers(self) -> None:
        table = self.by_id["main_table4"]
        self.assertEqual(table.page_idx, 1)
        self.assertEqual(table.original_caption, "WELLS FARGO vs. CONSENSUS")
        previous = [x for x in table.neighboring_visuals if x["relative_position"] == "before"]
        self.assertEqual(len(previous), 1)
        self.assertIn("2024A", previous[0]["table_body_excerpt"])
        self.assertIn("2026E", previous[0]["table_body_excerpt"])

    def test_source_is_grounded(self) -> None:
        figure = self.by_id["main_figure1"]
        source, provenance = select_source_footnote(figure)
        self.assertEqual(source, "Source: Wells Fargo Securities LLC research report, published 2024-08-29.")
        self.assertEqual(provenance, "institution_report_fallback")

    def test_original_visual_footnote_has_priority(self) -> None:
        table = self.by_id["main_table5"]
        source, provenance = select_source_footnote(table)
        self.assertEqual(source, "Source: Company Data; Wells Fargo Securities, LLC Estimates")
        self.assertEqual(provenance, "original_visual_footnote")

    def test_model_and_final_schema(self) -> None:
        model_output = {
            "decision": "keep",
            "category": "chart",
            "filter_reason": "",
            "title": "Dell Technologies price-target scenarios",
            "image_content": "The chart presents downside, base and upside price-target scenarios with their key values.",
            "is_blurry": False,
            "confidence": 0.95,
        }
        self.assertEqual(validate_model_output(model_output), [])
        with_source = {**model_output, "source_footnote": "Source: model-generated value"}
        self.assertTrue(any("unexpected model fields" in error for error in validate_model_output(with_source)))
        figure = self.by_id["main_figure1"]
        self.assertIsNotNone(figure.target_image)
        rendered = self.output / "rendered.jpg"
        render_labeled_image(
            figure.target_image,
            rendered,
            title=model_output["title"],
            source_footnote="Source: Wells Fargo Securities LLC estimates and FactSet.",
        )
        record = build_final_record(
            figure,
            title=model_output["title"],
            image_content=model_output["image_content"],
            image_file=rendered,
            source_footnote="Source: Wells Fargo Securities LLC estimates and FactSet.",
            language="en",
            is_blurry=False,
            publish_ready=True,
        )
        self.assertEqual(tuple(record), FINAL_FIELDS)
        self.assertEqual(validate_final_record(record), [])

    def test_english_output_guard(self) -> None:
        self.assertFalse(contains_cjk("Dell Technologies (DELL) Price-Target Scenarios"))
        self.assertTrue(contains_cjk("Dell Technologies 股价情景图"))

    def test_token_usage_aggregation(self) -> None:
        metadata = {
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}
        }
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.assertTrue(has_token_usage(metadata))
        self.assertTrue(add_token_usage(totals, metadata))
        self.assertEqual(
            totals,
            {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500},
        )
        self.assertFalse(has_token_usage({"usage": {}}))


if __name__ == "__main__":
    unittest.main()
