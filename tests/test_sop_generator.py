import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


WORKSPACE = Path(__file__).resolve().parents[1]
SKILLS = WORKSPACE / "submission" / "JCIIOT" / "src" / "robot_agent" / "skills"
sys.path.insert(0, str(SKILLS))

from sop_generator import (  # noqa: E402
    CASE_SPECS,
    analyze_images,
    build_sop_record,
    extract_docx,
    parse_prompt_task,
    render_knowledge_markdown,
    build_parser,
)


def _write_docx(path: Path, paragraphs: list[str], images: dict[str, bytes]) -> None:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = []
    for paragraph in paragraphs:
        body.append(
            f'<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>'
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{namespace}"><w:body>'
        + "".join(body)
        + "</w:body></w:document>"
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
        for name, payload in images.items():
            archive.writestr(f"word/media/{name}", payload)


class SopGeneratorTests(unittest.TestCase):
    def test_extract_docx_uses_xml_and_records_image_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.docx"
            _write_docx(
                path,
                ["Prompt: Move one bin.", "Step 1: Navigate safely."],
                {"image2.png": b"second", "image1.png": b"first"},
            )

            extracted = extract_docx(path)

        self.assertEqual(
            extracted["paragraphs"],
            ["Prompt: Move one bin.", "Step 1: Navigate safely."],
        )
        self.assertEqual(
            [image["name"] for image in extracted["images"]],
            ["image1.png", "image2.png"],
        )
        self.assertEqual(
            extracted["images"][0]["sha256"],
            hashlib.sha256(b"first").hexdigest(),
        )
        self.assertEqual(extracted["images"][0]["data"], b"first")

    def test_parse_all_official_prompt_styles(self):
        examples = {
            1: (
                'Task Prompt: For this task, you need to transport a blue, hollow '
                'plastic box. Please move it from the starting point "Pick Station 2" '
                'to the destination "Place Station 3".'
            ),
            3: (
                "Current Task Material Information: Material Name: Green-rimmed storage "
                "bin Starting Location: Pick Station 1 Target Location: Place Station 3 "
                "Quantity to Transport: 1"
            ),
            5: (
                "Please follow the SOP. The object is a blue material transfer bin. "
                "The Pick Station is Pick Station 1, and the Place Station is Place Station 2."
            ),
            7: (
                "The object to be handled is a blue, hollow plastic box. The Pick Station "
                "is designated as Pick Station 5, and the Place Station is designated as "
                "Place Station 2."
            ),
            9: (
                "Move the three white-rimmed storage bins from Pick Station 6 to "
                "Place Station 1."
            ),
        }

        for case_number, prompt in examples.items():
            with self.subTest(case=case_number):
                facts = parse_prompt_task(prompt)
                spec = CASE_SPECS[case_number]
                self.assertEqual(facts["source_label"], spec["prompt_source"])
                self.assertEqual(facts["target_label"], spec["prompt_target"])
                self.assertEqual(facts["quantity"], spec["quantity"])
                for word in spec["material_keywords"]:
                    self.assertIn(word, facts["material"].lower())

    def test_vlm_analysis_is_evidence_only_and_hash_addressed(self):
        images = [
            {
                "name": "map.png",
                "sha256": hashlib.sha256(b"map").hexdigest(),
                "size_bytes": 3,
                "data": b"map",
            }
        ]
        calls = []
        expected_response = json.dumps(
            {
                "visible_labels": ["Pick Station 1"],
                "material_observations": [],
                "route_or_arrow_observations": [],
                "safety_observations": [],
                "uncertainties": [],
            }
        )

        def fake_vision(prompt, data):
            calls.append((prompt, data))
            return expected_response

        result = analyze_images(
            images,
            vision_callback=fake_vision,
            model_id="Qwen/Qwen3-VL-2B-Instruct",
        )

        self.assertEqual(result[0]["status"], "analyzed")
        self.assertEqual(
            result[0]["description"],
            json.dumps(
                {
                    "material_observations": [],
                    "route_or_arrow_observations": [],
                    "safety_observations": [],
                    "uncertainties": [],
                    "visible_labels": ["Pick Station 1"],
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], b"map")
        self.assertIn(
            "Do not propose robot actions",
            " ".join(calls[0][0].split()),
        )
        self.assertIn(
            "at most 4 unique strings",
            " ".join(calls[0][0].split()),
        )
        self.assertEqual(result[0]["model"], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(result[0]["attempts"], 1)
        self.assertEqual(
            result[0]["raw_response_sha256"],
            hashlib.sha256(expected_response.encode()).hexdigest(),
        )

    def test_invalid_vlm_json_is_not_marked_analyzed(self):
        images = [
            {
                "name": "map.png",
                "sha256": hashlib.sha256(b"map").hexdigest(),
                "size_bytes": 3,
                "data": b"map",
            }
        ]

        calls = []

        def invalid_response(_prompt, _data):
            calls.append(True)
            return "a prose-only answer"

        result = analyze_images(
            images,
            vision_callback=invalid_response,
            model_id="test-vlm",
        )

        self.assertEqual(result[0]["status"], "invalid_response")
        self.assertIn("a prose-only answer", result[0]["description"])
        self.assertEqual(result[0]["attempts"], 2)
        self.assertEqual(len(calls), 2)

    def test_parser_accepts_isolated_local_vlm_backend(self):
        args = build_parser().parse_args(
            [
                "--app-root",
                "/tmp/app",
                "--output-dir",
                "/tmp/output",
                "--use-vision",
                "--local-vlm-model",
                "Qwen/Qwen3-VL-2B-Instruct",
                "--vision-model-id",
                "Qwen/Qwen3-VL-2B-Instruct@abc123",
                "--local-vlm-device",
                "cuda:1",
            ]
        )

        self.assertEqual(args.local_vlm_model, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(
            args.vision_model_id,
            "Qwen/Qwen3-VL-2B-Instruct@abc123",
        )
        self.assertEqual(args.local_vlm_device, "cuda:1")

    def test_build_record_detects_body_conflict_and_validates_official_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docx = root / CASE_SPECS[3]["filename"]
            _write_docx(
                docx,
                [
                    "Prompt: Current Task Material Information:",
                    "Material Name: Green-rimmed storage bin",
                    "Starting Location: Pick Station 1",
                    "Target Location: Place Station 3",
                    "Quantity to Transport: 1",
                    "JCIIOT 2026 Standard Operating Procedure",
                    "Visually inspect the area around Pick Station 2.",
                    "Precisely locate the target material (blue hollow plastic bin).",
                ],
                {"factory.png": b"factory"},
            )
            task_config = root / "task_config.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {},
                            {
                                "level": "L2",
                                "source": "input_6",
                                "target": "output_4",
                                "object": [
                                    "green_tote_b01_upper",
                                    "green_tote_b01_lower",
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            semantic_map = root / "semantic.json"
            semantic_map.write_text(
                json.dumps(
                    {
                        "input_ports": {"input_6": {}},
                        "output_ports": {"output_4": {}},
                    }
                ),
                encoding="utf-8",
            )

            record = build_sop_record(
                docx_path=docx,
                case_number=3,
                task_config_path=task_config,
                semantic_map_path=semantic_map,
                vision_evidence=analyze_images(extract_docx(docx)["images"]),
            )

        self.assertEqual(record["level"], "L2")
        self.assertEqual(record["status"], "ready_with_resolved_conflicts")
        self.assertEqual(record["source"]["prompt_locator"], "paragraphs:1-5")
        self.assertTrue(record["validation"]["task_config_match"])
        self.assertTrue(record["validation"]["semantic_ports_present"])
        self.assertTrue(any("Pick Station 2" in item["observed"] for item in record["conflicts"]))
        self.assertTrue(any("blue hollow plastic bin" in item["observed"].lower() for item in record["conflicts"]))
        self.assertTrue(all(item["resolution"] for item in record["conflicts"]))

    def test_case_five_applies_published_erratum_without_hiding_raw_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docx = root / CASE_SPECS[5]["filename"]
            _write_docx(
                docx,
                [
                    "Prompt: The object is a blue material transfer bin. The Pick Station "
                    "is Pick Station 1, and the Place Station is Place Station 2.",
                    "JCIIOT 2026 Standard Operating Procedure",
                ],
                {},
            )
            task_config = root / "task_config.json"
            task_config.write_text(
                json.dumps(
                    {
                        "tasks": [
                            {},
                            {},
                            {
                                "level": "L3",
                                "source": "aux_input_1",
                                "target": "output_5",
                                "object": ["blue_tote_b01_far_right"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            semantic_map = root / "semantic.json"
            semantic_map.write_text(
                json.dumps(
                    {
                        "input_ports": {"aux_input_1": {}},
                        "output_ports": {"output_5": {}},
                    }
                ),
                encoding="utf-8",
            )

            record = build_sop_record(
                docx_path=docx,
                case_number=5,
                task_config_path=task_config,
                semantic_map_path=semantic_map,
                vision_evidence=[],
            )

        self.assertEqual(record["task"]["raw_source_label"], "Pick Station 1")
        self.assertEqual(record["task"]["effective_source_label"], "Placement Point 1")
        self.assertEqual(record["task"]["official_source"], "aux_input_1")
        self.assertEqual(record["task"]["source_resolution"], "ERRATUM Case 3")

    def test_markdown_contains_provenance_task_and_safety_contract(self):
        record = {
            "generator": "competition-sop-generator/1.0",
            "level": "L5",
            "case_number": 9,
            "status": "ready",
            "source": {
                "file": "JCIIOT 2026 case 9 SOP.docx",
                "sha256": "a" * 64,
                "prompt_locator": "paragraph:1",
            },
            "task": {
                "material": "three white-rimmed storage bins",
                "quantity": 3,
                "raw_source_label": "Pick Station 6",
                "effective_source_label": "Pick Station 6",
                "target_label": "Place Station 1",
                "official_source": "input_1",
                "official_target": "aux_output_1",
                "official_objects": ["white_tote_front", "white_tote_middle", "white_tote_back"],
            },
            "validation": {
                "prompt_match": True,
                "task_config_match": True,
                "semantic_ports_present": True,
            },
            "conflicts": [],
            "images": [
                {
                    "name": "map.png",
                    "sha256": "b" * 64,
                    "status": "not_analyzed",
                    "description": "",
                    "attempts": 0,
                    "raw_response_sha256": "c" * 64,
                }
            ],
        }

        markdown = render_knowledge_markdown(record)

        self.assertIn("# Generated SOP Knowledge - L5", markdown)
        self.assertIn("input_1", markdown)
        self.assertIn("aux_output_1", markdown)
        self.assertIn("Repeat the complete cycle until 3 objects are verified", markdown)
        self.assertIn("Stop immediately on collision", markdown)
        self.assertIn("aaaaaaaa", markdown)
        self.assertIn("cccccccc", markdown)
        self.assertNotIn("knowledge/sop5.md", markdown)

        record["task"]["quantity"] = 1
        singular_markdown = render_knowledge_markdown(record)
        self.assertIn(
            "Repeat the complete cycle until 1 object is verified",
            singular_markdown,
        )


if __name__ == "__main__":
    unittest.main()
