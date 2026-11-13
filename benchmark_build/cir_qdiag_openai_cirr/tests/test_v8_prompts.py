from pathlib import Path


PROMPTS = Path(__file__).parents[1] / "prompts"


def test_only_v8_prompts_are_active() -> None:
    assert {path.name for path in PROMPTS.glob("*.txt")} == {
        "generate_add_v8.txt",
        "generate_delete_v8.txt",
        "generate_positive_paraphrases_v8.txt",
        "generate_substitute_v8.txt",
        "review_positive_paraphrases_v8.txt",
        "validate_visual_v8.txt",
    }


def test_generation_is_text_only_and_review_is_visual() -> None:
    generation = [
        (PROMPTS / f"generate_{operation}_v8.txt").read_text()
        for operation in ("add", "delete", "substitute")
    ]
    assert all("no images" in prompt for prompt in generation)
    assert all("null" in prompt for prompt in generation)
    review = (PROMPTS / "validate_visual_v8.txt").read_text()
    assert "Image 1" in review and "REFERENCE" in review
    assert "Image 2" in review and "TARGET POSITIVE" in review
    assert "clearly false in the TARGET image" in review


def test_paraphrase_prompts_preserve_meaning_without_images() -> None:
    generation = (PROMPTS / "generate_positive_paraphrases_v8.txt").read_text()
    review = (PROMPTS / "review_positive_paraphrases_v8.txt").read_text()
    assert "meaning-preserving" in generation
    assert "requested_variants" in generation
    assert "no images" in generation
    assert "fully semantically equivalent" in review
    assert "no images" in review
