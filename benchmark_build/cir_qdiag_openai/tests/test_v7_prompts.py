from pathlib import Path


PROMPTS = Path(__file__).parents[1] / "prompts"


def test_only_v7_prompts_are_active() -> None:
    assert {path.name for path in PROMPTS.glob("*.txt")} == {
        "generate_add_v7.txt",
        "generate_delete_v7.txt",
        "generate_substitute_v7.txt",
        "validate_group_v7.txt",
    }


def test_validation_contract_matches_group_design() -> None:
    prompt = (PROMPTS / "validate_group_v7.txt").read_text()
    assert "EVERY SUPPLIED POSITIVE" in prompt
    assert "At most eight Positive images" in prompt
    assert "do not require the weaker text to contradict" in prompt
    assert "official negative images" in prompt.casefold()
    assert "untrusted generator claims" in prompt
    assert "Reject as not_single_edit when the text is unchanged" in prompt
