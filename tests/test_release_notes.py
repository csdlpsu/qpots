import pytest

from tools.extract_release_notes import extract_release_notes, validate_tag_version


def project_metadata(version: str = "2.1.0") -> str:
    return f'[project]\nname = "qpots"\nversion = "{version}"\n'


def test_tag_must_match_project_version():
    assert validate_tag_version("v2.1.0", project_metadata()) == "2.1.0"

    with pytest.raises(ValueError, match="does not match"):
        validate_tag_version("v2.1.1", project_metadata())


@pytest.mark.parametrize("tag", ["2.1.0", "release-2.1.0", "v2.1"])
def test_tag_must_use_semantic_version_form(tag):
    with pytest.raises(ValueError, match="vX.Y.Z"):
        validate_tag_version(tag, project_metadata())


def test_release_notes_are_extracted_from_matching_section():
    changelog = "# Changelog\n\n## Unreleased\n\n- Next\n\n## v2.1.0\n\n- Added API\n\n## v2.0.1\n\n- Fixed bug\n"

    assert extract_release_notes(changelog, "v2.1.0") == "- Added API\n"


def test_release_notes_require_a_nonempty_matching_section():
    with pytest.raises(ValueError, match="no section"):
        extract_release_notes("# Changelog\n\n## Unreleased\n\n- Next\n", "v2.1.0")

    with pytest.raises(ValueError, match="empty"):
        extract_release_notes("# Changelog\n\n## v2.1.0\n\n## v2.0.1\n\n- Old\n", "v2.1.0")
