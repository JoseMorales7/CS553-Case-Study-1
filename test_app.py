from app import build_prompt, preview_upload

def test_build_prompt_default_rubric():
    """Test that the default rubric is used when an empty string is provided."""
    prompt = build_prompt("")
    assert "Evaluate the artwork as a thoughtful art critic" in prompt
    assert "## Score: X/10" in prompt

def test_build_prompt_custom_rubric():
    """Test that a custom rubric is successfully injected into the prompt."""
    custom_rubric = "Analyze the historical significance of this piece."
    prompt = build_prompt(custom_rubric)
    assert custom_rubric in prompt
    assert "## Score: X/10" in prompt
    assert "Evaluate the artwork as a thoughtful art critic" not in prompt

def test_preview_upload_none():
    """Test the preview logic when no file path is provided."""
    preview, path, status = preview_upload(None)
    assert preview is None
    assert path is None
    assert status == "Upload an image to begin."

def test_preview_upload_invalid_file():
    """Test the preview logic when an invalid file is provided."""
    preview, path, status = preview_upload("non_existent_file.jpg")
    assert preview is None
    assert path is None
    assert "Could not read that image" in status
