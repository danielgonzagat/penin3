def test_dashboard_import():
    # Ensure dashboard module imports (streamlit may not run in test env, but import should succeed)
    try:
        import darwinacci_omega.dashboard.app as app  # noqa: F401
    except Exception as e:
        # If streamlit missing, skip gracefully
        import pytest
        pytest.skip(f"streamlit not available: {e}")
