"""Minimal test to verify pytest collection works."""

import pytest


@pytest.mark.asyncio
async def test_minimal():
    """Minimal test."""
    assert True
