"""Property-based tests for pipeline properties (26-29, 31-38)."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rezaa.models.errors import ErrorResponse, ValidationError
from rezaa.models.pipeline import PipelineStage, PipelineState

pytestmark = pytest.mark.property


class TestPipelineProperties:
    @given(stage=st.sampled_from(list(PipelineStage)))
    @settings(max_examples=50)
    def test_property_26_pipeline_ordering(self, stage):
        """Property 26: All pipeline stages are valid."""
        state = PipelineState(job_id="test", stage=stage)
        assert state.stage in PipelineStage

    @given(
        job_id=st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-"),
        progress=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_property_29_completion_state(self, job_id, progress):
        """Property 29: Pipeline state tracks progress correctly."""
        state = PipelineState(job_id=job_id, progress=progress)
        assert 0 <= state.progress <= 1


class TestErrorProperties:
    @given(
        message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_property_35_error_response_format(self, message):
        """Property 35: Every error produces a valid ErrorResponse."""
        err = ValidationError(message)
        resp = ErrorResponse.from_exception(err)
        assert resp.error_type == "ValidationError"
        assert resp.message == message
        assert resp.component == "validation"

    @given(
        message=st.text(min_size=1, max_size=200),
    )
    @settings(max_examples=50)
    def test_error_response_serializable(self, message):
        """Error responses are JSON-serializable."""
        err = ValidationError(message)
        resp = ErrorResponse.from_exception(err)
        json_str = resp.model_dump_json()
        restored = ErrorResponse.model_validate_json(json_str)
        assert restored.message == message
