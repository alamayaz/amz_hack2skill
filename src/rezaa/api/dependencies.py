"""Dependency injection providers for FastAPI."""

from functools import lru_cache

from rezaa.config import Settings, get_settings
from rezaa.pipeline.manager import PipelineManager
from rezaa.storage.feature_store import FeatureStore
from rezaa.storage.temp_store import TempFileManager


@lru_cache
def get_pipeline_manager() -> PipelineManager:
    return PipelineManager()


@lru_cache
def get_temp_store() -> TempFileManager:
    return TempFileManager()


@lru_cache
def get_feature_store() -> FeatureStore:
    return FeatureStore()


def get_app_settings() -> Settings:
    return get_settings()
