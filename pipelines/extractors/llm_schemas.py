"""Pydantic schemas constraining Gemini structured outputs (no free-form dicts)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from pipelines.crawlers.product_loader import Category


class CategoryWeight(BaseModel):
    """A target category with a relevance weight in [0, 1]."""

    category: Category
    weight: float = Field(ge=0, le=1)


class ScriptLatentAnalysis(BaseModel):
    """Latent intent extracted from a short-video script."""

    dominant_mood: str
    latent_needs: list[str]
    recommended_categories: list[CategoryWeight]
    bridge_copy: str


class ProductContextProfile(BaseModel):
    """Usage context extracted from a product."""

    target_personas: list[str]
    usage_scenarios: list[str]
    solved_problems: list[str]


class ScriptAnalysisItem(ScriptLatentAnalysis):
    id: str


class ProductAnalysisItem(ProductContextProfile):
    id: str


class ScriptBatch(BaseModel):
    results: list[ScriptAnalysisItem]


class ProductBatch(BaseModel):
    results: list[ProductAnalysisItem]