""" Handles the pydantic structuring of model output."""

from typing import Optional

from pydantic import BaseModel, Field

class ModelResponse(BaseModel):
    answer: str = Field(description="The textual answer to the user's question.")
    code: Optional[str] = Field(default=None, description="The code snippet related to the answer, if applicable.")