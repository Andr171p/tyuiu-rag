from typing import Literal

from pydantic import BaseModel


class AnswerSchema(BaseModel):
    answer: str


class AnswerContent(AnswerSchema):
    status: Literal["ok"] = "ok"


class AnswerResponse(BaseModel):
    data: AnswerContent
