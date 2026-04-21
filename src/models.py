from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """ """

    type: str


class Definition(BaseModel):
    """ """

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
    raw: str = ""


class Prompt(BaseModel):
    """ """

    prompt: str = Field(min_length=1)
