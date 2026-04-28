from pydantic import BaseModel, Field


class Parameter(BaseModel):
    """

    Pydantic model class of a nested object the key-value pair in a function
    definition. Used for serlization/confirmity with pydantic.

    """

    type: str


class Definition(BaseModel):
    """

    Pydantic model class of a function definitions. Used for serilization and
    validation with the help of pydantic.

    """

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class Prompt(BaseModel):
    """

    Pydantic model class of a user prompt. Used for serilization and
    validation with the help of pydantic.

    """

    prompt: str = Field(min_length=1)
