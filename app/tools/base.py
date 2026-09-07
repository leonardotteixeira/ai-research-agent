"""The Tool interface — every capability the agent can invoke goes through
this. A tool declares its own input schema so the registry (app/tools/
registry.py) can validate arguments before execution; the registry is the
structural boundary that keeps tool invocation entirely code-controlled
(see registry.py's docstring for why that matters for security).
"""

from typing import Any, Protocol

from pydantic import BaseModel


class Tool(Protocol):
    name: str
    description: str
    input_schema: type[BaseModel]

    async def execute(self, **kwargs: Any) -> Any: ...
