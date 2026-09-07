"""Exception hierarchy for AI Research Agent."""


class ResearchAgentError(Exception):
    """Base class for all AI Research Agent errors."""


class ToolError(ResearchAgentError):
    """Base class for tool-related errors."""


class ToolNotFoundError(ToolError):
    """A ToolCall named a tool that isn't in the registry."""


class ToolValidationError(ToolError):
    """A ToolCall's arguments failed validation against the tool's input schema."""


class ToolExecutionError(ToolError):
    """The tool ran but failed (network error, parsing error, bad input, ...)."""


class ToolTimeoutError(ToolError):
    """The tool exceeded its per-tool timeout."""


class ProviderError(ResearchAgentError):
    """Base class for LLM/search provider errors."""


class ProviderTimeoutError(ProviderError):
    pass


class InvalidStructuredOutputError(ProviderError):
    """The provider returned a response that failed schema validation."""


class SecurityError(ResearchAgentError):
    """Base class for security-policy violations."""


class BlockedURLError(SecurityError):
    """FetchURLTool refused a URL (SSRF guard, disallowed scheme, etc.)."""


class ReportError(ResearchAgentError):
    """Base class for report generation errors."""


class ReportRenderingError(ReportError):
    """Failed to render a research report."""
