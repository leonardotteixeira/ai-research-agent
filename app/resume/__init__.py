"""Resume: continue a checkpointed-but-unfinished RunRecord (Fase 11E).

Distinct from `app/replay/` on purpose:
    replay  -- takes a FINISHED run and reproduces it 100% offline, using
               only data already persisted. Never touches OpenAI, a real
               tool, or the network. Proves internal consistency of the
               recorded trace; changes nothing on disk.
    resume  -- takes an UNFINISHED run (one whose checkpoint has no
               `research_result` yet) and continues its real execution --
               which can mean calling a real LLM and real tools again,
               exactly like the `run` command does. This is the only
               command capable of continuing execution; `replay` must
               never be made to do this.
"""

from app.resume.service import ResumeService

__all__ = ["ResumeService"]
