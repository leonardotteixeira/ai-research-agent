"""Academic report generation -- a presentation layer built on top of an
already-persisted, already-complete `RunRecord`. Deliberately separate
from `app/reports/` (the existing, simpler MarkdownReportRenderer, which
this package does not replace) and from the research core itself:

    RunRecord (already persisted, already complete)
        -> AcademicReportBuilder   (app/academic/builder.py)
        -> AcademicReport          (app/academic/schemas.py, structured data)
        -> AcademicPDFRenderer     (app/academic/renderer.py, ABNT-oriented)
        -> academic_report.pdf

Nothing in this package ever calls an LLM, a tool, or the network -- it
only ever reads fields already sitting on a `RunRecord`. See
app/academic/citations.py for how claims/evidence/sources become
traceable, numbered citations, and README "Academic Report Generation"
for exactly which ABNT rules are (and are not) implemented.
"""
