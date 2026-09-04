"""One module per report section, plus the helpers they share.

Each ``_build_*_section`` takes the run's dataframes and its slice of the
config, and returns the declarative section dict the PDF engine renders — or
``None`` when the data it needs is absent, which is how a section is skipped
rather than half-drawn.
"""
