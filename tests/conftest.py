import pytest


@pytest.fixture
def minimal_report_data():
    return {
        "meta": {
            "title": "Test Report",
            "author": "Pytest",
            "date": "2026-05-18",
            "version": "1.0",
            "subtitle": "Automated test suite",
            "summary": "A minimal report generated during testing.",
        },
        "sections": [
            {
                "title": "Section 1",
                "blocks": [
                    {"type": "paragraph",     "text": "Hello world."},
                    {"type": "rich_text",     "text": "<b>Bold</b> and <i>italic</i>."},
                    {"type": "spacer"},
                    {"type": "line"},
                    {"type": "bullets",       "items": ["Bullet A", "Bullet B"]},
                    {"type": "numbered_list", "items": ["First", "Second"]},
                    {"type": "key_value",     "items": [{"key": "Key", "value": "Value"}]},
                    {"type": "callout",       "title": "Note", "text": "A test callout."},
                    {"type": "code",          "text": "x = 1 < 2\ny = x & True"},
                    {"type": "subsection",    "title": "Sub", "blocks": [
                        {"type": "paragraph", "text": "Nested paragraph."},
                    ]},
                    {"type": "page_break"},
                ],
            }
        ],
    }


@pytest.fixture
def tiny_corpus():
    """A small corpus of scientific-flavoured sentences for NLP tests."""
    return [
        "Agent-based models simulate the actions of autonomous agents.",
        "Cellular automata are discrete dynamic systems used in complex systems.",
        "Social simulation explores emergent behaviour in multi-agent systems.",
        "Epidemiological models track the spread of infectious disease in populations.",
        "Network analysis reveals the structure of social and biological systems.",
        "Complexity science studies how simple rules give rise to complex phenomena.",
        "Computational social science applies algorithms to understand human behaviour.",
        "Game theory analyses strategic interactions between rational decision makers.",
    ]
