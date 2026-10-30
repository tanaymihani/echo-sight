from echosight.chunking import chunk_sections
from echosight.config import PDF_PATH
from echosight.document import _continues, normalize, parse_pdf

SECTIONS = parse_pdf(PDF_PATH)


def by_title(title):
    return next(s for s in SECTIONS if s.title == title)


def test_normalize_fixes_pdf_typography():
    assert normalize("ﬁle an appeal") == "file an appeal"
    assert normalize("FAILURE TO PRODUCE\xa0NORTHEASTERN") == "FAILURE TO PRODUCE NORTHEASTERN"
    assert normalize("1 . Failure to comply") == "1. Failure to comply"


def test_page_break_continuation():
    assert _continues("the Hearing Administrator will", "review the information")
    assert not _continues("ends with a period.", "New paragraph starts here")
    assert not _continues("Coercing Illegal Activity", "• Theft, trespassing, vandalism")
    assert not _continues("To prevent a student from acting.", "b. To restrain a student")


def test_heading_tree():
    assert len(SECTIONS) == 78
    assert by_title("FAILURE TO COMPLY").path == ("Standards of Conduct", "FAILURE TO COMPLY")
    assert by_title("Failure to Appear").path[1] == "DESCRIPTION OF ADMINISTRATIVE HEARINGS"
    # Headings with punctuation and title-case headings, both missed by v1's regex.
    assert by_title("ALCOHOL, MARIJUANA, OTHER DRUGS, AND MEDICAL AMNESTY")
    assert by_title("Procedures for Student Appeals").path == ("Procedures for Student Appeals",)


def test_intro_is_kept_and_headers_are_dropped():
    assert "Adopted November 12, 1971" in SECTIONS[0].text
    text = "\n".join(s.text for s in SECTIONS)
    assert "Code of Student Conduct 3" not in text
    assert "ﬁ" not in text and "\xa0" not in text


def test_page_split_paragraph_is_rejoined():
    sanctions = by_title("Sanctions").text
    assert "A list of possible sanctions follows." in sanctions


def test_chunks_respect_size_and_sections():
    for size in (400, 800):
        chunks = chunk_sections(SECTIONS, size)
        assert len({c.id for c in chunks}) == len(chunks)
        texts = {s.path: s.text for s in SECTIONS}     # titles repeat, e.g. "Notification"
        for chunk in chunks:
            assert all(line in texts[chunk.path] for line in chunk.text.split("\n"))
            assert len(chunk.text) <= size or "\n" not in chunk.text


def test_indexed_text_carries_heading_path():
    chunk = next(c for c in chunk_sections(SECTIONS, 800) if c.section == "DOXING")
    assert chunk.indexed_text().startswith("Standards of Conduct > DOXING\n")
    assert chunk.indexed_text(headers=False) == chunk.text
