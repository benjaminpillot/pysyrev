"""Tests for the canonical actors column (pysyrev.core.actors): the generic
core, each source adapter, the post-merge harmonisation, and the rankings /
collaboration networks built on it (pysyrev.core.networks.collaboration)."""

import numpy as np
import pandas as pd
import pytest

from pysyrev.core import actors as A
from pysyrev.core.actors import ACTORS, make_actors, parse_actors
from pysyrev.core.networks.collaboration import (
    actor_frame, build_collaboration, collaboration_matrix,
    collaboration_summary, rank_actors)


# =============================================================================
# Generic core
# =============================================================================

def test_make_actors_dedupes_and_unions_institution_countries():
    cell = make_actors(
        authors=[{"key": "orcid:X", "name": "A"}, {"key": "orcid:X", "name": "A bis"}],
        institutions=[{"key": "ror:1", "name": "U", "country": None},
                      {"key": "ror:1", "name": "U", "country": "FR"}],
        countries=["SN", None])
    p = parse_actors(cell)
    assert p["authors"] == [{"key": "orcid:X", "name": "A"}]
    # A later duplicate fills the field the first one left empty.
    assert p["institutions"] == [{"key": "ror:1", "name": "U", "country": "FR"}]
    assert p["countries"] == ["SN", "FR"]


def test_make_actors_without_any_actor_is_missing():
    assert make_actors() is None
    assert parse_actors(None) == A.empty_actors()
    assert parse_actors("not json") == A.empty_actors()


@pytest.mark.parametrize("text, code", [
    ("FR", "FR"), ("France", "FR"), ("Peoples R China", "CN"), ("England", "GB"),
    ("Scotland", "GB"), ("USA", "US"), ("South Korea", "KR"), ("Turkiye", "TR"),
    ("U Arab Emirates", "AE"), ("Cote Ivoire", "CI"), ("Dem Rep Congo", "CD"),
    ("Bosnia & Herceg", "BA"), ("Viet Nam", "VN"), ("Russian Federation", "RU"),
    ("Atlantis", None), ("", None), (None, None),
])
def test_country_code(text, code):
    assert A.country_code(text) == code


@pytest.mark.parametrize("address, code", [
    ("Tufts Univ, Friedman Sch, Boston, MA 02155 USA", "US"),
    ("Korea Adv Inst Sci & Technol, Dept Mech Engn, Daejeon 34141, South Korea", "KR"),
    ("Division of Energy Systems, KTH Royal Institute of Technology, Stockholm, Sweden.", "SE"),
    ("Some lab, Nowhere", None),
])
def test_country_from_address(address, code):
    assert A.country_from_address(address) == code


@pytest.mark.parametrize("address, inst", [
    ("Division of Energy Systems, KTH Royal Institute of Technology, Stockholm, Sweden",
     "KTH Royal Institute of Technology"),
    ("Dept Phys, Inst Fis, Univ Fed Fluminense, Niteroi, Brazil", "Univ Fed Fluminense"),
    ("Energy Program, International Institute of Applied Systems Analysis, Laxenburg, Austria",
     "International Institute of Applied Systems Analysis"),
])
def test_institution_from_address(address, inst):
    assert A.institution_from_address(address) == inst


@pytest.mark.parametrize("name, initials, full", [
    ("Pires, Marcelo A.", "pires m", "pires marcelo a"),
    ("Marcelo A. Pires", "pires m", "pires marcelo a"),
    ("Pires M.A.", "pires m", "pires m a"),
    ("Kim, H (Corresponding Author)", "kim h", "kim h"),
    ("Émile Chappin", "chappin e", "chappin emile"),
])
def test_person_name_norm(name, initials, full):
    assert A.person_name_norm(name) == initials
    assert A.person_name_norm(name, initials=False) == full


def test_institution_norm_bridges_wos_and_openalex_spellings():
    assert (A.institution_norm("Korea Advanced Institute of Science & Technology (KAIST)")
            == A.institution_norm("Korea Advanced Institute of Science and Technology"))
    assert A.institution_norm("The University of Texas at Austin") == "university of texas at austin"


@pytest.mark.parametrize("fn, value, key", [
    (A.orcid_key, "https://orcid.org/0000-0002-3611-534x", "orcid:0000-0002-3611-534X"),
    (A.orcid_key, "not-an-orcid", None),
    (A.ror_key, "https://ror.org/02wnmk332", "ror:02wnmk332"),
    (A.openalex_key, "https://openalex.org/A5033832888", "openalex:A5033832888"),
])
def test_identity_keys(fn, value, key):
    assert fn(value) == key


def test_actors_from_addresses_generic_fallback():
    p = parse_actors(A.actors_from_addresses(
        ["Kim H.", "Han S."],
        ["Dept Mech Engn, Korea Adv Inst Sci & Technol, Daejeon, South Korea",
         "Univ Tokyo, Tokyo, Japan"]))
    assert [a["key"] for a in p["authors"]] == ["name:kim h", "name:han s"]
    assert [i["name"] for i in p["institutions"]] == ["Korea Adv Inst Sci & Technol", "Univ Tokyo"]
    assert p["countries"] == ["KR", "JP"]


# =============================================================================
# Source adapters
# =============================================================================

def test_openalex_api_adapter():
    authorships = [
        {"author": {"id": "https://openalex.org/A1", "display_name": "Jane Doe",
                    "orcid": "https://orcid.org/0000-0001-0000-0001"},
         "institutions": [{"id": "https://openalex.org/I1", "display_name": "Univ A",
                           "ror": "https://ror.org/0aaa", "country_code": "FR"}],
         "countries": ["FR"]},
        # No ORCID → OpenAlex id; no matched institution → raw string fallback.
        {"author": {"id": "https://openalex.org/A2", "display_name": "John Roe"},
         "institutions": [],
         "raw_affiliation_strings": ["Dept X, University of Dakar, Dakar, Senegal"],
         "countries": []},
    ]
    p = parse_actors(A.actors_from_openalex(authorships))
    assert [a["key"] for a in p["authors"]] == ["orcid:0000-0001-0000-0001", "openalex:A2"]
    assert p["institutions"] == [
        {"key": "ror:0aaa", "name": "Univ A", "country": "FR"},
        {"key": "name:university of dakar", "name": "University of Dakar", "country": "SN"}]
    assert p["countries"] == ["FR", "SN"]


def test_openalex_csv_adapter_reads_flattened_columns():
    row = {
        "authorships.author.id": "https://openalex.org/A1|https://openalex.org/A2",
        "authorships.author.display_name": "Derek W. Bunn|Fernando S. Oliveira",
        "authorships.author.orcid": "https://orcid.org/0000-0003-1770-4488|",
        "authorships.institutions": (
            "{'id': 'https://openalex.org/I6', 'display_name': 'London Business School', "
            "'ror': 'https://ror.org/001c2sn75', 'country_code': 'GB', 'lineage': ['x']}|"
            "|{'id': 'https://openalex.org/I7', 'display_name': 'Univ | Pipe', "
            "'ror': None, 'country_code': 'FR', 'lineage': []}"),
        "authorships.countries": "GB|FR",
    }
    p = parse_actors(A.actors_from_openalex_csv(row))
    assert [a["key"] for a in p["authors"]] == ["orcid:0000-0003-1770-4488", "openalex:A2"]
    assert [i["key"] for i in p["institutions"]] == ["ror:001c2sn75", "openalex:I7"]
    assert p["institutions"][1]["name"] == "Univ | Pipe"
    assert p["countries"] == ["GB", "FR"]


def test_openalex_csv_adapter_drops_misaligned_orcids():
    row = {"authorships.author.id": "https://openalex.org/A1|https://openalex.org/A2",
           "authorships.author.display_name": "A One|B Two",
           "authorships.author.orcid": "https://orcid.org/0000-0003-1770-4488"}
    p = parse_actors(A.actors_from_openalex_csv(row))
    assert [a["key"] for a in p["authors"]] == ["openalex:A1", "openalex:A2"]


# Trimmed from a real WoS Expanded API response (names + addresses only).
_WOS_NAME = {
    "role": "author", "display_name": "Oestereich, Andre L.", "daisng_id": 12659230,
    "data-item-ids": {"data-item-id": [
        {"type": "person", "content": "DKN-8826-2022", "id-type": "PreferredRID"},
        {"type": "person", "content": "0000-0002-0050-4888", "id-type": "PreferredORCID"}]},
}
_WOS_RECORD = {
    "UID": "WOS:000434902400004",
    "static_data": {
        "summary": {"names": {"count": 3, "name": [
            {"role": "author", "display_name": "Pires, Marcelo A.", "daisng_id": 3053418,
             "orcid_id": "0000-0002-8333-7236"},
            _WOS_NAME,
            {"role": "author", "display_name": "Kim, Hyuncheol", "daisng_id": 10223153},
            {"role": "book_editor", "display_name": "Editor, Some"},
        ]}},
        "fullrecord_metadata": {"addresses": {"count": 2, "address_name": [
            {"address_spec": {
                "country": "Brazil", "full_address": "Ctr Brasileiro Pesquisas Fis, Rio De Janeiro, RJ, Brazil",
                "organizations": {"organization": [
                    {"pref": None, "content": "Ctr Brasileiro Pesquisas Fis"},
                    {"level": 0, "pref": "Y", "ror_id": "https://ror.org/02wnmk332",
                     "content": "Centro Brasileiro de Pesquisas Fisicas"}]}}},
            {"address_spec": {
                "country": "Brazil", "full_address": "Univ Fed Fluminense, Inst Fis, Niteroi, RJ, Brazil",
                "organizations": {"organization": [
                    {"pref": None, "content": "Univ Fed Fluminense"},
                    {"level": 1, "pref": "N", "content": "Federal Fluminense University Institute of Physics"},
                    {"level": 0, "pref": "Y", "ror_id": "https://ror.org/02rjhbb08",
                     "content": "Universidade Federal Fluminense"}]}}},
        ]}},
    },
}


def test_wos_api_adapter():
    p = parse_actors(A.actors_from_wos_record(_WOS_RECORD))
    assert [a["key"] for a in p["authors"]] == [
        "orcid:0000-0002-8333-7236", "orcid:0000-0002-0050-4888", "wos:10223153"]
    assert p["institutions"] == [
        {"key": "ror:02wnmk332", "name": "Centro Brasileiro de Pesquisas Fisicas", "country": "BR"},
        {"key": "ror:02rjhbb08", "name": "Universidade Federal Fluminense", "country": "BR"}]
    assert p["countries"] == ["BR"]


def test_wos_api_adapter_single_address_dict():
    rec = {"static_data": {"summary": {"names": {"name": {"role": "author", "display_name": "Kim, H"}}},
                           "fullrecord_metadata": {"addresses": {"address_name": {"address_spec": {
                               "country": "South Korea", "full_address": "KAIST, Daejeon, South Korea",
                               "organizations": {"organization": {"pref": None, "content": "KAIST"}}}}}}}}
    p = parse_actors(A.actors_from_wos_record(rec))
    assert p["authors"] == [{"key": "name:kim h", "name": "Kim, H"}]
    assert p["institutions"] == [{"key": "name:kaist", "name": "KAIST", "country": "KR"}]


_WOS_RAW_AFF = (
    "Sanga, U (Corresponding Author), Tufts Univ, Friedman Sch Nutr Sci \\& Policy, "
    "Boston, MA 02155 USA.   Sanga, Udita; Schluter, Maja, Stockholm Univ, Stockholm "
    "Resilience Ctr, Stockholm, Sweden.   Sanga, Udita, Tufts Univ, Friedman Sch Nutr "
    "Sci \\& Policy, Boston, MA 02155 USA.")


def test_wos_address_blocks_skip_reprint_blocks():
    blocks = A.wos_address_blocks(_WOS_RAW_AFF)
    assert blocks == [
        (["Sanga, Udita", "Schluter, Maja"], "Stockholm Univ, Stockholm Resilience Ctr, Stockholm, Sweden"),
        (["Sanga, Udita"], "Tufts Univ, Friedman Sch Nutr Sci & Policy, Boston, MA 02155 USA"),
    ]


def test_wos_file_adapter_with_and_without_organisations():
    with_orgs = parse_actors(A.actors_from_wos_file(
        "Sanga, Udita and Schluter, Maja", _WOS_RAW_AFF,
        "Stockholm University; Tufts University",
        "Sanga, Udita/0000-0003-0552-4797"))
    assert [a["key"] for a in with_orgs["authors"]] == ["orcid:0000-0003-0552-4797", "name:schluter m"]
    assert [i["name"] for i in with_orgs["institutions"]] == ["Stockholm University", "Tufts University"]
    assert with_orgs["countries"] == ["SE", "US"]

    without = parse_actors(A.actors_from_wos_file("Sanga, Udita and Schluter, Maja", _WOS_RAW_AFF))
    assert without["institutions"] == [
        {"key": "name:stockholm univ", "name": "Stockholm Univ", "country": "SE"},
        {"key": "name:tufts univ", "name": "Tufts Univ", "country": "US"}]


_WOS_BIB = """@article{ WOS:000442626800021,
Author = {Kim, Hyuncheol and Han, Soonhung},
Title = {Crowd evacuation simulation},
Journal = {SIMULATION MODELLING PRACTICE AND THEORY},
Year = {2018},
Type = {Article},
Language = {English},
Affiliation = {Kim, H (Corresponding Author), Korea Adv Inst Sci \\& Technol, Dept Mech Engn, Daejeon 34141, South Korea.
   Kim, Hyuncheol; Han, Soonhung, Korea Adv Inst Sci \\& Technol, Dept Mech Engn, Daejeon 34141, South Korea.},
DOI = {10.1016/j.simpat.2018.07.014},
Abstract = {Modern buildings have become larger.},
Affiliations = {Korea Advanced Institute of Science \\& Technology (KAIST)},
ORCID-Numbers = {han, soonhung/0000-0001-5676-8121},
Times-Cited = {12},
Unique-ID = {WOS:000442626800021},
}
"""


def test_generate_bib_wos_file_carries_actors(tmp_path):
    from pysyrev.core.bib import generate_bib
    path = tmp_path / "savedrecs.bib"
    path.write_text(_WOS_BIB, encoding="utf8")
    df = generate_bib(str(path), "wos", del_duplicated=True, print_log=False)
    assert ACTORS in df.columns
    p = parse_actors(df[ACTORS].iloc[0])
    assert [a["key"] for a in p["authors"]] == ["name:kim h", "orcid:0000-0001-5676-8121"]
    assert p["institutions"] == [{"key": "name:korea advanced institute of science and technology",
                                  "name": "Korea Advanced Institute of Science & Technology (KAIST)",
                                  "country": "KR"}]


def test_openalex_mapper_carries_actors():
    from pysyrev.core.mappers import from_openalex_records
    df = from_openalex_records([{
        "id": "https://openalex.org/W1", "title": "T", "publication_year": 2020,
        "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Jane Doe"},
                         "institutions": [{"id": "https://openalex.org/I1", "display_name": "U",
                                           "country_code": "SN"}]}]}])
    p = parse_actors(df[ACTORS].iloc[0])
    assert p["countries"] == ["SN"]


# =============================================================================
# Harmonisation
# =============================================================================

def _df(*cells):
    return pd.DataFrame({ACTORS: list(cells)})


def test_harmonize_rekeys_name_keys_onto_a_unique_identity():
    oa = make_actors([{"key": "orcid:1", "name": "Hyuncheol Kim"}],
                     [{"key": "ror:05apxxy63", "name": "Korea Advanced Institute of Science and Technology",
                       "country": "KR"}])
    wos = make_actors([{"key": "name:kim h", "name": "Kim, Hyuncheol"}],
                      [{"key": "name:korea advanced institute of science and technology",
                        "name": "Korea Advanced Institute of Science & Technology (KAIST)", "country": None}])
    out = A.harmonize_actors(_df(oa, wos))
    p = parse_actors(out[ACTORS].iloc[1])
    assert p["authors"][0]["key"] == "orcid:1"
    assert p["institutions"][0]["key"] == "ror:05apxxy63"
    # Idempotent.
    assert A.harmonize_actors(out)[ACTORS].tolist() == out[ACTORS].tolist()


def test_harmonize_leaves_ambiguous_names_alone():
    cells = [make_actors([{"key": "orcid:1", "name": "Li Wang"}]),
             make_actors([{"key": "orcid:2", "name": "Lei Wang"}]),
             make_actors([{"key": "name:wang l", "name": "Wang, L."}])]
    out = A.harmonize_actors(_df(*cells))
    assert parse_actors(out[ACTORS].iloc[2])["authors"][0]["key"] == "name:wang l"


def test_harmonize_without_column_is_noop():
    df = pd.DataFrame({"id": [1]})
    assert A.harmonize_actors(df) is df


# =============================================================================
# Rankings & collaboration networks
# =============================================================================

def _corpus():
    def inst(k, c):
        return {"key": f"ror:{k}", "name": f"Inst {k}", "country": c}
    rows = [
        # (authors, institutions, year, citations)
        (["a", "b"], [inst("1", "FR"), inst("2", "SN")], 2010, 10),
        (["a", "c"], [inst("1", "FR")], 2012, 4),
        (["a"],      [inst("1", "FR")], 2015, 0),
        (["b", "c"], [inst("2", "SN"), inst("3", "US")], 2020, 6),
        ([],         [], 2021, 1),
    ]
    return pd.DataFrame({
        "id": [f"W{i}" for i in range(len(rows))],
        "year": [r[2] for r in rows],
        "cited_by": [r[3] for r in rows],
        ACTORS: [make_actors([{"key": f"orcid:{a}", "name": a.upper()} for a in r[0]], r[1])
                 for r in rows],
    })


def test_actor_frame_shape():
    long = actor_frame(_corpus(), "country")
    assert set(long["key"]) == {"FR", "SN", "US"}
    assert long.loc[long["key"] == "FR", "name"].iloc[0] == "France"


def test_rank_actors_full_and_fractional_counting():
    r = rank_actors(_corpus(), "author", top=None).set_index("key")
    assert r.loc["orcid:a", "papers"] == 3
    assert r.loc["orcid:a", "fractional"] == pytest.approx(0.5 + 0.5 + 1)
    assert r.loc["orcid:a", "citations"] == 14
    assert (r.loc["orcid:a", "first_year"], r.loc["orcid:a", "last_year"]) == (2010, 2015)
    assert r.loc["orcid:a", "share"] == pytest.approx(3 / 4)   # 4 papers carry authors
    assert r.index[0] == "orcid:a"

    inst = rank_actors(_corpus(), "institution", top=2)
    assert list(inst["key"]) == ["ror:1", "ror:2"]
    assert inst["country"].tolist() == ["FR", "SN"]


def test_rank_actors_sort_by_citations_and_empty():
    r = rank_actors(_corpus(), "country", sort_by="citations")
    assert r["key"].iloc[0] == "SN"            # 10 + 6
    assert rank_actors(pd.DataFrame({"id": []}), "author").empty
    with pytest.raises(ValueError):
        rank_actors(_corpus(), "author", sort_by="h_index")


def test_collaboration_matrix_counts_shared_papers():
    keys, W, papers = collaboration_matrix(_corpus(), "author", min_papers=1,
                                           normalize="count")
    i = {k: n for n, k in enumerate(keys)}
    assert W[i["orcid:a"], i["orcid:b"]] == 1
    assert W[i["orcid:b"], i["orcid:c"]] == 1
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
    _, Ws, _ = collaboration_matrix(_corpus(), "author", min_papers=1, normalize="salton")
    assert Ws[i["orcid:a"], i["orcid:b"]] == pytest.approx(1 / np.sqrt(3 * 2))


def test_build_collaboration_returns_network_with_metadata():
    res = build_collaboration(_corpus(), "author", min_papers=1, min_size=1)
    assert res.n_nodes == 3
    assert res.node_meta["orcid:a"]["papers"] == 3
    with pytest.raises(ValueError):
        build_collaboration(_corpus(), "author", min_papers=10)


def test_collaboration_summary():
    s = collaboration_summary(_corpus())
    assert s["n_papers"] == 5 and s["with_author"] == 4 and s["n_country"] == 3
    assert s["single_author_share"] == pytest.approx(1 / 4)
    assert s["international_share"] == pytest.approx(2 / 4)


# =============================================================================
# Report section & config
# =============================================================================

def _blocks_of(section, btype):
    out = []
    for b in section["blocks"]:
        if b.get("type") == btype:
            out.append(b)
        if b.get("type") == "subsection":
            out.extend(x for x in b["blocks"] if x.get("type") == btype)
    return out


def test_actors_section_renders_rankings_and_networks(tmp_path):
    from pysyrev.core.config import ActorsSectionConfig
    from pysyrev.core.report.sections.actors import _build_actors_section

    cfg = ActorsSectionConfig(network={"min_papers": 1, "min_size": 1})
    section = _build_actors_section(_corpus(), cfg, str(tmp_path), 3)
    assert section["title"].startswith("3. Actors")
    subs = [b["title"] for b in section["blocks"] if b.get("type") == "subsection"]
    assert subs == ["Countries", "Institutions", "Authors"]
    tables = _blocks_of(section, "table")
    countries = next(t for t in tables if "countries" in t["title"])
    assert countries["rows"][0][1] == "France"
    assert len(countries["headers"]) == len(countries["rows"][0]) == len(countries["col_widths"])
    assert len(_blocks_of(section, "plotly")) == 3
    # Every actor carries an identifier here: no caveat about name matching.
    callout = _blocks_of(section, "callout")[0]
    assert "100% of authors" in callout["text"] and "homonyms" not in callout["text"]
    assert (tmp_path / "authors_collaboration_network.html").exists()


def test_actors_section_without_network_or_column():
    from pysyrev.core.config import ActorsSectionConfig
    from pysyrev.core.report.sections.actors import _build_actors_section

    cfg = ActorsSectionConfig(levels=["country"], network={"enabled": False})
    section = _build_actors_section(_corpus(), cfg, None, 1)
    assert not _blocks_of(section, "plotly")
    assert _build_actors_section(pd.DataFrame({"id": [1]}), cfg, None, 1) is None
    assert _build_actors_section(None, cfg, None, 1) is None


def test_actors_section_explains_a_missing_network():
    from pysyrev.core.config import ActorsSectionConfig
    from pysyrev.core.report.sections.actors import _build_actors_section

    cfg = ActorsSectionConfig(levels=["author"], network={"min_papers": 50})
    section = _build_actors_section(_corpus(), cfg, None, 1)
    texts = [b["text"] for b in _blocks_of(section, "paragraph")]
    assert any("No author collaboration network" in t for t in texts)


@pytest.mark.parametrize("kwargs", [
    {"levels": ["country", "lab"]}, {"sort_by": "h_index"}, {"top_n": 0},
    {"network": {"normalize": "jaccard"}}, {"network": {"max_nodes": 1}},
])
def test_actors_config_rejects_bad_values(kwargs):
    from pysyrev.core.config import ActorsSectionConfig
    with pytest.raises(ValueError):
        ActorsSectionConfig(**kwargs)


def test_report_sections_config_has_actors_defaults():
    from pysyrev.core.config import ReportSectionsConfig
    cfg = ReportSectionsConfig()
    assert cfg.actors.levels == ["country", "institution", "author"]
    assert cfg.actors.network.normalize == "salton"


def test_plot_network_size_by_and_node_text():
    from pysyrev.core.networks import plot_network
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    fig = plot_network(np.random.RandomState(0).rand(3, 2), np.array([0, 0, 0]), W,
                       size_by=np.array([1, 10, 100]), node_text=["", "B", "C"])
    texts = [t for t in fig.data if t.mode == "text"]
    assert len(texts) == 1 and list(texts[0].text) == ["B", "C"]
    sizes = [s for t in fig.data if t.mode == "markers" for s in t.marker.size]
    assert sizes == sorted(sizes)          # driven by size_by, not by degree


def test_sqrt_node_sizes_are_area_proportional():
    from pysyrev.core.networks.plotting import _node_sizes
    d = _node_sizes(None, size_min=0.0, size_max=40.0,
                    values=np.array([100.0, 400.0]), scale="sqrt")
    assert d[1] == pytest.approx(40.0) and d[0] == pytest.approx(20.0)  # 4x value → 4x area
    # The log scale (coupling default) is unchanged.
    assert _node_sizes(np.ones((2, 2)))[0] == pytest.approx(22.0)


def test_actors_network_sized_by_config():
    from pysyrev.core.config import ActorsSectionConfig
    from pysyrev.core.report.sections.actors import _build_actors_section
    cfg = ActorsSectionConfig(levels=["author"],
                              network={"min_papers": 1, "min_size": 1, "size_by": "papers"})
    fig = _blocks_of(_build_actors_section(_corpus(), cfg, None, 1), "plotly")[0]
    assert "number of papers" in fig["caption"]
    with pytest.raises(ValueError):
        ActorsSectionConfig(network={"size_by": "h_index"})
    with pytest.raises(ValueError):
        ActorsSectionConfig(network={"size_range": [40, 5]})


def test_build_collaboration_keeps_raw_counts():
    res = build_collaboration(_corpus(), "author", min_papers=1, min_size=1)
    i = {k: n for n, k in enumerate(res.node_ids)}
    assert res.counts[i["orcid:a"], i["orcid:b"]] == 1
    assert res.W[i["orcid:a"], i["orcid:b"]] == pytest.approx(1 / np.sqrt(3 * 2))


def test_sqrt_edge_widths_follow_values_not_weights():
    from pysyrev.core.networks.plotting import _edge_traces, _EDGE_LEVELS
    coords = np.zeros((3, 2))
    bb = [(0, 1, 0.9), (1, 2, 0.1)]                # strong relative tie, weak one
    counts = np.array([[0, 1, 0], [1, 0, 100], [0, 100, 0]], dtype=float)
    traces = _edge_traces(coords, bb, values=counts, scale="sqrt", width_range=(0.5, 6.0))
    widths = sorted(t.line.width for t in traces)
    assert widths[-1] == pytest.approx(6.0)        # the 100-paper link is the thickest
    assert widths[0] == pytest.approx(0.5 + 5.5 * round(0.1 * (_EDGE_LEVELS - 1)) / (_EDGE_LEVELS - 1))
    # The tertile default is unchanged: three fixed widths at most.
    assert {t.line.width for t in _edge_traces(coords, bb)} <= {0.6, 1.3, 2.2}


def test_actors_edge_width_config():
    from pysyrev.core.config import ActorsSectionConfig
    from pysyrev.core.report.sections.actors import _build_actors_section
    cfg = ActorsSectionConfig(levels=["author"], network={"min_papers": 1, "min_size": 1})
    fig = _blocks_of(_build_actors_section(_corpus(), cfg, None, 1), "plotly")[0]
    assert "papers co-signed" in fig["caption"]
    for bad in ({"edge_width_by": "jaccard"}, {"edge_width_range": [3, 1]}):
        with pytest.raises(ValueError):
            ActorsSectionConfig(network=bad)
