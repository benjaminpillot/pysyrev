"""
Unit tests for WosClient.

Run all unit tests:
    pytest test_wos_client.py

Run the live integration test (requires WOS_API_KEY env var):
    pytest test_wos_client.py -m integration
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from pysyrev.core.api.wos_client import WosClient, WosSearchResult, _PAGE_SIZE, _MAX_RETRIES
from pysyrev.core.mappers import _wos_cited_by, _wos_references


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int, body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = str(body)
    if status_code < 400:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status_code}", response=resp
        )
    return resp


def _make_page(total_found: int, records: list) -> dict:
    """Minimal WoS Expanded API page structure."""
    return {
        'QueryResult': {'RecordsFound': total_found},
        'Data': {'Records': {'records': {'REC': records}}},
    }


def _make_record(uid: str) -> dict:
    return {'UID': uid}


def _client(session_file=None):
    return WosClient(api_key='test-key', session_file=session_file)


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestExtractTotalFound:
    def test_happy_path(self):
        assert WosClient._extract_total_found(
            {'QueryResult': {'RecordsFound': '42'}}
        ) == 42

    def test_integer_value(self):
        assert WosClient._extract_total_found(
            {'QueryResult': {'RecordsFound': 7}}
        ) == 7

    def test_missing_query_result(self):
        assert WosClient._extract_total_found({}) == 0

    def test_missing_records_found(self):
        assert WosClient._extract_total_found({'QueryResult': {}}) == 0

    def test_non_numeric_value(self):
        assert WosClient._extract_total_found(
            {'QueryResult': {'RecordsFound': 'N/A'}}
        ) == 0


class TestExtractRecords:
    def test_list_response(self):
        recs = [_make_record('A'), _make_record('B')]
        assert WosClient._extract_records(_make_page(2, recs)) == recs

    def test_single_dict_response(self):
        """API returns a bare dict (not a list) when exactly one record matches."""
        rec = _make_record('A')
        page = {'Data': {'Records': {'records': {'REC': rec}}}}
        assert WosClient._extract_records(page) == [rec]

    def test_missing_data_key(self):
        assert WosClient._extract_records({}) == []

    def test_missing_rec_key(self):
        assert WosClient._extract_records(
            {'Data': {'Records': {'records': {}}}}
        ) == []


# ---------------------------------------------------------------------------
# _fetch_page
# ---------------------------------------------------------------------------

class TestFetchPage:
    def test_success(self):
        client = _client()
        body = _make_page(5, [_make_record('A')])
        with patch.object(client._session, 'get', return_value=_make_response(200, body)):
            assert client._fetch_page('TS=test', 1, 10) == body

    def test_retries_on_429(self):
        client = _client()
        ok_body = _make_page(1, [_make_record('A')])
        with patch.object(client._session, 'get',
                          side_effect=[_make_response(429, {}),
                                       _make_response(200, ok_body)]):
            with patch('time.sleep'):
                assert client._fetch_page('TS=test', 1, 10) == ok_body

    def test_retries_on_500(self):
        client = _client()
        ok_body = _make_page(1, [_make_record('A')])
        with patch.object(client._session, 'get',
                          side_effect=[_make_response(500, {}),
                                       _make_response(200, ok_body)]):
            with patch('time.sleep'):
                assert client._fetch_page('TS=test', 1, 10) == ok_body

    def test_raises_after_max_retries(self):
        client = _client()
        with patch.object(client._session, 'get',
                          return_value=_make_response(500, {})):
            with patch('time.sleep'):
                with pytest.raises(RuntimeError, match=f'after {_MAX_RETRIES} retries'):
                    client._fetch_page('TS=test', 1, 10)

    def test_4xx_non_retryable_raises_immediately(self):
        """A 401/403 is not retried — raise_for_status propagates directly."""
        client = _client()
        with patch.object(client._session, 'get',
                          return_value=_make_response(401, {})):
            with patch('time.sleep') as mock_sleep:
                with pytest.raises(requests.HTTPError):
                    client._fetch_page('TS=test', 1, 10)
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class TestSessionManagement:
    def test_load_no_file(self, tmp_path):
        client = _client(session_file=str(tmp_path / 'session.json'))
        state = client._load_session('TS=test')
        assert state == {'query': 'TS=test', 'records': [],
                         'next_first_record': 1, 'total_found': None}

    def test_load_existing_file(self, tmp_path):
        session_file = tmp_path / 'session.json'
        saved = {'query': 'TS=test', 'records': [_make_record('A')],
                 'next_first_record': 101, 'total_found': 150}
        session_file.write_text(json.dumps(saved))

        state = _client(str(session_file))._load_session('TS=test')
        assert state['next_first_record'] == 101
        assert state['total_found'] == 150

    def test_load_ignores_stale_query(self, tmp_path):
        session_file = tmp_path / 'session.json'
        saved = {'query': 'TS=old', 'records': [_make_record('A')],
                 'next_first_record': 101, 'total_found': 150}
        session_file.write_text(json.dumps(saved))

        state = _client(str(session_file))._load_session('TS=new')
        assert state['records'] == []
        assert state['next_first_record'] == 1

    def test_save_creates_parent_dirs(self, tmp_path):
        session_file = tmp_path / 'subdir' / 'deep' / 'session.json'
        client = _client(str(session_file))
        client._save_session({'query': 'q', 'records': [],
                               'next_first_record': 1, 'total_found': 0})
        assert session_file.exists()

    def test_clear_removes_file(self, tmp_path):
        session_file = tmp_path / 'session.json'
        session_file.write_text('{}')
        client = _client(str(session_file))
        client._clear_session()
        assert not session_file.exists()

    def test_clear_no_file_is_noop(self, tmp_path):
        client = _client(str(tmp_path / 'missing.json'))
        client._clear_session()  # must not raise


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearch:
    def test_single_page(self):
        records = [_make_record(str(i)) for i in range(3)]
        page = _make_page(3, records)
        client = _client()
        with patch.object(client, '_fetch_page', return_value=page):
            result = client.search('TS=test', show_progress=False)
        assert isinstance(result, WosSearchResult)
        assert result.total_found == 3
        assert len(result.records) == 3

    def test_multi_page(self):
        p1_recs = [_make_record(str(i)) for i in range(_PAGE_SIZE)]
        p2_recs = [_make_record(str(i)) for i in range(_PAGE_SIZE, _PAGE_SIZE + 5)]
        total = _PAGE_SIZE + 5

        client = _client()
        with patch.object(client, '_fetch_page',
                          side_effect=[_make_page(total, p1_recs),
                                       _make_page(total, p2_recs)]):
            result = client.search('TS=test', show_progress=False)
        assert result.total_found == total
        assert len(result.records) == total

    def test_max_records_cap(self):
        """Returned list is trimmed to max_records even if the API has more."""
        records = [_make_record(str(i)) for i in range(50)]
        page = _make_page(200, records)
        client = _client()
        with patch.object(client, '_fetch_page', return_value=page):
            result = client.search('TS=test', max_records=10, show_progress=False)
        assert len(result.records) == 10

    def test_empty_result(self):
        page = _make_page(0, [])
        client = _client()
        with patch.object(client, '_fetch_page', return_value=page):
            result = client.search('TS=test', show_progress=False)
        assert result.total_found == 0
        assert result.records == []

    def test_session_cleared_on_success(self, tmp_path):
        session_file = tmp_path / 'session.json'
        client = _client(str(session_file))
        client._save_session({'query': 'TS=test', 'records': [],
                               'next_first_record': 1, 'total_found': None})
        page = _make_page(1, [_make_record('A')])
        with patch.object(client, '_fetch_page', return_value=page):
            client.search('TS=test', show_progress=False)
        assert not session_file.exists()

    def test_resumes_from_session(self, tmp_path):
        """An interrupted run continues from where it left off."""
        session_file = tmp_path / 'session.json'
        already = [_make_record('existing')]
        saved = {'query': 'TS=test', 'records': already,
                 'next_first_record': 2, 'total_found': 2}
        session_file.write_text(json.dumps(saved))

        remaining_page = _make_page(2, [_make_record('new')])
        client = _client(str(session_file))
        with patch.object(client, '_fetch_page', return_value=remaining_page):
            result = client.search('TS=test', show_progress=False)

        assert len(result.records) == 2
        assert result.records[0]['UID'] == 'existing'
        assert result.records[1]['UID'] == 'new'

    def test_empty_page_breaks_loop(self):
        """If the API stops returning records before total is reached, exit cleanly."""
        p1 = _make_page(100, [_make_record(str(i)) for i in range(_PAGE_SIZE)])
        p2 = _make_page(100, [])  # unexpected empty page mid-pagination
        client = _client()
        with patch.object(client, '_fetch_page', side_effect=[p1, p2]):
            result = client.search('TS=test', show_progress=False)
        assert len(result.records) == _PAGE_SIZE


# ---------------------------------------------------------------------------
# WoS mappers
# ---------------------------------------------------------------------------

def _record_with_cited_by(silo_tc):
    return {'dynamic_data': {'citation_related': {'tc_list': {'silo_tc': silo_tc}}}}


class TestWosCitedBy:
    def test_list_wos_entry(self):
        """Real API case: silo_tc is a list, pick the WOS entry."""
        silo_tc = [
            {'coll_id': 'WOK', 'local_count': 5},
            {'coll_id': 'WOS', 'local_count': 3},
            {'coll_id': 'INSPEC', 'local_count': 0},
        ]
        assert _wos_cited_by(_record_with_cited_by(silo_tc)) == 3

    def test_list_falls_back_to_wok(self):
        """WOS entry absent: fall back to WOK total."""
        silo_tc = [{'coll_id': 'WOK', 'local_count': 7}]
        assert _wos_cited_by(_record_with_cited_by(silo_tc)) == 7

    def test_list_no_wos_or_wok(self):
        """Neither WOS nor WOK present: return None."""
        silo_tc = [{'coll_id': 'INSPEC', 'local_count': 2}]
        assert _wos_cited_by(_record_with_cited_by(silo_tc)) is None

    def test_dict_form(self):
        """Legacy / single-database case: silo_tc is a plain dict."""
        silo_tc = {'coll_id': 'WOS', 'local_count': 10}
        assert _wos_cited_by(_record_with_cited_by(silo_tc)) == 10

    def test_missing_path(self):
        assert _wos_cited_by({}) is None

    def test_non_numeric_count(self):
        silo_tc = [{'coll_id': 'WOS', 'local_count': 'N/A'}]
        assert _wos_cited_by(_record_with_cited_by(silo_tc)) is None


class TestWosReferences:
    def test_file_sourced_record_with_uids(self):
        """BibTeX-sourced records carry references with uid fields."""
        record = {
            'static_data': {'fullrecord_metadata': {'references': {'reference': [
                {'uid': 'WOS:000A'},
                {'uid': 'WOS:000B'},
            ]}}}
        }
        assert _wos_references(record) == 'WOS:000A; WOS:000B'

    def test_api_sourced_record_no_references(self):
        """API search response omits fullrecord_metadata.references entirely."""
        record = {'static_data': {'fullrecord_metadata': {}}}
        assert _wos_references(record) is None

    def test_references_without_uid(self):
        """References present but no uid field: skip them."""
        record = {
            'static_data': {'fullrecord_metadata': {'references': {'reference': [
                {'citedAuthor': 'Smith J', 'year': '2020'},
            ]}}}
        }
        assert _wos_references(record) is None


# ---------------------------------------------------------------------------
# Integration (opt-in, hits the live API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_real_search():
    """Hits the live WoS Expanded API.

    Requires WOS_API_KEY to be set in the environment.
    Run with: pytest test_wos_client.py -m integration
    """
    import os
    api_key = os.environ.get('WOS_API_KEY')
    if not api_key:
        pytest.skip('WOS_API_KEY not set')

    client = WosClient(api_key=api_key)
    result = client.search(
        'TS=("agent-based") AND PY=2023',
        max_records=5,
        show_progress=False,
    )
    assert result.total_found > 0
    assert 1 <= len(result.records) <= 5
    assert all('UID' in r for r in result.records)
