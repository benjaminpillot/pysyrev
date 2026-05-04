"""
Diagnostic: dump the raw JSON paths relevant to cited_by and references
for the first record returned by the WoS API.

Usage:
    WOS_API_KEY=<your_key> python diag_wos_record.py
"""
import json, os

from dotenv import load_dotenv

from pysyrev.core.api.wos_client import WosClient

load_dotenv("/home/benjaminpillot/Documents/PRO/ABM_LITERATURE_REVIEW/.env")
api_key = os.environ.get('WOS_API_KEY')
if not api_key:
    raise SystemExit('Set WOS_API_KEY in your environment first.')

client = WosClient(api_key=api_key)
result = client.search('TS=("agent-based") AND PY=2023', max_records=1, show_progress=False)

if not result.records:
    raise SystemExit('No records returned.')

rec = result.records[0]

def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return f'<not a dict: {type(d).__name__}>'
        d = d.get(k)
        if d is None:
            return f'<missing after key {k!r}>'
    return d

print('=== cited_by path ===')
print('dynamic_data.citation_related:',
      json.dumps(_get(rec, 'dynamic_data', 'citation_related'), indent=2, default=str))

print('\n=== references path ===')
print('static_data.fullrecord_metadata.references:',
      json.dumps(_get(rec, 'static_data', 'fullrecord_metadata', 'references'), indent=2, default=str))

print('\n=== top-level keys ===')
print(list(rec.keys()))
print('\n=== dynamic_data keys ===')
print(list(_get(rec, 'dynamic_data').keys()) if isinstance(_get(rec, 'dynamic_data'), dict) else _get(rec, 'dynamic_data'))
