"""Bounded offline tests for source-only recovery; no Riot requests."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import refetch_missing_sources as refetch
from app.riot import RiotClient, RiotError


def source(mid, split='train', creation=1, paired=False):
    return {'match_id':mid, 'patch':'16.17', 'queue_id':420, 'legacy_split':split,
            'game_creation':creation, 'game_duration':60, 'raw_pair_exists':paired,
            'prior_probe_train_assignment':split in ('train', 'val'),
            'prior_probe_val_assignment':False}


def record(row):
    return {**{key:row[key] for key in refetch.RECORD_FIELDS},
            'existing_files':{'match':None, 'timeline':None}}


def payloads(row):
    participants = [{'participantId':i, 'puuid':str(i)} for i in range(1, 11)]
    return {
        'match':{'metadata':{'matchId':row['match_id']}, 'info':{
            'gameVersion':'16.17.1', 'queueId':420, 'gameCreation':row['game_creation'],
            'gameDuration':row['game_duration'], 'participants':participants}},
        'timeline':{'metadata':{'matchId':row['match_id']}, 'info':{
            'participants':participants, 'frames':[{'timestamp':0,
                'participantFrames':{str(i):{} for i in range(1, 11)}}]}}}


class OfflineClient:
    match = RiotClient.match
    timeline = RiotClient.timeline

    def __init__(self, rows, fail=None):
        self.responses = {r['match_id']:payloads(r) for r in rows}
        self.calls = []
        self.fail = fail
        self.closed = False

    def _get(self, url):
        bits = url.split('/')
        kind = 'timeline' if bits[-1] == 'timeline' else 'match'
        mid = bits[-2] if kind == 'timeline' else bits[-1]
        self.calls.append((mid, kind))
        if (mid, kind) == self.fail:
            raise RiotError('secret-like response must not reach status', 401)
        return copy.deepcopy(self.responses[mid][kind])

    def close(self):
        self.closed = True


class RefetchTests(unittest.TestCase):
    def test_scope_excludes_test_and_keeps_exposure_with_training_first(self):
        rows = [source('KR_VAL', 'val'), source('KR_TEST', 'test'),
                source('KR_OLD', creation=1), source('KR_NEW', creation=2),
                source('KR_SHORT', 'unassigned'), source('KR_PAIRED', paired=True)]
        selected, test, missing = refetch.select_sources({'games_metadata':rows})
        self.assertEqual([r['match_id'] for r in selected], ['KR_NEW','KR_OLD','KR_VAL','KR_SHORT'])
        self.assertEqual((test, missing), (1,5))
        self.assertTrue(selected[2]['prior_probe_train_assignment'])
        with self.assertRaises(ValueError):
            refetch.select_sources({'games_metadata':rows+[rows[0]]})

    def test_direct_test_guard_precedes_any_archive_or_client_access(self):
        with self.assertRaisesRegex(ValueError, 'Test source'):
            refetch.fetch_record(record(source('KR_TEST', 'test')), None, None)

    def test_missing_only_and_no_overwrite_even_publication_race(self):
        row = source('KR_SAMPLE')
        with tempfile.TemporaryDirectory() as temp:
            archive = refetch.AppendOnlyArchive(Path(temp))
            archive.write('match', row['match_id'], payloads(row)['match'])
            path = archive._path('match', row['match_id'])
            before = path.read_bytes()
            item = record(row)
            item['existing_files']['match'] = refetch.sha256(path)
            client = OfflineClient([row])
            client._raw_archive = archive
            result = refetch.fetch_record(item, client, archive)
            self.assertEqual(result['status'], 'raw_pair_recovered')
            self.assertEqual(client.calls, [('KR_SAMPLE','timeline')])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                archive.write('match', row['match_id'], payloads(row)['match'])
            self.assertEqual(path.read_bytes(), before)
            other = source('KR_RACE')
            target = archive._path('match', other['match_id'])
            def race(temporary, destination):
                destination.write_bytes(b'concurrent writer')
                raise FileExistsError('simulated race')
            with patch.object(refetch.os, 'link', side_effect=race):
                with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                    archive.write('match', other['match_id'], payloads(other)['match'])
            self.assertEqual(target.read_bytes(), b'concurrent writer')
            self.assertFalse(list(target.parent.glob('.*.tmp')))

    def test_changed_or_invalid_existing_archive_never_fetches(self):
        row = source('KR_SAMPLE')
        with tempfile.TemporaryDirectory() as temp:
            archive = refetch.AppendOnlyArchive(Path(temp))
            archive.write('match', row['match_id'], payloads(row)['match'])
            path = archive._path('match', row['match_id'])
            item = record(row)
            item['existing_files']['match'] = refetch.sha256(path)
            client = OfflineClient([row])
            client._raw_archive = archive
            path.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Pre-existing source changed'):
                refetch.fetch_record(item, client, archive)
            item['existing_files']['match'] = None
            self.assertEqual(refetch.fetch_record(item, client, archive)['status'], 'invalid_existing_archive')
            self.assertFalse(client.calls)

    def test_frozen_membership_schema_and_code_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            preview = base/'preview.json'
            archive = base/'raw'
            rows = [source('KR_TRAIN'), source('KR_TEST', 'test')]
            refetch.write_json(preview, {'games_metadata':rows})
            directory = base/'run'
            directory.mkdir()
            plan = {'schema':1, 'purpose':'missing_source_archive_only', 'patch':'16.17',
                'test_payload_reads':False, 'current_training_integration':False,
                'archive':str(archive), 'preview':str(preview), 'preview_sha256':refetch.sha256(preview),
                'records':[record(rows[0])], 'selected_games':1, 'reserved_test_games_not_requested':1,
                'missing_games_in_original_inventory':2, 'code_sha256':{}, 'protected_sha256':{}}
            with patch.object(refetch,'ARCHIVE',archive), patch.object(refetch,'CODE',()):
                refetch.write_json(directory/'plan.json', plan)
                self.assertEqual(refetch.read_plan(directory)['selected_games'], 1)
                for mutation in ('test', 'missing_field', 'code'):
                    bad = copy.deepcopy(plan)
                    if mutation == 'test': bad['records'] = [record(rows[1])]
                    elif mutation == 'missing_field': del bad['records'][0]['legacy_split']
                    else: bad['code_sha256']['unbound.py'] = '0'*64
                    refetch.write_json(directory/'plan.json', bad)
                    with self.assertRaises(ValueError):
                        refetch.read_plan(directory)

    def test_auth_failure_checkpoints_then_resumes_without_repeat_fetch(self):
        rows = [source('KR_ONE'), source('KR_TWO')]
        plan = {'records':[record(row) for row in rows]}
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'run'
            archive = Path(temp)/'raw'
            (directory/'results').mkdir(parents=True)
            refetch.write_json(directory/'plan.json', plan)
            first = OfflineClient(rows, fail=('KR_TWO','timeline'))
            fake_lease = SimpleNamespace(__exit__=lambda *args:None)
            with patch.object(refetch,'ARCHIVE',archive), \
                 patch.object(refetch,'read_plan',return_value=plan), \
                 patch.object(refetch,'assert_unchanged'), \
                 patch.object(refetch.time,'sleep'), \
                 patch.object(refetch.shutil,'disk_usage',return_value=SimpleNamespace(free=100e9)), \
                 patch('app.collector_lock.CollectorLease',return_value=fake_lease), \
                 patch.object(refetch,'RiotClient',return_value=first):
                self.assertEqual(refetch.worker(directory), 2)
                status = json.loads((directory/'status.json').read_text())
                self.assertEqual(status['reason'], 'riot_authentication_rejected')
                self.assertEqual(status['processed_games'], 1)
                self.assertEqual(status['http_status'], 401)
                self.assertNotIn('secret-like', (directory/'status.json').read_text())
                self.assertFalse((directory/'results/KR_TWO.json').exists())
                self.assertTrue(first.closed)
                second = OfflineClient(rows)
                with patch.object(refetch,'RiotClient',return_value=second):
                    self.assertEqual(refetch.worker(directory), 0)
                self.assertEqual(second.calls, [('KR_TWO','timeline')])
                self.assertEqual(json.loads((directory/'completion.json').read_text())['processed_games'], 2)
                result_path = directory/'results/KR_ONE.json'
                result = json.loads(result_path.read_text())
                result['files']['match']['path'] = str(archive/'match/KR_TWO.json.gz')
                refetch.write_json(result_path, result)
                with self.assertRaisesRegex(ValueError, 'Previously recovered archive changed'):
                    refetch.worker(directory)

    def test_daily_cutoff_and_five_hour_cap(self):
        dt = lambda h,m:datetime(2026,9,11,h,m,tzinfo=timezone.utc)
        self.assertEqual(refetch.collection_budget_seconds(dt(20,0)), 5*3600)
        self.assertEqual(refetch.collection_budget_seconds(dt(3,15)), 3600)
        for hour, minute in ((4,14),(4,15),(4,20),(4,29)):
            with self.assertRaises(ValueError):
                refetch.collection_budget_seconds(dt(hour,minute))


if __name__ == '__main__':
    unittest.main(verbosity=2)
