from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from winprob.history_probe_data import (BaseCache,DONOR_PREFIX,FOCAL_PREFIX,VISIT_PREFIX,
    donor_indices,focal_pid,history_content_indices,history_timing,history_slots,make_selection,source_closures)
from winprob import history_probe_data as module


def episode(mid,pid,lo,n,game=0):
    return {'trajectory_id':f'{mid}:{pid}','match_id':mid,'participant_id':pid,'first_state':lo,'states':n,'game_index':game}


class HistoryProbeDataTests(unittest.TestCase):
    def test_own_code_is_bound_before_base_metadata_is_opened(self):
        calls=[]
        def code():calls.append('initial_code');return {}
        def base(*args):calls.append('base_metadata');raise RuntimeError('synthetic stop before inputs')
        with tempfile.TemporaryDirectory() as directory,patch.object(module,'initial_code_bindings',side_effect=code),patch.object(module,'BaseCache',side_effect=base):
            with self.assertRaisesRegex(RuntimeError,'synthetic stop'):
                module.prepare('unused',Path(directory)/'new','unused')
        self.assertEqual(calls,['initial_code','base_metadata'])

    def test_selection_hashes_exact_first8_focal_and_full_visit_digest(self):
        mid='EUW1_123456';pid=1+int.from_bytes(hashlib.sha256((FOCAL_PREFIX+mid).encode()).digest()[:8],'big')%10
        self.assertEqual(focal_pid(mid),pid)
        ts=np.array([101,202,303,404],dtype=np.int64)
        games=[{'match_id':mid,'game_index':0,'fold':0,'patch':'16.17'}]
        eps=[episode(mid,pid,0,4)]
        selected=make_selection(games,eps,ts)[0]
        oracle=min(range(4),key=lambda j:(hashlib.sha256(f'{VISIT_PREFIX}{mid}|{pid}|{j}|{ts[j]}'.encode()).digest(),int(ts[j]),j))
        self.assertEqual(selected['selected_state_index'],oracle)
        self.assertEqual(selected['source_node_ordinal'],oracle)
        self.assertEqual(selected['participant_id'],pid)

    def test_missing_focal_never_selects_other_player(self):
        mid='KR_987';pid=focal_pid(mid);other=pid%10+1
        selected=make_selection([{'match_id':mid,'game_index':0,'fold':1,'patch':'16.18'}],
                                [episode(mid,other,0,2)],np.array([10,20]))[0]
        self.assertEqual(selected['status'],'no_eligible_focal_episode')
        self.assertIsNone(selected['selected_state_index'])
        self.assertEqual(selected['participant_id'],pid)

    def test_closure_tie_masks_without_replacement_by_older_purchase(self):
        ts=np.array([0,100,200,300]);ends=np.array([50,200,250,320]);ids=np.array([1,2,3,4])
        prev,present,age,counts=history_slots([episode('a',1,0,4)],ts,ends,ids)
        np.testing.assert_array_equal(prev[2],[-1,0])
        np.testing.assert_array_equal(present[2],[False,True])
        self.assertEqual(age[2,0],0)
        self.assertEqual(age[2,1],np.float32(150/360000))
        self.assertEqual(counts['closure_tied_slots_masked'],1)

    def test_save_removed_before_adjacency_but_ward_purchase_retained(self):
        ts=np.array([0,100,200,300]);ends=ts+10;ids=np.array([8,0,2,9])
        prev,present,age,counts=history_slots([episode('a',1,0,4)],ts,ends,ids)
        np.testing.assert_array_equal(prev[3],[2,0])
        self.assertEqual(ids[prev[3,0]],2)
        self.assertEqual(counts['synthetic_save_nodes_excluded_before_adjacency'],1)

    def test_inserting_future_qualified_save_node_cannot_change_purchase_history(self):
        ts=np.array([0,100,300]);ends=ts+10;ids=np.array([7,2,9])
        before=history_slots([episode('a',1,0,3)],ts,ends,ids)
        ts2=np.array([0,50,100,300]);ends2=ts2+10;ids2=np.array([7,0,2,9])
        after=history_slots([episode('a',1,0,4)],ts2,ends2,ids2)
        np.testing.assert_array_equal(ids[before[0][2]],ids2[after[0][3]])
        np.testing.assert_array_equal(before[1][2],after[1][3])
        np.testing.assert_array_equal(before[2][2],after[2][3])

    def test_current_and_future_actions_do_not_change_current_history(self):
        ts=np.array([0,100,200,300]);ends=ts+10;ids=np.array([1,2,3,4])
        a=history_slots([episode('a',1,0,4)],ts,ends,ids)
        changed=ids.copy();changed[2:]=[0,99]
        b=history_slots([episode('a',1,0,4)],ts,ends,changed)
        for aa,bb in zip(a[:3],b[:3]):np.testing.assert_array_equal(aa[2],bb[2])
        np.testing.assert_array_equal(ids[a[0][2]],changed[b[0][2]])
        self.assertFalse(np.array_equal(a[0][3],b[0][3]))

    def test_no_episode_crossing_and_no_age_clipping(self):
        ts=np.array([0,500000,100,700000]);ends=ts+10;ids=np.array([1,2,3,4])
        prev,present,age,_=history_slots([episode('a',1,0,2),episode('a',2,2,2)],ts,ends,ids)
        np.testing.assert_array_equal(prev[2],[-1,-1])
        self.assertGreater(age[1,0],1)
        self.assertFalse(present[2].any())

    def test_overlapping_purchase_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            history_slots([episode('a',1,0,2)],np.array([0,10]),np.array([20,30]),np.array([1,2]))

    def test_donor_exact_recipient_hash_trainfold_differentgame_and_matching(self):
        games=[{'match_id':f'g{i}'} for i in range(4)]
        eps=[episode(f'g{i}',1,2*i,2,i) for i in range(4)]
        ts=np.array([100,200]*4);champ=np.full((8,10),7);pi=np.zeros(8,dtype=np.uint8)
        fold=np.repeat([0,1,1,0],2);gi=np.repeat(np.arange(4),2);present=np.ones((8,2),dtype=bool)
        donors,summary=donor_indices(games,eps,ts,champ,pi,fold,gi,present)
        for heldout in (0,1):
            for row in range(8):
                eligible=np.flatnonzero((fold!=heldout)&(gi!=gi[row]))
                raw=f'{DONOR_PREFIX}{heldout}|g{gi[row]}|1|{row%2}|{ts[row]}'
                expected=eligible[int.from_bytes(hashlib.sha256(raw.encode()).digest(),'big')%len(eligible)]
                self.assertEqual(donors[heldout,row],expected)
                self.assertNotEqual(fold[expected],heldout)
                self.assertNotEqual(gi[expected],gi[row])
        # A different native patch must not borrow the otherwise matching row.
        pi[0]=1
        different,_=donor_indices(games,eps,ts,champ,pi,fold,gi,present)
        self.assertEqual(different[0,0],-1)
        present[1]=[True,False]
        absent,_=donor_indices(games,eps,ts,champ,pi,fold,gi,present)
        self.assertEqual(absent[0,1],-1)

    def test_donor_only_copies_content_and_never_ages_or_current_action(self):
        previous=np.array([[-1,-1],[0,-1],[1,0],[2,1]])
        donors=np.array([[3,-1,1,2],[2,3,0,1]])
        original=previous.copy()
        np.testing.assert_array_equal(history_content_indices(previous,donors,[0,1],'donor',0),[[2,1],[-1,-1]])
        np.testing.assert_array_equal(history_content_indices(previous,donors,[2],'history',0),[[1,0]])
        np.testing.assert_array_equal(history_content_indices(previous,donors,[2],'state',0),[[-1,-1]])
        np.testing.assert_array_equal(original,previous)
        present=np.array([[True,False],[True,True]]);age=np.array([[.1,0],[.3,.8]],dtype=np.float32)
        for arm in ('history','donor'):
            pp,aa=history_timing(present,age,[0,1],arm)
            np.testing.assert_array_equal(pp,present);np.testing.assert_array_equal(aa,age)
        pp,aa=history_timing(present,age,[0,1],'state')
        self.assertFalse(pp.any());self.assertFalse(aa.any())

    def test_target_arrays_cannot_open_before_selection_and_reward_never_opens(self):
        cache=BaseCache.__new__(BaseCache);cache.selection_frozen=False
        for name in ('full_action_id','main_group_id'):
            with self.assertRaisesRegex(ValueError,'before selection'):cache.array(name)
        cache.selection_frozen=True
        for name in ('reward','terminated','next_state_index','fold_action_counts'):
            with self.assertRaisesRegex(ValueError,'outside history-probe scope'):cache.array(name)

    def test_stream_sidecar_aligns_all_nodes_but_never_requires_terminal_fields(self):
        ep={**episode('g',1,0,2),'source_index':0,'patch':'16.17'}
        record={'trajectory_id':'g:1','match_id':'g','participant_id':1,'role':'fit','consumer_role':'fit','patch':'16.17',
                'whole_player_game_eligible':True,'ineligibility_reasons':[],
                'terminal':'Deliberately not a valid terminal mapping: this adapter must not inspect it.',
                'decisions':[{'node_id':f'g:1:{10+i}','event_index':10+i,'timestamp_ms':i*100,'visit_end_ms':i*100+5,
                              'kind':'decision','action':{'label_counts':{'2003':1},'kind':'retained_net_purchase',
                              'wards_preserved':True,'full_action_id':1,'main_group_id':1}} for i in range(2)]}
        encoded=(json.dumps(record)+'\n').encode()
        class Cache:
            selection_frozen=True
            episodes=[ep]
            def array(self,name):return {'timestamp_ms':np.array([0,100]),'full_action_id':np.array([1,1]),'main_group_id':np.array([1,1])}[name]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episodes.gz'
            with gzip.open(path,'wb') as stream:stream.write(encoded)
            bundle={'sources':[{'rows_path':str(path),'manifest':{'patch':'16.17','uncompressed_rows_sha256':hashlib.sha256(encoded).hexdigest(),
                                                               'trajectory_count':1,'counts':{'decision_nodes':2}}}],
                    'lookup':{('retained_net_purchase',((2003,1),)):1},'catalogue':{'full_to_group':[0,1]}}
            ends,event,summary=source_closures(Cache(),bundle)
            np.testing.assert_array_equal(ends,[5,105]);np.testing.assert_array_equal(event,[10,11])
            self.assertEqual(summary[0]['nodes'],2)
            blocked=Cache();blocked.selection_frozen=False
            with self.assertRaisesRegex(ValueError,'precede'):source_closures(blocked,bundle)


if __name__=='__main__':unittest.main()
