import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from winprob.future_goal_data import (DONOR_PREFIX,collapse_goals,kth_unexcluded,padded_baskets,public_tables,
                                     shuffle_training_goals,bind,bound)
from winprob import future_goal_data as module


def vocabulary():
    return [{'type':'NONE_FULL_HORIZON','goal_counts':[]},{'type':'NONE_TERMINAL_HORIZON','goal_counts':[]}]+[
        {'type':'SINGLE','goal_counts':[[i+1,1]]} for i in range(257)]


class FakeCompiler:
    item_index={10:1,20:2,30:3}
    static={p:{'items':{str(i):{} for i in ([10,20] if p=='16.17' else [10,20,30])}}
            for p in ('16.17','16.18','16.19')}
    def full_action(self,basket,kind,patch):
        if patch=='16.17' and any(i==30 for i,c in basket):raise KeyError('native item absent')
        vector=np.zeros(23,dtype=np.float32)
        if not basket:vector[-1]=1
        else:vector[0]=sum(c for i,c in basket)
        return tuple((self.item_index[i],c) for i,c in basket),vector


class FutureGoalDataTests(unittest.TestCase):
    def test_empty_diagnostic_collapse_preserves_all_nonempty_multisets(self):
        vocab=vocabulary();vocab[2]={'type':'MULTIPLE','goal_counts':[[1,2],[258,1]]}
        ids,goals=collapse_goals(np.array([0,1,2,258]),vocab)
        np.testing.assert_array_equal(ids,[0,0,1,257]);self.assertEqual(len(goals),258)
        self.assertEqual(goals[1]['goal_counts'],[[1,2],[258,1]])
        self.assertEqual(goals[0]['goal_counts'],[])

    def test_unknown_or_float_diagnostic_codes_fail(self):
        for ids in (np.array([259]),np.array([-1]),np.array([2.0])):
            with self.assertRaises(ValueError):collapse_goals(ids,vocabulary())

    def test_kth_allowed_matches_materialized_sorted_oracle_exhaustively(self):
        for total in range(1,9):
            for mask in range(1<<total):
                excluded=np.array([i for i in range(total) if mask&(1<<i)],dtype=np.int64)
                available=[i for i in range(total) if i not in excluded]
                for rank,wanted in enumerate(available):self.assertEqual(kth_unexcluded(excluded,rank,total),wanted)

    def test_padding_preserves_counts_without_truncation(self):
        ids,counts=padded_baskets([[],[[10,2],[20,3],[30,1]]],FakeCompiler.item_index)
        self.assertEqual(ids.shape,(2,3));np.testing.assert_array_equal(ids[1],[1,2,3]);np.testing.assert_array_equal(counts[1],[2,3,1])
        with self.assertRaises(ValueError):padded_baskets([[[10,1],[10,1]]],FakeCompiler.item_index)

    def test_main_SAVE_and_goal_EMPTY_distinct_native_invalid_composition_explicit(self):
        tables=public_tables(FakeCompiler(),[[],[[10,2]],[[30,1]]],[{'goal_counts':[]},{'goal_counts':[[20,2]]}])
        self.assertTrue((tables['main_static'][:,0,22]==1).all());self.assertTrue((tables['goal_static'][:,0]==0).all())
        self.assertFalse(tables['main_static_valid'][0,2]);self.assertTrue((tables['main_static'][0,2]==0).all())
        self.assertEqual(tables['main_item_indices'][2,0],3)
        self.assertTrue(tables['goal_static_valid'].all())

    def test_missing_native_goal_attributes_fail_instead_of_masking_goal(self):
        with self.assertRaisesRegex(ValueError,'Every fixed future goal'):
            public_tables(FakeCompiler(),[[],[[10,1]]],[{'goal_counts':[]},{'goal_counts':[[30,1]]}])

    def test_known_native_metadata_compilation_error_is_not_masked(self):
        class BrokenCompiler(FakeCompiler):
            def full_action(self,basket,kind,patch):
                if basket:raise ValueError('invalid known item metadata')
                return super().full_action(basket,kind,patch)
        with self.assertRaisesRegex(ValueError,'invalid known item metadata'):
            public_tables(BrokenCompiler(),[[],[[10,1]]],[{'goal_counts':[]}])

    def test_shuffle_exact_hash_whole_label_train_games_and_EMPTY_stratum(self):
        games=[{'match_id':f'g{i}'} for i in range(4)];game=np.repeat(np.arange(4),3)
        fold=np.repeat([0,1,0,1],3);ts=np.tile([0,100,200],4);main=np.tile([1,1,2],4);patch=np.zeros(12,dtype=int)
        real=np.array([1,2,0,3,4,0,5,6,0,7,8,0],dtype=np.int64)
        episodes=[{'first_state':i*3,'states':3,'participant_id':1} for i in range(4)]
        shuffled,donors,summary=shuffle_training_goals(real,main,patch,fold,game,ts,games,episodes)
        for heldout in (0,1):
            for row in range(12):
                if fold[row]==heldout:
                    self.assertEqual(shuffled[heldout,row],-1);self.assertEqual(donors[heldout,row],-1);continue
                pool=np.flatnonzero((fold!=heldout)&(game!=game[row])&(main==main[row])&(patch==patch[row])&((real==0)==(real[row]==0)))
                raw=f'{DONOR_PREFIX}{heldout}|g{game[row]}|1|{row%3}|{ts[row]}'
                expected=pool[int.from_bytes(hashlib.sha256(raw.encode()).digest(),'big')%len(pool)]
                self.assertEqual(donors[heldout,row],expected);self.assertEqual(shuffled[heldout,row],real[expected])
                self.assertEqual(shuffled[heldout,row]==0,real[row]==0)

    def test_missing_other_train_game_keeps_original_not_drops_or_broadens(self):
        real=np.array([2,0,3]);main=np.array([1,1,1]);patch=np.array([0,0,1]);fold=np.array([0,0,1]);game=np.array([0,0,1])
        eps=[{'first_state':0,'states':2,'participant_id':1},{'first_state':2,'states':1,'participant_id':1}]
        shuffled,donors,counts=shuffle_training_goals(real,main,patch,fold,game,np.array([1,2,1]),[{'match_id':'a'},{'match_id':'b'}],eps)
        np.testing.assert_array_equal(shuffled[1,:2],real[:2]);np.testing.assert_array_equal(donors[1,:2],[-1,-1])
        self.assertEqual(counts['1']['actually_changed_fraction'],0)

    def test_heldout_goal_mutation_cannot_change_training_shuffle(self):
        games=[{'match_id':f'g{i}'} for i in range(4)];game=np.arange(4);fold=np.array([0,1,0,1]);real=np.array([1,2,3,4])
        eps=[{'first_state':i,'states':1,'participant_id':1} for i in range(4)];args=(np.ones(4,int),np.zeros(4,int),fold,game,np.ones(4,int),games,eps)
        before=shuffle_training_goals(real,*args);changed=real.copy();changed[fold==0]=0;after=shuffle_training_goals(changed,*args)
        np.testing.assert_array_equal(before[0][0],after[0][0]);np.testing.assert_array_equal(before[1][0],after[1][0])

    def test_binding_conflict_and_reward_or_timing_reads_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('reward.npy','elapsed_ms.npy','terminated.npy'):
                path=Path(directory)/name;path.write_bytes(b'not to be opened')
                with self.assertRaisesRegex(ValueError,'Forbidden'):bind(path,'x'*64,{})
                with patch.object(module,'file_hash',side_effect=AssertionError('forbidden payload hashed')):
                    with self.assertRaisesRegex(ValueError,'Forbidden'):bound(path,{})
            path=Path(directory)/'meta.json';path.write_text('{}');digest=hashlib.sha256(b'{}').hexdigest()
            with self.assertRaisesRegex(ValueError,'Conflicting'):bind(path,digest,{str(path.resolve()):'x'*64})


if __name__=='__main__':unittest.main()
