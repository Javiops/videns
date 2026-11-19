from copy import deepcopy
import unittest
from winprob.wp_scale_scope_v1 import plan_rows,ROLE_MAP,EXPECTED


def fixture():
    entries=[];pilot=[];number=0;old_counts={'train':4002,'calibration':1000,'development':998}
    for role,count in EXPECTED.items():
        for offset in range(count):
            row={'match_id':f'EUW1_{1000000000+number}','role':role,'historical_validation':False,'queue_id':420,
                 'patch':'16.17' if role=='train' else '16.18','game_creation_ms':number}
            entries.append(row)
            if offset<old_counts[role]:pilot.append({'game_index':len(pilot),'match_id':row['match_id'],
                'role':ROLE_MAP[role],'source_role':role,'patch':row['patch'],'game_creation_ms':row['game_creation_ms']})
            number+=1
    return {'games':entries},{'games':pilot}


class ScaleScopeTests(unittest.TestCase):
    def test_preserves_pilot_and_appends_only_safe_complement(self):
        manifest,pilot=fixture();rows=plan_rows(manifest,pilot,set())
        self.assertEqual(len(rows),17195)
        self.assertEqual([r['match_id'] for r in rows[:6000]],[r['match_id'] for r in pilot['games']])
        self.assertEqual([r['pilot_game_index'] for r in rows[:6000]],list(range(6000)))
        order=[(r['game_creation_ms'],r['match_id']) for r in rows[6000:]];self.assertEqual(order,sorted(order))

    def test_reserved_or_historical_training_identity_fails(self):
        manifest,pilot=fixture()
        with self.assertRaisesRegex(ValueError,'Unsafe'):plan_rows(manifest,pilot,{manifest['games'][8000]['match_id']})
        manifest['games'][8000]['historical_validation']=True
        with self.assertRaisesRegex(ValueError,'Unsafe'):plan_rows(manifest,pilot,set())

    def test_confirmation_and_exposed_roles_never_selected(self):
        manifest,pilot=fixture()
        for role,mid in [('confirmatory_reserved','EUW1_2000000001'),('development_exposed','EUW1_2000000002')]:
            manifest['games'].append({'match_id':mid,'role':role,'historical_validation':True,'queue_id':420,'patch':'16.19','game_creation_ms':99999})
        rows=plan_rows(manifest,pilot,{'EUW1_2000000001','EUW1_2000000002'})
        self.assertFalse({'EUW1_2000000001','EUW1_2000000002'}&{r['match_id'] for r in rows})

    def test_pilot_role_date_or_order_cannot_change(self):
        manifest,pilot=fixture();pilot['games'][0]['role']='development'
        with self.assertRaisesRegex(ValueError,'subset'):plan_rows(manifest,pilot,set())
        manifest,pilot=fixture();pilot['games'][0]['game_index']=7
        with self.assertRaisesRegex(ValueError,'order'):plan_rows(manifest,pilot,set())

if __name__=='__main__':unittest.main()
