import unittest
import numpy as np
from winprob.wp_distillation import (privileged_fit_matrix,teacher_plan,fold_for_match,CurrentWinStudent,useful_gate)


class WinDistillationTests(unittest.TestCase):
    def test_joint_teacher_contains_current_information(self):
        x=np.arange(12,dtype=np.float32).reshape(4,3)
        fit,joint,future=privileged_fit_matrix(x,[0,0,1,1],[180025,360044,180000,360000],['fit','fit','development','development'])
        np.testing.assert_array_equal(fit,[0,1]);np.testing.assert_array_equal(future,[1,-1])
        np.testing.assert_array_equal(joint[:,:3],x[:2]);np.testing.assert_array_equal(joint[0,3:6],x[1])
        np.testing.assert_array_equal(joint[:,6],[1,0]);np.testing.assert_array_equal(joint[1,3:6],0)

    def test_external_feature_mutations_cannot_change_fit_teacher_matrix(self):
        x=np.arange(12,dtype=np.float32).reshape(4,3);meta=([0,0,1,1],[180025,360044,180000,360000],['fit','fit','calibration','calibration'])
        before=privileged_fit_matrix(x,*meta);x[2:]=np.nan;after=privileged_fit_matrix(x,*meta)
        for left,right in zip(before,after,strict=True):np.testing.assert_array_equal(left,right)

    def test_missing_future_retains_current_rows_and_never_crosses_games(self):
        fit,joint,future=privileged_fit_matrix(np.ones((3,2)),[0,1,1],[180000,360000,540000],['fit']*3)
        np.testing.assert_array_equal(fit,[0,1,2]);np.testing.assert_array_equal(future,[-1,2,-1])
        self.assertEqual(joint.shape,(3,5));self.assertEqual(int(joint[:,-1].sum()),1)

    def test_first_frame_of_target_clock_bin_is_deterministic(self):
        _,joint,future=privileged_fit_matrix(np.array([[1.],[3.],[2.]]),[0,0,0],[180001,361000,360010],['fit']*3)
        self.assertEqual(future[0],2);self.assertEqual(joint[0,1],2.)

    def test_duplicate_exact_frame_fails(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            privileged_fit_matrix(np.ones((2,2)),[0,0],[180000,180000],['fit','fit'])

    def test_game_cannot_cross_roles(self):
        with self.assertRaisesRegex(ValueError,'crosses roles'):
            privileged_fit_matrix(np.ones((2,2)),[0,0],[180000,360000],['fit','development'])

    @staticmethod
    def source():
        result={};i=0
        while len(result)<4:
            mid='synthetic-teacher-'+str(i);fold=fold_for_match(mid)
            if fold not in result:result[fold]={'match_id':mid,'role':'fit'}
            i+=1
        return result

    def test_teacher_fit_cal_target_game_sets_disjoint_and_complete(self):
        source=self.source();games=np.repeat(np.arange(4),3);folds,plans=teacher_plan(games,source,np.arange(12))
        for plan in plans:
            sets=[set(games[plan[key]]) for key in ('train','cal','target')]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            self.assertEqual(set.union(*sets),set(range(4)))
            self.assertEqual(set(folds[plan['target']]),{plan['heldout_fold']})
        np.testing.assert_array_equal(np.sort(np.concatenate([p['target'] for p in plans])),np.arange(12))

    def test_external_teacher_game_is_rejected(self):
        source=self.source();source[0]['role']='calibration'
        with self.assertRaisesRegex(ValueError,'external'):
            teacher_plan(np.repeat(np.arange(4),3),source,np.arange(12))

    def test_missing_sampled_game_is_not_silently_dropped(self):
        source=self.source()
        with self.assertRaisesRegex(ValueError,'drops FIT games|separation'):
            teacher_plan(np.repeat(np.arange(4),3),source,np.arange(3,12))

    def test_rao_blackwell_squared_risk_identity(self):
        weights=np.array([[.1,.2],[.3,.4]]);m=np.array([[.1,.7],[.4,.9]]);f=np.array([[.25],[.65]])
        binary_risk=np.sum(weights*(m*(f-1)**2+(1-m)*f**2))
        teacher_risk=np.sum(weights*(f-m)**2);noise=np.sum(weights*m*(1-m))
        self.assertAlmostEqual(binary_risk,teacher_risk+noise,places=14)
        # Future-only information is insufficient in the elementary Y=X world.
        conditional_y=np.array([0.,1.]);future_only_teacher=np.array([.5,.5])
        self.assertFalse(np.array_equal(conditional_y,future_only_teacher))

    def test_student_rejects_privileged_width_and_unbounded_targets(self):
        student=CurrentWinStudent(['a','b'])
        with self.assertRaisesRegex(ValueError,'CURRENT'):
            student.fit(np.ones((4,5)),np.zeros(4),np.arange(4))
        with self.assertRaisesRegex(ValueError,'bounded'):
            student.fit(np.ones((4,2)),np.full(4,1.1),np.arange(4))

    def test_student_inference_has_no_teacher_dependency(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(400,2));y=(x[:,0]>0).astype(int);games=np.repeat(np.arange(100),4)
        target=.2+.6*y
        student=CurrentWinStudent(['a','b']).fit(x,target,games).calibrate(x,y,games)
        original=student.predict(x);target[:]=1-target
        np.testing.assert_array_equal(student.predict(x),original)
        self.assertEqual(set(student.__dict__),{'feature_names','model','calibrator'})
        with self.assertRaisesRegex(ValueError,'CURRENT'):student.predict(np.zeros((1,5)))

    def test_primary_gate_cannot_be_rescued_by_one_comparison(self):
        good=lambda:{'relative_log_loss_reduction':.03,'log_loss':{'ci95':[.001,.02]},'brier':{'ci95':[.0001,.01]}}
        comparisons={'future_teacher_vs_'+key:good() for key in ('hard','frozen_rich_classifier','current_teacher')}
        self.assertTrue(all(useful_gate(comparisons).values()))
        comparisons['future_teacher_vs_frozen_rich_classifier']['brier']['ci95']=[-.001,.01]
        self.assertFalse(all(useful_gate(comparisons).values()))


if __name__=='__main__':unittest.main()
