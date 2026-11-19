"""The preregistered player-representation screen, without metric-driven tuning."""
import math

CANDIDATES=('deepset','attention')


def extra_gain(record):
    return record['relative_log_loss_reduction']>=.005 and record['log_loss']['ci95'][0]>0 and record['brier']['estimate']>=0


def select_player_candidate(comparisons,lead_comparisons,coverage,raw_seed_gains):
    gates={};eligible=[]
    for candidate in CANDIDATES:
        reference=comparisons[candidate+'_vs_combat_full'];global_control=comparisons[candidate+'_vs_global']
        lead=lead_comparisons[candidate+'_vs_combat_full'];seeds=raw_seed_gains[candidate]
        check={'complete_declared_scope':coverage['fit_games']==11468 and coverage['calibration_games']==2866
                    and coverage['development_games']==2861 and coverage['fit_sampled_frames']==64422,
            'all12_models_and30epochs':coverage['completed_models']==12 and coverage['minimum_completed_epochs']==30
                    and coverage['maximum_completed_epochs']==30,
            'three_seed_ensemble_and_inference_checks':coverage['three_seed_ensembles'] is True and coverage['inference_checks'] is True,
            'two_percent_loss_vs_strong_reference':reference['relative_log_loss_reduction']>=.02,
            'positive_loss_vs_strong_reference':reference['log_loss']['ci95'][0]>0,
            'positive_brier_vs_strong_reference':reference['brier']['ci95'][0]>0,
            'half_percent_resolved_gain_vs_global_neural':extra_gain(global_control),
            'minimum1700_lead_games':coverage['lead_games']>=1700,
            'half_percent_resolved_lead_gain':extra_gain(lead),
            'at_least_two_positive_raw_seeds':len(seeds)==3 and all(math.isfinite(v) for v in seeds) and sum(v>0 for v in seeds)>=2}
        if candidate=='attention':check['half_percent_resolved_gain_vs_position_shuffle']=extra_gain(comparisons['attention_vs_attention_position_shuffled'])
        gates[candidate]=check
        if all(check.values()):eligible.append(candidate)
    chosen=eligible[0] if len(eligible)==1 else None
    if len(eligible)==2:chosen='attention' if extra_gain(comparisons['attention_vs_deepset']) else 'deepset'
    return {'gates':gates,'eligible':eligible,'selected_candidate':chosen,'useful_development_signal':chosen is not None,
            'confirmation_completed':False,'practical_benefit_established':False,
            'cohort_status':'previously exposed development; exploratory selection'}
