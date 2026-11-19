"""Strict current-input observational WP models and the fixed screen decision."""
import numpy as np
from .experiment import CalibratedModel

CANDIDATES=('health_availability','combat_resources')


class CurrentObservedModel(CalibratedModel):
    def __init__(self,feature_names):
        self.feature_names=tuple(feature_names)
        if not self.feature_names or len(set(self.feature_names))!=len(self.feature_names):raise ValueError('Distinct current feature names required')
        super().__init__('snapshot',list(range(len(self.feature_names))))
        self.forecast_scope='retrospective_OMNI_observational_current_state'
        self.supports_counterfactual_items=False
        self.live_safe=False

    def _current(self,x):
        x=np.asarray(x)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Exactly the declared finite CURRENT feature columns are required')
        return x

    def fit(self,x,y,games):return super().fit(self._current(x),y,games)
    def raw(self,x):return super().raw(self._current(x))
    def calibrate(self,x,y,games):return super().calibrate(self._current(x),y,games)


def choose_candidate(comparisons,lead_comparisons,lead_games):
    gates={};eligible=[]
    for name in CANDIDATES:
        overall=comparisons[name+'_vs_reference'];lead=lead_comparisons.get(name+'_vs_reference')
        checks={'two_percent_overall_log_loss':overall['relative_log_loss_reduction']>=.02,
                'positive_overall_log_loss_interval':overall['log_loss']['ci95'][0]>0,
                'positive_overall_brier_interval':overall['brier']['ci95'][0]>0,
                'minimum500_lead_time_games':lead_games>=500,
                'half_percent_lead_time_log_loss':lead is not None and lead['relative_log_loss_reduction']>=.005,
                'positive_lead_time_log_loss_interval':lead is not None and lead['log_loss']['ci95'][0]>0,
                'nonnegative_lead_time_brier_point':lead is not None and lead['brier']['estimate']>=0}
        gates[name]=checks
        if all(checks.values()):eligible.append(name)
    selected=None;reason='no_candidate_clears_all_fixed_gates'
    if len(eligible)==1:selected=eligible[0];reason='only_qualifying_declared_candidate'
    elif len(eligible)==2:
        extra=comparisons['combat_resources_vs_health_availability']
        if extra['relative_log_loss_reduction']>=.005 and extra['log_loss']['ci95'][0]>0 and extra['brier']['estimate']>=0:
            selected='combat_resources';reason='resolved_additional_gain_over_qualifying_health_model'
        else:selected='health_availability';reason='fixed_preference_for_simpler_qualifying_health_model'
    return {'gates':gates,'eligible':eligible,'selected_candidate':selected,'selection_reason':reason,
            'useful_development_signal':selected is not None,'confirmatory_or_practical_benefit_established':False}
