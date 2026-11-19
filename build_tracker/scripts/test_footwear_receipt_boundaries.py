"""Tied evidence and split source batches cannot escape footwear recovery gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from replay_footwear_windows import split_receipt_batches,receipt_exclusion

def event(kind,pid,item=0):
    return {'type':kind,'participantId':pid,'itemId':item,'timestamp':720000}

receipt={(1,720000)}
timeline=lambda events:{'info':{'frames':[{'events':events}]}}
destroy=event('ITEM_DESTROYED',1,2422)
buy=event('ITEM_PURCHASED',1,3020)
assert not split_receipt_batches(timeline([destroy,buy]),receipt)
assert split_receipt_batches(timeline([destroy,event('ITEM_PURCHASED',2,1036),buy]),receipt)==[(1,720000)]
assert split_receipt_batches(timeline([destroy,event('WARD_PLACED',1),buy]),receipt)==[(1,720000)]
assert not split_receipt_batches(timeline([event('ITEM_SOLD',1,2422)]),receipt)
assert receipt_exclusion({'participant_id':2,'ts':720000,'ts_end':720001},receipt)=='decision_tied_to_receipt'
assert receipt_exclusion({'participant_id':1,'ts':719000,'ts_end':721000},receipt)=='visit_spans_receipt'
assert receipt_exclusion({'participant_id':2,'ts':720001,'ts_end':720002},receipt) is None
assert receipt_exclusion({'participant_id':2,'ts':735000,'ts_end':735000,'is_save':True},receipt)=='decision_tied_to_receipt'
print('ok footwear receipt boundaries: all-board ties, death anchors, receipt-spanning owner visits, contiguous accounting and split batch refusal')
