import unittest
from pipeline_core import reconcile_boundary_evidence

class BoundaryEvidence(unittest.TestCase):
    def test_interruption_must_explain_this_boundary(self):
        records=[{'utterances':[{'utterance_id':'left','start':1},{'utterance_id':'right','start':5}]}]
        def boundary():return dict(candidate_id='b',left_id='left',right_id='right',decision='KEEP_BOUNDARY',model_result={'code':'E'})
        def event(floor,start):return dict(event_id='e',floor_id=floor,start=start,status='accepted',interaction_type='INTERRUPT',floor_response='YIELD')
        for invalid in [[],[event('right',6)],[event('left',6)]]:
            b=boundary();issues=reconcile_boundary_evidence(records,[b],invalid)
            self.assertEqual(b['decision'],'PENDING_AUTOMATIC');self.assertEqual(len(issues),1)
        b=boundary();issues=reconcile_boundary_evidence(records,[b],[event('left',3)])
        self.assertEqual(b['decision'],'KEEP_BOUNDARY');self.assertEqual(issues,[])
        self.assertEqual(b['supporting_interaction_ids'],['e'])

if __name__=='__main__':unittest.main()
