import unittest
from expanded_review_bi import check_graph
def c(i,t,s='A',r='B'):return dict(candidate_id=i,start=t,initiator=s,recipient=r)
def d(i,a,l='INTERRUPT',p=''):return dict(candidate_id=i,action=a,label=l,merge_into=p,reason='')
class EventGraph(unittest.TestCase):
 def test_nonoverlapping_continuation_can_extend_event(self):
  check_graph([c('a',1),c('b',3)],dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a')]))
 def test_cannot_silently_merge_into_dropped_parent(self):
  with self.assertRaisesRegex(ValueError,'dropped_or_wrong_function'):check_graph([c('a',1),c('b',2)],dict(decisions=[d('a','DROP'),d('b','MERGE',p='a')]))
 def test_no_cross_speaker_merge(self):
  with self.assertRaisesRegex(ValueError,'cross_direction'):check_graph([c('a',1),c('b',2,'B','A')],dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a')]))
 def test_feedback_is_not_child_of_interruption(self):
  with self.assertRaisesRegex(ValueError,'wrong_function'):check_graph([c('a',1),c('b',2)],dict(decisions=[d('a','KEEP'),d('b','MERGE','BACKCHANNEL','a')]))
 def test_cycle_rejected(self):
  with self.assertRaisesRegex(ValueError,'earlier'):check_graph([c('a',1),c('b',2)],dict(decisions=[d('a','MERGE',p='b'),d('b','MERGE',p='a')]))
 def test_duplicate_id_rejected(self):
  with self.assertRaisesRegex(ValueError,'coverage'):check_graph([c('a',1),c('b',2)],dict(decisions=[d('a','KEEP'),d('a','KEEP')]))
 def test_reverse_interruption_and_nested_feedback_allowed(self):
  check_graph([c('a',1),c('b',2,'B','A'),c('c',3)],dict(decisions=[d('a','KEEP'),d('b','KEEP'),d('c','KEEP','BACKCHANNEL')]))
if __name__=='__main__':unittest.main()
