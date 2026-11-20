import unittest
from guarded_expanded_bi import apply_graph_proposals,remove_root,canonicalize_non_event_links,removal_supported
def c(uid,t,text):return dict(candidate_id=uid,start=t,initiator='A',recipient='B',text=text,other_active_run_at_onset=True)
def d(uid,a,l='INTERRUPT',p=''):return dict(candidate_id=uid,action=a,label=l,merge_into=p,reason='')
class GuardedGraph(unittest.TestCase):
 def test_active_feedback_not_removed_as_an_answer(self):
  target=dict(start=2,recipient='B',recipient_clause_at_onset={'end':3})
  ok,_=removal_supported({},target,dict(label='BACKCHANNEL'),dict(finding='NORMAL_RESPONSE'))
  self.assertFalse(ok)
 def test_response_to_unfinished_statement_does_not_establish_free_turn(self):
  case=dict(streams={'B':[dict(text='Complete?',end=1),dict(text='Unfinished.',end=4)]})
  target=dict(start=2,recipient='B')
  ok,_=removal_supported(case,target,dict(label='INTERRUPT'),dict(finding='NORMAL_RESPONSE',evidence_quote='Unfinished.'))
  self.assertFalse(ok)
  ok,_=removal_supported(case,target,dict(label='INTERRUPT'),dict(finding='NORMAL_RESPONSE',evidence_quote='Complete?'))
  self.assertTrue(ok)
 def test_legacy_dropped_parent_links_are_explicit_non_events(self):
  j=dict(decisions=[d('a','DROP',p='stale'),d('b','MERGE',p='a'),d('c','MERGE',p='b')])
  canonicalize_non_event_links(j)
  self.assertEqual([x['action'] for x in j['decisions']],['DROP']*3)
  self.assertTrue(all(not x['merge_into'] for x in j['decisions']))
 def test_existing_feedback_not_erased_by_graph_rewrite(self):
  cs=[c('a',1,'Yeah.')];b=dict(decisions=[d('a','KEEP','BACKCHANNEL')]);p=dict(decisions=[d('a','DROP')])
  result,_=apply_graph_proposals(cs,b,p);self.assertEqual(result['decisions'][0]['action'],'KEEP')
 def test_independent_feedback_bursts_not_collapsed(self):
  cs=[c('a',1,'Yeah.'),c('b',3,'Yeah.')];b=dict(decisions=[d('a','KEEP','BACKCHANNEL'),d('b','KEEP','BACKCHANNEL')]);p=dict(decisions=[d('a','KEEP','BACKCHANNEL'),d('b','MERGE','BACKCHANNEL','a')])
  result,_=apply_graph_proposals(cs,b,p);self.assertEqual(result['decisions'][1]['action'],'KEEP')
 def test_duplicate_interruption_can_merge(self):
  cs=[c('a',1,'Claim.'),c('b',2,'Claim.')];b=dict(decisions=[d('a','KEEP'),d('b','KEEP')]);p=dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a')])
  result,_=apply_graph_proposals(cs,b,p);self.assertEqual(result['decisions'][1]['merge_into'],'a')
 def test_fragmented_root_can_recover_with_continuation(self):
  cs=[c('a',1,'That is like'),c('b',2,'a new idea.')];b=dict(decisions=[d('a','DROP'),d('b','DROP')]);p=dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a')])
  result,_=apply_graph_proposals(cs,b,p);self.assertEqual([d['action'] for d in result['decisions']],['KEEP','MERGE'])
 def test_complete_new_statement_not_added_by_unconstrained_proposal(self):
  cs=[c('a',1,'It is good.'),c('b',2,'More.')];b=dict(decisions=[d('a','DROP'),d('b','DROP')]);p=dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a')])
  result,_=apply_graph_proposals(cs,b,p);self.assertEqual([d['action'] for d in result['decisions']],['DROP','DROP'])
 def test_remove_normal_answer_also_removes_its_continuations_not_other_event(self):
  j=dict(decisions=[d('a','KEEP'),d('b','MERGE',p='a'),d('c','MERGE',p='b'),d('x','KEEP','BACKCHANNEL')]);removed=remove_root(j,'a','normal response')
  self.assertEqual(set(removed),{'a','b','c'});self.assertEqual(j['decisions'][3]['action'],'KEEP')
if __name__=='__main__':unittest.main()
