import unittest
from pipeline_core import assemble_units,validate_export,control_timeline,make_events,validate_event,causal_text_context,make_boundaries,boundary_rule

def u(uid,speaker,start,end,text='hello'):
    return dict(utterance_id=uid,speaker=speaker,start=start,end=end,text=text,words=[])
def b(left,right,decision='MERGE'):
    return dict(candidate_id=left+'::'+right,left_id=left,right_id=right,decision=decision)
def event(incoming,floor,kind='BACKCHANNEL',reaction='CONTINUE'):
    return dict(event_id='e1',incoming_id=incoming,floor_id=floor,status='accepted',interaction_type=kind,floor_response=reaction,floor_completion='INCOMPLETE',start=1.5,end=2.2)

class PipelineTests(unittest.TestCase):
    def test_multitoken_alignment_anomaly_blocks_training_source(self):
        bad=u('a','A',0,9,'got it');bad['words']=[dict(word='got',start=0,end=8),dict(word='it',start=8.5,end=9)]
        r=dict(conversation_id='c',utterances=[bad,u('b','B',8,8.3),u('c','A',9.5,10)])
        candidate=make_boundaries(r)[0]
        self.assertEqual(boundary_rule(candidate),('PENDING_AUTOMATIC','alignment_quality'))
        self.assertTrue(make_events(r)[0]['quality_flags'])
        units,_,_=assemble_units(r,[],[])
        self.assertTrue(units[0]['source_quality_flags'])
        self.assertTrue(control_timeline(r)[0][0]['input_quality_flags'])
    def test_missing_or_pending_decision_never_merges(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,1),u('b','A',1.2,2)])
        for decisions in [[],[b('a','b','PENDING_AUTOMATIC')]]:
            units,mapping,overrides=assemble_units(r,decisions,[])
            self.assertEqual(len(units),2);self.assertEqual(validate_export(r,units,[],mapping),[])
    def test_main_merge_preserves_interleaved_backchannel(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,2),u('bc','B',1.5,1.8),u('b','A',2.1,3)])
        units,mapping,_=assemble_units(r,[b('a','b')],[event('bc','a')])
        self.assertEqual(mapping['a'],mapping['b']);self.assertNotEqual(mapping['bc'],mapping['a'])
        self.assertEqual(len(units),2);self.assertEqual(validate_export(r,units,[event('bc','a')],mapping),[])
    def test_role_change_prevents_absorbing_feedback_into_main(self):
        r=dict(conversation_id='c',utterances=[u('floor','A',0,2),u('bc','B',1,1.2),u('main','B',2.2,4)])
        units,mapping,changes=assemble_units(r,[b('bc','main')],[event('bc','floor')])
        self.assertNotEqual(mapping['bc'],mapping['main']);self.assertEqual(changes[0]['reason'],'backchannel_role_changed')
    def test_interruption_attempt_vs_yield(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,2),u('i','B',1.5,2.2),u('b','A',2.4,3)])
        for reaction,expected in [('CONTINUE',True),('YIELD',False)]:
            units,mapping,_=assemble_units(r,[b('a','b')],[event('i','a','INTERRUPT',reaction)])
            self.assertEqual(mapping['a']==mapping['b'],expected)
    def test_gap_feedback_candidate_is_not_omitted(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,1),u('bc','B',1.2,1.5)])
        cs=make_events(r);self.assertEqual(len(cs),1);self.assertEqual(cs[0]['candidate_kind'],'gap')
    def test_activity_uses_source_spans_not_merged_envelope(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,1),u('b','A',2,3)])
        intervals,transitions=control_timeline(r)
        self.assertEqual(intervals[1]['acoustic_state'],'SILENCE')
        self.assertEqual(transitions[-1]['action'],'RECORDING_END')
        self.assertTrue(all(not x['policy_target'] for x in transitions))
    def test_uptake_cannot_cite_prior_response(self):
        r=dict(conversation_id='c',utterances=[u('a','A',0,2),u('bc','B',1,1.5)])
        c=make_events(r)[0]
        raw=dict(interaction_type='BACKCHANNEL',function='ACKNOWLEDGE',floor_completion='INCOMPLETE',floor_response='CONTINUE',content_uptake='ADDRESSED',response_context_index=0,floor_quote='hello',incoming_quote='hello',reason='The speaker visibly addresses this listener feedback.')
        self.assertEqual(validate_event(c,raw),'invalid_response_chronology')
    def test_causal_prefix_excludes_future_words(self):
        first=u('a','A',0,3,'past future');first['words']=[dict(word='past',start=0,end=.8),dict(word='future',start=2,end=3)]
        result=causal_text_context(dict(utterances=[first]),1)
        self.assertEqual(result[0]['text'],'past');self.assertLessEqual(result[0]['end'],1)

if __name__=='__main__':unittest.main()
