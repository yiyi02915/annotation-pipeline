"""Retrieve complete transcript blocks; never infer a final floor label here.

A block is a whole punctuated utterance, joined to consecutive utterances by the
same speaker in onset order. It is a retrieval unit, not a gold dialogue turn.
Three exchanges are operationalized as six blocks on EACH side. Short feedback
is preserved and does not consume that six-block budget. No second-based crop.
"""
import copy
import math

VERSION = 'complete_blocks_v1'
END = ('.', '?', '!', '。', '？', '！')


class DialogueContext:
    def __init__(self, streams):
        self.streams = streams
        utterances = []
        for speaker, words in streams.items():
            current = []
            for word in sorted(words, key=lambda w: (w['start'], w['end'])):
                current.append(word)
                if word['text'].rstrip().endswith(END):
                    utterances.append(self.block(speaker, current)); current = []
            if current:
                utterances.append(self.block(speaker, current))
        self.blocks = []
        for block in sorted(utterances, key=lambda b: (b['start'], b['speaker'])):
            if self.blocks and self.blocks[-1]['speaker'] == block['speaker']:
                old = self.blocks[-1]
                old['words'] += block['words']
                self.blocks[-1] = self.block(block['speaker'], old['words'])
            else:
                self.blocks.append(block)
        for index, block in enumerate(self.blocks):
            block['block_id'] = f'B{index:05d}'
        self.by_word = {w['id']: i for i,b in enumerate(self.blocks) for w in b['words']}

    @staticmethod
    def block(speaker, words):
        return dict(speaker=speaker, start=min(w['start'] for w in words),
                    end=max(w['end'] for w in words), text=' '.join(w['text'] for w in words), words=words)

    @staticmethod
    def public(block):
        result = {k:v for k,v in block.items() if k != 'words'}
        result['word_timing_columns'] = ['text','start','end']
        result['word_timing'] = [[w['text'],round(w['start'],3),round(w['end'],3)] for w in block['words']]
        return result

    def surrounding(self, start, end, exchanges=3):
        hits = [i for i,b in enumerate(self.blocks) if b['start'] <= end and b['end'] >= start]
        if not hits:
            raise ValueError('context_target_not_in_full_transcript')
        lo, hi = min(hits), max(hits)
        for direction in (-1, 1):
            cursor = lo-1 if direction < 0 else hi+1
            count = 0
            while 0 <= cursor < len(self.blocks) and count < 2*exchanges:
                # This length affects retrieval only; it is NOT a feedback label.
                count += len(self.blocks[cursor]['words']) > 3
                if direction < 0: lo = cursor
                else: hi = cursor
                cursor += direction
        return self.blocks[lo:hi+1]

    def bi(self, case, exchanges=3):
        result = copy.deepcopy(case)
        blocks = self.surrounding(*case['interval'], exchanges=exchanges)
        result['dialogue_context'] = [self.public(b) for b in blocks]
        result['context_policy'] = dict(version=VERSION, exchanges_each_side=exchanges,
            unit='complete_transcript_blocks_not_gold_turn_labels', blocks=len(blocks),
            interval=[min(b['start'] for b in blocks),max(b['end'] for b in blocks)],
            target_interval=case['target'], timestamps='original_recording_absolute_seconds')
        # Candidate streams/IDs/core stay exactly as extracted by the stable version.
        return result

    def gap(self, payload, history_char_budget=12000):
        result = copy.deepcopy(payload)
        target = payload['target']; lo, hi = target['absolute_gap']
        left_i = self.by_word[target['left_word_id']]
        right_i = self.by_word[target['right_word_id']]
        # Keep the whole target blocks and three exchanges after the boundary.
        local = self.surrounding(lo, hi)
        first = min(self.blocks.index(b) for b in local)
        last = max(self.blocks.index(b) for b in local)
        selected = self.blocks[:last+1]
        mode = 'all_prior_text_plus_complete_following_blocks'
        if sum(len(b['text']) for b in selected) > history_char_budget:
            selected = self.blocks[first:last+1]
            mode = 'previous_3_exchanges_plus_complete_following_blocks'
        if left_i not in [self.blocks.index(b) for b in selected] or right_i not in [self.blocks.index(b) for b in selected]:
            raise ValueError('missing_target_block')
        result['context'] = [self.public(b) for b in selected]
        focus_words = [w for i in sorted({left_i,right_i}) for w in self.blocks[i]['words']]
        result['focus_before_target'] = ' '.join(w['text'] for w in focus_words if w['end'] <= lo+1e-6)
        result['focus_after_target'] = ' '.join(w['text'] for w in focus_words if w['start'] >= hi-1e-6)
        right_words = self.blocks[right_i]['words']
        at = next(i for i,w in enumerate(right_words) if w['id']==target['right_word_id'])
        right_unit = []
        for w in right_words[at:]:
            right_unit.append(w)
            if w['text'].rstrip().endswith(END): break
        result['target_right_unit'] = dict(start=hi,end=right_unit[-1]['end'],text=' '.join(w['text'] for w in right_unit))
        # Do not duplicate the entire word timeline in auxiliary speaker views.
        result['other_near_target'] = [{k:v for k,v in b.items() if k not in ('word_timing','word_timing_columns')}
                                       for b in result['context'] if b['speaker']!=payload['focus']]
        result['other_before_right'] = [b for b in result['other_near_target'] if b['start']<hi]
        result['long_gap_middle_omitted'] = False
        occupied=[]
        for words in self.streams.values():
            occupied.extend((max(lo,w['start']),min(hi,w['end'])) for w in words if w['end']>lo and w['start']<hi)
        cursor=lo;quiet=[]
        for a,b in sorted(occupied):
            if a>cursor: quiet.append([cursor,a])
            cursor=max(cursor,b)
        if cursor<hi: quiet.append([cursor,hi])
        result['gap_activity'] = dict(
            same_speaker_gap_seconds=round(hi-lo,6),
            longest_no_aligned_word_seconds=round(max((b-a for a,b in quiet),default=0),6),
            no_aligned_word_seconds=round(sum(b-a for a,b in quiet),6),
            no_aligned_word_intervals=quiet,
            note='FA-based absence of words, not VAD-verified silence; laughter/noise may exist')
        result['context_policy'] = dict(version=VERSION,mode=mode,blocks=len(selected),
            first_block=selected[0]['block_id'],last_block=selected[-1]['block_id'],
            history_truncated=selected[0] is not self.blocks[0],
            timestamps='original_recording_absolute_seconds',retrieval_only=True)
        return result


def apply_long_gap_policy(decision, payload, limit=3.0):
    """User-requested engineering policy, not a claim about human physiology.

    Preserve the semantic answer and policy conflict. No new business label.
    This is a same-speaker word gap; it may contain the partner's speech.
    """
    result = copy.deepcopy(decision)
    if not math.isfinite(limit) or limit < 0:
        raise ValueError('invalid_long_gap_limit')
    if result.get('status') != 'ok': return result
    lo, hi = payload['target']['absolute_gap']
    if not math.isfinite(lo+hi) or lo < 0 or hi <= lo:
        raise ValueError('invalid_gap_timestamps')
    result['semantic_label'] = result['label']
    result['long_gap_semantic_pause'] = round(hi-lo,6)>3 and result['label']=='PAUSE'
    result['long_gap_limit_seconds'] = limit or None
    result['long_gap_policy_applied'] = bool(limit and round(hi-lo,6)>limit)
    result['duration_semantics_conflict'] = result['long_gap_policy_applied'] and result['label']=='PAUSE'
    if result['long_gap_policy_applied']:
        result['label'] = 'TURN_TRANSITION'
        result['label_source'] = 'engineering_duration_cap'
    else:
        result['label_source'] = 'semantic_judge'
    return result
