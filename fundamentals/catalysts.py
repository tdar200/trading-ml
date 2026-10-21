"""Evidence-bound document annotations. Never an order or trading score."""
from __future__ import annotations
from bisect import bisect_right
import re
from fundamentals.store import utc,now

TAGS={'guidance_up','guidance_down','margin_warning','demand_strength','demand_weakness','inventory_issue','debt_concern'}
ANNOTATION_KEYS=('tag','start','end','quote','model_version')
CHANNELS=('official_filing','official_news','financial_news','social')
OFFICIAL_CHANNELS=('official_filing','official_news')


def ingest_document_annotations(store,*,symbol,document,source,published_at,observed_at,annotations,channel='official_news',vintage_verified=False):
    if channel not in CHANNELS:raise ValueError('unknown news channel')
    if not isinstance(document,str):raise ValueError('document must be text')
    if not isinstance(annotations,(list,tuple)):raise ValueError('annotations must be a list')
    published=utc(published_at);observed=utc(observed_at)
    # Without an immutable historical vintage, an old publish date does not
    # backdate the text currently fetched from a mutable web page.
    available=published if vintage_verified else max(published,observed)
    # The fetched document is evidence in its own right: it is archived (append-only) even if an annotation is then rejected.
    digest=store.archive(source,document,observed)
    clean=[]
    for annotation in annotations:
        if not isinstance(annotation,dict):raise ValueError('annotation must be a mapping')
        tag=annotation.get('tag')
        if not isinstance(tag,str) or tag not in TAGS:raise ValueError('unsupported structured annotation')
        start,end=annotation.get('start'),annotation.get('end')
        if type(start) is not int or type(end) is not int or not 0<=start<end<=len(document):raise ValueError('invalid evidence span')
        if document[start:end]!=annotation.get('quote'):raise ValueError('annotation not grounded in source text')
        if not annotation.get('model_version'):raise ValueError('extractor version required')
        # Allowlist: caller-supplied extras (execution_authority, action, score...) are never persisted as evidence.
        clean.append({key:annotation[key] for key in ANNOTATION_KEYS})
    return store.add_event(symbol=symbol,event_type='document_annotation',event_at=published,reported_at=published,
        available_at=available,source=source,revision=digest,payload={'document_sha256':digest,'channel':channel,
            'annotations':clean,'historical_vintage_verified':vintage_verified,'event_key':source,'score':None,
            'confidence':None,'execution_authority':False})


EXTRACTOR_VERSION='literal-company-disclosure-v2'
# Deliberately narrow literal claims, not general sentiment or inferred guidance.
_CLAIMS={
 'guidance_up':r'(?:raised|raises|increased) (?:its |our )?(?:full-year |annual )?(?:earnings |revenue )?guidance\b',
 'guidance_down':r'(?:lowered|lowers|reduced|cuts) (?:its |our )?(?:full-year |annual )?(?:earnings |revenue )?guidance\b',
 'margin_warning':r'(?:expects|expect|reported) (?:lower|declining) (?:gross |operating )?margins\b',
 'demand_strength':r'(?:reported|reports|observed) (?:strong|growing|increasing) demand\b',
 'demand_weakness':r'(?:reported|reports|observed) (?:weak|declining|falling) demand\b',
 'inventory_issue':r'(?:reported|reports|disclosed) excess inventory\b',
 'debt_concern':r'(?:breached|violated) (?:a |its |our )?debt covenant\b',
}


def _claim_regex(subject):
    return re.compile(r'\b(?:'+'|'.join(f'(?P<{tag}>{subject} {body})' for tag,body in _CLAIMS.items())+')',re.I)


# First-person "we" is the company only in the company's own channels; in news/analyst/social text it is the author.
_CLAIM_REGEX={True:_claim_regex('(?:we|the company)'),False:_claim_regex('the company')}

# Any of these in the claim's sentence makes the extractor abstain. Single words are matched as whole tokens
# (linear time on a 2 MB sentence); phrases by substring on whitespace-normalised lower case.
_NEGATION=frozenset('not never no nor neither none cannot without untrue false falsely incorrect incorrectly inaccurate inaccurately wrong wrongly '
    'baseless unfounded unsubstantiated unconfirmed unverified deny denies denied denial dispute disputes disputed refute refutes refuted '
    'rumor rumors rumour rumours'.split())
_HYPOTHETICAL=frozenset('if might may could would should suppose supposing imagine consider scenario hypothetical hypothetically whether '
    'allegedly alleged allege alleges claim claims claimed assert asserts asserted purported purportedly supposed supposedly presumably '
    'unclear potential potentially'.split())
_HYPOTHETICAL_PREFIXES=('speculat','assum')
_INSTRUCTION_LIKE=frozenset('ignore instruction instructions prompt prompts pretend example examples'.split())
_THIRD_PARTY=frozenset('competitor competitors rival rivals peer peers'.split())
_PRIOR_PERIOD_WORDS=frozenset('ago previously previous historically formerly'.split())
_PRIOR_PERIOD_PHRASES=('last quarter','last year','last month','last week','last fiscal','last period','earlier this','earlier in','prior year',
    'prior-year','prior quarter','prior-quarter','prior period','prior-period','year-ago','has since','have since','since then','in the past',
    'back in','according to')
_YEAR_ANCHOR=re.compile(r'\b(?:in|during|of|since|from|throughout) (?:the )?(?:(?:fiscal|calendar|financial) )?(?:year )?((?:19|20)\d\d)\b',re.I)
_TOKEN=re.compile(r"[a-z]+(?:['’][a-z]+)?")
_QUOTE_MARKS=frozenset('"“”„‟‘«»‹›')
_OPENING_SINGLE_QUOTE=re.compile(r"(?:^|[\s(\[])'(?=\S)")
_TERMINATOR=re.compile(r'[.!?\n]')
_PRECEDING_WORD=re.compile(r'([A-Za-z](?:\.[A-Za-z])*|[A-Za-z]+)$')
_ABBREVIATIONS=frozenset('inc corp co ltd llc plc lp mr mrs ms dr prof sr jr st vs etc no nos approx dept est e.g i.e u.s u.k u.s.a fig '
    'jan feb mar apr jun jul aug sep sept oct nov dec'.split())


def _sentence_spans(text):
    """Sentence boundaries that survive 'Inc.', 'U.S.', '2.5%' and 'example.com' (a bare split on '.' detached negations)."""
    spans=[];start=0;n=len(text)
    for match in _TERMINATOR.finditer(text):
        i=match.start()
        if text[i]=='.':
            following=text[i+1] if i+1<n else ''
            if following and not following.isspace() and following not in '"\'”’)]':continue   # decimal, domain, internal dot
            word=_PRECEDING_WORD.search(text,max(start,i-16),i)
            if word and (word.group(1).lower() in _ABBREVIATIONS or len(word.group(1))==1):
                after=text[i+1:i+4].lstrip()[:1]
                if after=='' or after.islower() or after.isdigit() or after in ',;:)':continue          # abbreviation mid-sentence
        spans.append((start,i));start=i+1
    spans.append((start,n))
    return spans


def _abstains(sentence,reference_year):
    """True when the sentence negates, disputes, hypothesises, quotes, attributes to a third party or dates the claim to a prior period."""
    if any(c in _QUOTE_MARKS for c in sentence) or _OPENING_SINGLE_QUOTE.search(sentence):return True
    low=' '.join(sentence.lower().split())
    words=_TOKEN.findall(low)
    for w in words:
        if (w in _NEGATION or w in _HYPOTHETICAL or w in _INSTRUCTION_LIKE or w in _THIRD_PARTY or w in _PRIOR_PERIOD_WORDS
                or w.endswith(("n't","n’t")) or w.startswith(_HYPOTHETICAL_PREFIXES)):return True
    if any(p in low for p in _PRIOR_PERIOD_PHRASES):return True
    for anchor in _YEAR_ANCHOR.finditer(sentence):
        # A year anchor is a prior-period reference unless it is the publication year itself.
        if reference_year is None or int(anchor.group(1))!=reference_year:return True
    return False


def extract_disclosures(document,*,channel='official_news',reference_year=None):
    """High-precision literal supported phrases; everything else abstains.

    Negated, denied, disputed, hypothetical, speculative, quoted, third-party and prior-period sentences abstain,
    as does first-person "we" outside the company's own channels. Extracted text remains untrusted data with
    zero execution authority. Recall is deliberately low (paraphrases, passives and named subjects are missed).
    """
    if not isinstance(document,str) or len(document)>2000000:raise ValueError('invalid document')
    if channel not in CHANNELS:raise ValueError('unknown news channel')
    spans=_sentence_spans(document);starts=[s for s,_ in spans];verdict={};annotations=[]
    for match in _CLAIM_REGEX[channel in OFFICIAL_CHANNELS].finditer(document):
        index=bisect_right(starts,match.start())-1
        if index not in verdict:
            s,e=spans[index];verdict[index]=not _abstains(document[s:e],reference_year)
        if verdict[index]:
            start,end=match.start(),match.end()
            annotations.append({'tag':match.lastgroup,'start':start,'end':end,'quote':document[start:end],'model_version':EXTRACTOR_VERSION})
    return sorted(annotations,key=lambda a:(a['start'],a['tag']))


def ingest_disclosures(store,**kwargs):
    """Extraction and evidence storage reuse the strict publication-time gate."""
    channel=kwargs.get('channel','official_news')
    year=int(utc(kwargs['published_at'])[:4])
    return ingest_document_annotations(store,annotations=extract_disclosures(kwargs['document'],channel=channel,reference_year=year),**kwargs)
