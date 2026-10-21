"""Read-only heuristic secret scan. Reports locations/categories, NEVER values.

Not a certification: unknown formats, removed refs and external logs are out of
scope. Scan every reachable historical blob without checking out old versions.
"""
import json,re,subprocess
from pathlib import Path

PATTERNS={
    'aws_access_id':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_token':re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'literal_secret_assignment':re.compile(rb'''(?i)(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token)\s*[=:]\s*["']([A-Za-z0-9_+/=-]{20,})["']'''),
}

def scan():
    objects=subprocess.check_output(['git','rev-list','--objects','--all']).decode().splitlines()
    findings=[]; scanned=0; skipped=0
    for row in objects:
        oid,_,name=row.partition(' ')
        if subprocess.check_output(['git','cat-file','-t',oid]).strip()!=b'blob': continue
        size=int(subprocess.check_output(['git','cat-file','-s',oid]))
        if size>2_000_000: skipped+=1; continue
        data=subprocess.check_output(['git','cat-file','blob',oid]); scanned+=1
        for kind,pattern in PATTERNS.items():
            if pattern.search(data): findings.append({'blob':oid,'path':name,'category':kind})
    return {'scope':'all reachable git blobs <=2MB; heuristic patterns; values redacted',
            'blobs_scanned':scanned,'large_blobs_skipped':skipped,'findings':findings}

if __name__=='__main__':
    report=scan(); p=Path('results/small_account/secret_audit.json'); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
