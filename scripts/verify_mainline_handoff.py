#!/usr/bin/env python3
"""Verify a handoff directory using Python's standard library only."""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

IGNORED_DIRECTORIES={'.venv','__pycache__'}
GENERATED_DIRECTORIES={'outputs/handoff_verification','outputs/candidates_rebuilt','outputs/handoff_rebuilt'}


def ignored(relative):
    return any(p in IGNORED_DIRECTORIES for p in relative.parts) or relative.name=='.DS_Store' or any(relative.as_posix()==p or relative.as_posix().startswith(p+'/') for p in GENERATED_DIRECTORIES)


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text());expected=manifest['files'];seen=set()
    for name,item in expected.items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or ignored(relative):raise ValueError('Unsafe manifest path: '+name)
        path=root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or linked file: '+name)
        value=path.read_bytes()
        if len(value)!=item['bytes'] or hashlib.sha256(value).hexdigest()!=item['sha256']:raise ValueError('Content mismatch: '+name)
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if ignored(relative):continue
        if path.is_symlink():raise ValueError('Unexpected symbolic link: '+str(relative))
        if path.is_file() and relative.as_posix()!='PACKAGE_MANIFEST.json':seen.add(relative.as_posix())
    if seen!=set(expected):raise ValueError('Unexpected files: '+repr(sorted(seen-set(expected))))
    return {'checked_files':len(expected),'manifest_files':len(seen),'all_checksums_match':True,'no_unlisted_files':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
if __name__=='__main__':main()
