#!/usr/bin/env python3
"""Build a deterministic source-and-existing-record handoff ZIP from an allow-list."""
import argparse,hashlib,json,tomllib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    cfgpath=ROOT/'reproduction/handoff_v1.toml';cfg=tomllib.loads(cfgpath.read_text())['package'];artifacts=Path(a.artifacts_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);excluded=set(cfg['excluded_source_files']);files={}
    for name in cfg['source_top_files']:
        path=ROOT/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing or unsafe source: '+name)
        files[name]=path.read_bytes()
    for directory in cfg['source_directories']:
        for path in sorted((ROOT/directory).rglob('*')):
            name=path.relative_to(ROOT).as_posix()
            if name in excluded or '__pycache__' in path.parts or path.suffix not in cfg['allowed_suffixes']:continue
            if path.is_symlink():raise ValueError('Symlink in source snapshot: '+name)
            if path.is_file():files[name]=path.read_bytes()
    candidate_manifest=json.loads((artifacts/'outputs/final_candidates_v1/manifest.json').read_text())
    for name in cfg['generated_files']:
        path=artifacts/name
        if path.is_symlink() or not path.is_file():raise ValueError('Missing generated input: '+name)
        value=path.read_bytes();digest=hashlib.sha256(value).hexdigest()
        if name=='outputs/m4_target_repeats/rankings_all.csv':
            if digest!=candidate_manifest['input_sha256']:raise ValueError('Ranking input changed')
        elif path.name in candidate_manifest['file_sha256']:
            if digest!=candidate_manifest['file_sha256'][path.name]:raise ValueError('Candidate artifact changed: '+name)
        files[name]=value
    archived=json.loads(files['reports/final_candidates_v1_2026-11-17.json'])
    if candidate_manifest!=archived:raise ValueError('Candidate manifest differs from archived report')
    manifest={'package_name':cfg['name'],'base_repository_commit':cfg['base_repository_commit'],'scope':cfg['scope'],'raw_course_data_included':False,'trained_model_checkpoints_included':False,'files':{name:{'bytes':len(value),'sha256':hashlib.sha256(value).hexdigest()} for name,value in sorted(files.items())}}
    files['PACKAGE_MANIFEST.json']=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode('utf-8');dest=out/(cfg['name']+'.zip')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in sorted(files.items()):
            info=zipfile.ZipInfo(cfg['name']+'/'+name,date_time=(2026,11,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16;archive.writestr(info,value,compresslevel=9)
    with zipfile.ZipFile(dest) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP integrity failed')
        expected={cfg['name']+'/'+name for name in files}
        if len(archive.namelist())!=len(expected) or set(archive.namelist())!=expected:raise ValueError('ZIP member list differs')
        for name,value in files.items():
            if archive.read(cfg['name']+'/'+name)!=value:raise ValueError('ZIP source mismatch: '+name)
    result={'package':dest.name,'zip_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'zip_bytes':dest.stat().st_size,'entry_count':len(files),'manifest_exact':True,'zip_testzip_passed':True,'all_members_match_source':True,'raw_course_data_included':False,'trained_model_checkpoints_included':False}
    (out/'build_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
