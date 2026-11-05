"""Author-derived exact molecular voting; no property or success access."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from contracts import read,write,verify
from runtime_common import require_allocation
from p10_vote import classify_ev,prefix_vote,native_projection


def main():
    import argparse,os,time
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--input-sha',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    require_allocation();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';verify(a.input,a.input_sha)
    data=read(a.input);out=[];t=time.perf_counter()
    for item in data:
        rows=item['proposals'];assert len(rows)==400
        for row in rows:row.update(classify_ev(row['parsed_candidate'],row['raw_response']))
        winner=prefix_vote(rows,[400])[0]
        out.append(dict(record=item['record'],classified_proposals=rows,winner=winner,native=native_projection(winner,item['record'])))
    write(a.output,dict(inputs=out,identity_and_vote_seconds=time.perf_counter()-t,property_access=False))


if __name__=='__main__':main()
