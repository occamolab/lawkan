#!/usr/bin/env python3
"""Build repro/registry.json from canonical.json + run_all.sh + manual metadata.

The registry is the single machine-readable map: block -> script, args, tier,
paper item, reference device/dtype, canonical dump.  Run after adding a block.
"""
import json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = json.load(open(os.path.join(ROOT, 'repro', 'canonical.json')))
SH = open(os.path.join(ROOT, 'repro', 'run_all.sh')).read()

# run_all.sh: run_one <name> "$PY" <script> <args...>
RUN = {}
for m in re.finditer(r'run_one\s+(\S+)\s+"\$PY"\s+(\S+)([^\n]*)', SH):
    RUN[m.group(1)] = (m.group(2), m.group(3).strip().split())

ALIAS = {
    'e16v2': 'e16', 'x7vqf_disturbed': 'x7vqf', 'x7vqf_undisturbed': 'x7vqfu',
    'mk1_fr3': 'mk1fr3', 'mk1_views': 'mk1views', 'kinova': 'mk1',
    'x26a': 'x26b', 'x26a2': 'x26b2', 'x19': 'x19rec', 'e31d_corr': 'e31d', 'e30_corr': 'e30',
}

MANUAL = {
    'x7bench': ('experiments/real/X7_bench_main.py', [], 'cpu', 'float64'),
    'x7gn': ('experiments/real/X7_gn_ls.py', [], 'cpu', 'float64'),
    'x10_p48': ('experiments/real/X10_fk_calib_real.py', [], 'cpu', 'float64'),
    'e30_corr': ('experiments/synthetic/E30_full_so3.py', [], 'cpu', 'float32'),
    'pool': ('experiments/real/POOL2_verdict.py', [], 'cpu', 'float64'),
    'mk1c': ('experiments/real/MK1c_shuffle_control.py', [], 'cpu', 'float64'),
    'rc1_5seed': ('experiments/real/RC1_robotcali.py', ['--seeds', '5'], 'cpu', 'float64'),
    'e20s0': ('experiments/synthetic/E20_higher_harmonic.py', ['--seed_start', '0', '--seeds', '5'], 'cuda', 'float32'),
    'x10b': ('experiments/real/X10b_poe_ls_baseline.py', [], 'cpu', 'float64'),
    'g2bc_fig': ('experiments/real/G2bc_isolated_psi.py', ['--part', 'fr3', '--fr3-split', 'random', '--fr3-psi', 'fourier'], 'cpu', 'float64'),
    'x36e': ('experiments/real/X36_recount_socket.py', ['--probe', 'e'], 'cpu', 'float64'),
    'x36seq': ('experiments/real/X36_recount_socket.py', [], 'cpu', 'float64'),
    'x36a': ('experiments/real/X36_recount_socket.py', ['--probe', 'a'], 'cpu', 'float64'),
    'x36c': ('experiments/real/X36_recount_socket.py', ['--probe', 'c'], 'cpu', 'float64'),
    'x38': ('experiments/real/X38_perjoint_socket.py', [], 'cpu', 'float64'),
    'x38b': ('experiments/real/X38b_kernel_socket.py', [], 'cpu', 'float64'),
    'x38c': ('experiments/real/X38c_screw_heads.py', [], 'cpu', 'float64'),
    'x42': ('experiments/real/X42_optimizer_matched.py', [], 'cpu', 'float64'),
    'x43': ('experiments/real/X43_abb_lm.py', [], 'cpu', 'float64'),
    'x44': ('experiments/real/X44_fk_matched.py', [], 'cpu', 'float64'),
    'x44_ur5': ('experiments/real/X44_fk_matched.py', ['--platform', 'ur5'], 'cpu', 'float64'),
    'x45': ('experiments/real/X45_gyro_matched.py', [], 'cpu', 'float64'),
    'x46': ('experiments/real/X46_convergence.py', ['--epochs','4000','--schedule-epochs','4000','--ckpt-step','10','--a1-only'], 'cpu', 'float64'),
    'x46a': ('experiments/real/X46_convergence.py', ['--epochs','4000','--schedule-epochs','4000','--a1-only'], 'cpu', 'float64'),
    'x46b': ('experiments/real/X46_convergence.py', ['--epochs','8000','--schedule-epochs','8000','--a1-only'], 'cpu', 'float64'),
    'x46b2': ('experiments/real/X46_convergence.py', ['--skip-a1'], 'cpu', 'float64'),
    'x47': ('experiments/real/X47_socket_joint_vs_seq.py', [], 'cpu', 'float64'),
    'x52': ('experiments/real/X52_reg_sweep.py', [], 'cpu', 'float64'),
    'x53sock': ('experiments/real/X53_socket_convergence.py', [], 'cpu', 'float64'),
    'x61': ('experiments/real/X61_all.py', ['--seeds', '1,2,3'], 'cpu', 'float64'),
    'x61b': ('experiments/real/X61_all.py', ['--seeds', '6'], 'cpu', 'float64'),
    'x16b': ('experiments/real/X16b_psi_diagnostics_twoslot.py', [], 'cpu', 'float64'),
    'g2bc_cluster': ('experiments/real/G2bc_isolated_psi.py', ['--part','fr3','--fr3-split','cluster','--sizes_fr3','60,80','--seeds_fr3','5'], 'cpu', 'float64'),
}

HEADLINE = {
    'run_main': 'Table 2; Fig. 2; §4.2.1',
    'e31': 'Table 18; §4.2.2',
    'x7bench': 'Table 3; Fig. 3(a,b)',
    'x7': 'Table 3 (bench protocol)',
    'x13': 'Table 3 (learned law)', 'x14': 'Table 3 (learned law)',
    'g2bc_fig': 'Table 3; Fig. 3(a)', 'x22': 'Table 3 (row 4)',
    'x24': 'Table 3 (gate)', 'x33': 'Table 3 (row 7)', 'x35': 'Table 3 (recount)',
    'x42': 'Table 3 (matched optimizer)', 'x44': 'Table 3/§5',
    'x45': 'Table 3/§5',
    'mk1': 'Table 4', 'mk1_fr3': 'Table 4', 'x36seq': 'Table 4 (seq)',
    'x38': 'Table 4', 'x38b': 'Table 4', 'x38c': 'Table 4',
    'rc1': 'Table 5 (ABB)', 'pool': '§5 pooled',
    'x16iso': 'Fig. 3(c)', 'x16b': 'Fig. 3(c)', 'x46': 'Fig. 4',
    'x5': 'Table 19/Fig. A5', 'x6': 'Table 20', 'x6b': 'Table 20 (competitor)',
    'x7vqf_disturbed': 'Table 19 (VQF)', 'x7vqf_undisturbed': 'Table 19 (VQF)',
    'x10': 'Table 21/Fig. A6', 'mk1views': 'Table 4 (cross-view)',
    # figure-data dependencies
    'e38': 'Fig. 2 (tuned opponents)', 'e39': 'Fig. A3', 'e40': 'Fig. A2',
    'e31b': 'Fig. A2', 'e31c': 'Fig. A2', 'e32': 'Fig. A2',
    's1': 'Fig. 3/A9', 's2': 'Fig. A9', 's3': 'Fig. 3',
    'x32': 'Fig. 3/A7', 'x34': 'Fig. 3', 'x11': 'Fig. A9(c)',
}

GPU = {'run_main','run_ext','e16v2','e19','e19b','e20','e20s0','e21','e22','e23',
       'e24','e25','e25b','e29','e30','e31','e31b','e31c','e31d','e32','e33',
       'e33b','e33c','e36','x5','x6','x8','x8t','x9','x6b'}

_OPT_CACHE = {}


def options_for(script):
    if script in _OPT_CACHE: return _OPT_CACHE[script]
    path = os.path.join(ROOT, script)
    opts = {}
    try:
        text = open(path, encoding='utf-8').read()
        for o in re.findall(r"add_argument\(\s*['\"](--[A-Za-z0-9_-]+)", text):
            opts[o.replace('_', '-').lower()] = o
    except Exception:
        pass
    _OPT_CACHE[script] = opts
    return opts


def script_for(key):
    if key in MANUAL: return MANUAL[key][0], MANUAL[key][1]
    name = ALIAS.get(key, key)
    if name in RUN: return RUN[name]
    # fall back: an uppercase prefix script
    pref = key.upper()
    for d in ('synthetic', 'real'):
        p = os.path.join(ROOT, 'experiments', d)
        for f in sorted(os.listdir(p)):
            if f.upper().startswith(pref + '_') and f.endswith('.py'):
                return 'experiments/%s/%s' % (d, f), []
    return None, []

reg = {}
for key, v in CANON.items():
    if not isinstance(v, dict) or 'file' not in v: continue
    script, args = script_for(key)
    dev, dt = ('cuda', 'float32') if key in GPU else ('cpu', 'float64')
    if key in MANUAL: dev, dt = MANUAL[key][2], MANUAL[key][3]
    f = os.path.join(ROOT, 'repro', 'json', v['file'])
    status = 'ok' if os.path.exists(f) else 'missing_dump'
    try:
        d = json.load(open(f)); m = d.get('_meta', {}) if isinstance(d, dict) else {}
        if isinstance(m, dict) and m.get('device'):
            dev = m['device']; dt = 'float32' if dev == 'cuda' else 'float64'
        # prefer the exact recorded CLI args of the canonical run
        if isinstance(m, dict) and isinstance(m.get('args'), dict):
            auto = []
            for k, val in m['args'].items():
                if val is None or val is False: continue
                if isinstance(val, float) and val != val: continue
                opts = options_for(script) if script else {}
                flag = opts.get('--' + k.replace('_', '-').lower(), '--' + k.replace('_', '-'))
                if val is True: auto.append(flag)
                else: auto += [flag, str(val)]
            if auto: args = auto
    except Exception:
        pass
    # normalise quotes and machine-specific data paths
    fixed = []
    for a in args:
        a = a.strip('"').strip("'")
        a = re.sub(r'\$([A-Z_]+)', r'${\1}', a)
        if a.rstrip('/').endswith('broad/data_hdf5') or a in ('../broad/data_hdf5', './broad/data_hdf5'):
            a = '${BROAD}'
        elif a.endswith('calib_data.pickle'):
            a = '${CALIB}'
        elif a.endswith('RobotCali-new.xlsx'):
            a = '${ABB_XLSX}'
        elif a.startswith('./data/robotcali/') and a.endswith('.xlsx'):
            a = '${ABB_XLSX}'
        elif a.rstrip('/').endswith('/quatkan/data') or a in ('../data', './data', 'data'):
            a = '${LAWKAN_DATA}'
        fixed.append(a)
    args = fixed
    native = dev
    if dev == 'cuda':
        dev, dt = 'cpu', 'float32'   # portable reference; GPU via --mode gpu
    entry = {
        'script': script, 'args': args,
        'tier': 'headline' if key in HEADLINE else 'full',
        'paper': HEADLINE.get(key, ''), 'group': 'revision' if key.startswith('x3') or key.startswith('x4') or key.startswith('x5') or key.startswith('x6') else ('synthetic' if script and '/synthetic/' in script else 'real'),
        'device': dev, 'dtype': dt, 'canonical': v['file'],
        'status': status if script else 'missing_script',
    }
    if native != dev: entry['native_device'] = native
    reg[key] = entry
# not-yet-dumped branch blocks (scripts exist; dumps to be regenerated)
for key, sc, ar in (
    ('x54','X54_pole_check.py',[]),('x55','X55_qfeat_fix.py',[]),('x56','X56_kgrid.py',[]),
    ('x57','X57_cells.py',[]),('x58','X58_swa.py',[]),('x59','X59_ema.py',[]),
    ('x60','X60_geo.py',[]),('x37','',[])):
    if key in reg: continue
    reg[key] = {'script': ('experiments/real/'+sc) if sc else None, 'args': ar,
                'tier': 'full', 'paper': '', 'group': 'revision',
                'device': 'cpu', 'dtype': 'float64', 'canonical': None,
                'status': 'missing_dump' if sc else 'missing_script'}
reg['derived_stats'] = {
    'script': 'repro/stats/paper_stats.py', 'args': [], 'tier': 'full',
    'paper': 'Table 3 p/CI; Table 4 BH; §5.2 T1/T2', 'group': 'stats',
    'device': 'cpu', 'dtype': 'float64', 'canonical': None, 'status': 'ok'}
json.dump(reg, open(os.path.join(ROOT, 'repro', 'registry.json'), 'w'), indent=1, sort_keys=True)
print('registry entries:', len(reg))
from collections import Counter
print(Counter(v['tier'] for v in reg.values()))
print(Counter(v['status'] for v in reg.values()))
