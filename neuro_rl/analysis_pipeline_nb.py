# %%
import sys
from .analysis_pipeline_cycle_avg import run_analysis

%matplotlib widget

sys.argv = ['analysis_pipeline_nb.py', 'cfg/analyze/analysis_a1_7speed.yaml']

# %%

run_analysis()

# %%

print('done')
