"""PACE (Precision-Aware Conditional Execution) extensions to π0.5.

See docs/pace_v5_proposal.tex for the current design (PSW + CPR).

Legacy v3/v4 flags remain for old configs: use_cas, use_da_fta, use_pps.
PACE v5 is opt-in via Pi0Config(use_psw=True, use_cpr=True) with
pps_weight_path pointing at universal PPS weights.
"""
