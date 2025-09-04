
# Data Access & Mapping

Primary dataset: OpenEnergyHub / ORNL EV charging sessions (3,395 sessions).  
Save as `data/openenergyhub_ev_charging.csv` and adjust `config.yaml` column names if needed.

Expected columns: start_time, end_time, energy_kwh, session_cost, site_id.

Labels: add `label` (0/1) if you have ground truth; otherwise use `scripts/inject_attacks.py` which logs all injections.
