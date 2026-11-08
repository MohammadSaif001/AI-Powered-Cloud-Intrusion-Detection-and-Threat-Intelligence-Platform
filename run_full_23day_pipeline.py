"""
run_full_23day_pipeline.py
Complete Days 1-23 Master Pipeline Runner
Runs Phase 1 (Days 1-15) then Phase 1 Extended (Days 16-23).
Zero cost. All references from synopsis papers.
"""
import subprocess, sys, os

BANNER = '''
================================================================================
  AI-POWERED CLOUD INTRUSION DETECTION & THREAT INTELLIGENCE PLATFORM
  COMPLETE 23-DAY PHASE 1 PIPELINE (August 1-25, 2026)
  Integral University, Lucknow | B.Tech CSE 2026-2027
  Team: M. Rahil | M. Saif | M. Alman Khan | Supervisor: Ms. Falak Alam
================================================================================
'''

def run_script(name):
    print(f"\n>>> Running: {name}")
    result = subprocess.run([sys.executable, name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] {name} exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"  [OK] {name} completed successfully")

print(BANNER)
print("\n[PHASE 1 - DAYS 1-15] Running core EDA pipeline...")
run_script('run_phase1_pipeline.py')

print("\n[PHASE 1 - DAYS 16-23] Running extended pipeline...")
run_script('run_phase1_extended_pipeline.py')

print("\n" + "=" * 80)
print("  ALL 23 DAYS COMPLETE - FULL PHASE 1 VERIFIED SUCCESSFULLY")
print("  NSL-KDD EDA, CICIDS2017, UNSW-NB15, CM-Format, Baseline ML,")
print("  SHAP Analysis, Autoencoder Zero-Day Detection all executed.")
print("=" * 80)
