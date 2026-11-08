import sys, subprocess, os

def main():
    print('=' * 75)
    print('  AI-Powered Cloud Intrusion Detection & Threat Intelligence Platform')
    print('  Integral University, Lucknow | Major Project 2026-2027')
    print('  Zero-Cost Phase 1 Environment Setup & Verification')
    print('=' * 75)
    print('[*] Python Executable:', sys.executable)
    print('[*] Python Version:', sys.version.split()[0])
    
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        print('\n[*] Checking and installing dependencies from requirements.txt...')
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_file], check=True)
            print('[OK] Requirements installed successfully.')
        except Exception as e:
            print('[!] Notice during installation:', e)
            
    print('\n[*] Verifying core libraries:')
    for mod in ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']:
        try:
            __import__(mod)
            print('    [+] ' + mod + ': Ready')
        except ImportError:
            print('    [-] ' + mod + ': Missing')

    print('\n' + '=' * 75)
    print('Phase 1 Environment Ready! Run python run_phase1_pipeline.py next.')
    print('=' * 75)

if __name__ == '__main__':
    main()
