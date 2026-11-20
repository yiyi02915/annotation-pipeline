import sys
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'reference_dossier.txt').read_text(encoding='utf8').split('\n\n=== ')
print('\n\n=== '.join(s[int(sys.argv[1]):int(sys.argv[2])]))
