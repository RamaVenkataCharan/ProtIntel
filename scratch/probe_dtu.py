"""Probe NetSurfP-2.0 download page and extract file links."""
import urllib.request
import re
import sys

url = 'https://services.healthtech.dtu.dk/services/NetSurfP-2.0/5-Dataset.php'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, timeout=20).read().decode('utf-8', errors='ignore')
except Exception as e:
    print(f"FAILED: {e}", file=sys.stderr)
    sys.exit(1)

# Find all href links containing file extensions
links = re.findall(r'href=["\']([^"\']*\.(?:npz|npy|zip|tar\.gz|gz))["\'>]', html, re.I)
print('Found file links:')
for l in links:
    print(' ', l)

if not links:
    all_hrefs = re.findall(r'href=["\']([^"\']+)["\']', html)
    dl = [h for h in all_hrefs if any(k in h.lower() for k in ['download', 'dataset', 'data', 'netsurfp'])]
    print('Download-ish links:')
    for h in dl[:30]:
        print(' ', h)

    # Also dump a snippet of raw html for inspection
    print('\n--- HTML snippet (first 3000 chars) ---')
    print(html[:3000])
