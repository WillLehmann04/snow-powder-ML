import re
import requests

h = {"User-Agent": "powder-ml-research/1.0"}

# ---- 1. Check which URL patterns return 200 ----
print("URL scan:")
resorts = {
    "mt_hutt": [
        "https://www.mthutt.co.nz/mountain-information/snow-report/",
        "https://www.mthutt.co.nz/snow-conditions/",
        "https://www.mthutt.co.nz/conditions/",
        "https://www.mthutt.co.nz/snow-report/",
    ],
    "cardrona": [
        "https://cardrona-treblecone.com/snow-report/",
        "https://cardrona-treblecone.com/mountain-conditions/",
    ],
}
for resort, urls in resorts.items():
    print(f"  {resort}:")
    for url in urls:
        try:
            r = requests.get(url, headers=h, timeout=10)
            is_xml = "<snowReport" in r.text
            print(f"    {r.status_code} xml={is_xml}  {r.url}")
        except Exception as e:
            print(f"    ERR  {type(e).__name__}  {url}")

# ---- 2. Inspect the Cardrona page for API endpoints ----
print("\nCardrona page inspection:")
try:
    r = requests.get("https://cardrona-treblecone.com/snow-report", headers=h, timeout=10)
    # Look for any URLs containing snow/api/conditions keywords
    api_hits = re.findall(r'["\']https?://[^"\' ]+(?:api|snow|condition|report)[^"\' ]*["\']', r.text)
    print(f"  Page size: {len(r.text)} chars")
    print(f"  Potential API URLs found: {len(api_hits)}")
    for hit in api_hits[:10]:
        print(f"    {hit}")
    if not api_hits:
        # Check if there's any JSON data at all
        json_hits = re.findall(r'\{[^{}]{20,200}snow[^{}]{0,100}\}', r.text, re.IGNORECASE)
        print(f"  JSON snow data fragments: {len(json_hits)}")
        for jh in json_hits[:3]:
            print(f"    {jh[:120]}")
except Exception as e:
    print(f"  ERR: {e}")
