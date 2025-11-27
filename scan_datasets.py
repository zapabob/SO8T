from huggingface_hub import HfApi
keywords = ['japanese', 'instruction', 'cot', 'public sector', 'government', 'cursor']
api = HfApi()
allowed = {'apache-2.0', 'mit', 'Apache-2.0', 'MIT'}
results = {}
for kw in keywords:
    datasets = api.list_datasets(search=kw, limit=100)
    for ds in datasets:
        if ds.id in results:
            continue
        try:
            info = api.dataset_info(ds.id)
        except Exception:
            continue
        card = info.card_data or {}
        license_name = card.get('license') or info.info.get('license') if hasattr(info, 'info') else None
        if not license_name:
            continue
        if license_name.lower() not in allowed and license_name.upper() not in allowed:
            continue
        results[ds.id] = {
            'license': license_name,
            'description': (card.get('description') or '').strip(),
            'size': info.siblings[0].rfilename if info.siblings else None
        }
        if len(results) >= 12:
            break
    if len(results) >= 12:
        break
for ds_id, meta in results.items():
    print(ds_id, meta['license'])
