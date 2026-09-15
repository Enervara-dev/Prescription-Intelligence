"""
data/indian_brands.py
------------------------
Hand-curated Indian brand -> generic mappings, with strength/dosage-form,
supplementing the flat `medicine_list.txt` names with the structured detail
(brand+generic+strength+form+combination-components) the resolver needs.

PROVENANCE: every entry below is well-established, publicly-known
pharmacology (the same kind of fact as "Tylenol = acetaminophen" in the US —
printed on the product's own packaging/package insert). Nothing here was
scraped from a third-party catalog or database; this is manual curation
covering representative categories (analgesics/antipyretics, antibiotics,
antihistamines, antacids/PPIs, antihypertensives, antidiabetics, pediatric
syrups, and combination products), not an exhaustive brand list. All rows are
tagged source="legacy_manual", source_version="hand_curated_indian_brands_v1"
so their provenance is explicit and queryable.

Each entry becomes one `medicines` row via scripts/seed_indian_brands.py
(idempotent upsert, same as the rest of the legacy catalog).
"""

INDIAN_BRANDS: list[dict] = [
    # -- Analgesics / Antipyretics --
    {
        "brand_name": "Dolo 650", "generic_name": "Paracetamol",
        "aliases": ["dolo", "dolo650", "dolo 650", "dolo-650"],
        "strength": "650mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Crocin 650", "generic_name": "Paracetamol",
        "aliases": ["crocin", "crocin650", "crocin 650", "crocin-650", "crocin advance"],
        "strength": "650mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Calpol 500", "generic_name": "Paracetamol",
        "aliases": ["calpol", "calpol500", "calpol 500", "calpol-500"],
        "strength": "500mg", "dosage_form": "tablet",
    },
    {
        # Common pediatric syrup strength/form -- distinct SKU from the
        # tablet above; dosage-form-aware ranking must not conflate them.
        "brand_name": "Calpol Syrup", "generic_name": "Paracetamol",
        "aliases": ["calpol syrup", "calpol susp"],
        "strength": "250mg/5ml", "dosage_form": "syrup",
    },
    {
        "brand_name": "Meftal", "generic_name": "Mefenamic Acid",
        "aliases": ["meftal", "meftal 250"],
        "strength": "250mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Meftal-P", "generic_name": "Mefenamic Acid + Paracetamol",
        "aliases": ["meftal p", "meftalp", "meftal-p"],
        "strength": "250mg + 325mg", "dosage_form": "tablet",
        "combination_components": [
            {"strength": 250, "unit": "mg", "name": "Mefenamic Acid"},
            {"strength": 325, "unit": "mg", "name": "Paracetamol"},
        ],
    },
    {
        "brand_name": "Combiflam", "generic_name": "Ibuprofen + Paracetamol",
        "aliases": ["combiflam"],
        "strength": "400mg + 325mg", "dosage_form": "tablet",
        "combination_components": [
            {"strength": 400, "unit": "mg", "name": "Ibuprofen"},
            {"strength": 325, "unit": "mg", "name": "Paracetamol"},
        ],
    },
    # -- Antibiotics --
    {
        "brand_name": "Augmentin 625", "generic_name": "Amoxicillin + Clavulanic Acid",
        "aliases": ["augmentin", "augmentin625", "augmentin 625", "augmentin-625"],
        "strength": "500mg + 125mg", "dosage_form": "tablet",
        "combination_components": [
            {"strength": 500, "unit": "mg", "name": "Amoxicillin"},
            {"strength": 125, "unit": "mg", "name": "Clavulanic Acid"},
        ],
    },
    {
        "brand_name": "Azithral 500", "generic_name": "Azithromycin",
        "aliases": ["azithral", "azithral500", "azithral 500", "azithral-500"],
        "strength": "500mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Zifi 200", "generic_name": "Cefixime",
        "aliases": ["zifi", "zifi200", "zifi 200", "zifi-200"],
        "strength": "200mg", "dosage_form": "tablet",
    },
    # -- Antihistamines --
    {
        "brand_name": "Allegra 120", "generic_name": "Fexofenadine",
        "aliases": ["allegra", "allegra120", "allegra 120", "allegra-120"],
        "strength": "120mg", "dosage_form": "tablet",
    },
    # -- Antacids / PPIs --
    {
        "brand_name": "Pan 40", "generic_name": "Pantoprazole",
        "aliases": ["pan", "pan40", "pan 40", "pan-40"],
        "strength": "40mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Pantocid 40", "generic_name": "Pantoprazole",
        "aliases": ["pantocid", "pantocid40", "pantocid 40", "pantocid-40"],
        "strength": "40mg", "dosage_form": "tablet",
    },
    # -- Antihypertensives --
    {
        "brand_name": "Amlopres 5", "generic_name": "Amlodipine",
        "aliases": ["amlopres", "amlopres5", "amlopres 5", "amlopres-5"],
        "strength": "5mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Telma 40", "generic_name": "Telmisartan",
        "aliases": ["telma", "telma40", "telma 40", "telma-40"],
        "strength": "40mg", "dosage_form": "tablet",
    },
    # -- Antidiabetics --
    {
        "brand_name": "Glycomet 500", "generic_name": "Metformin",
        "aliases": ["glycomet", "glycomet500", "glycomet 500", "glycomet-500"],
        "strength": "500mg", "dosage_form": "tablet",
    },
    {
        "brand_name": "Januvia 100", "generic_name": "Sitagliptin",
        "aliases": ["januvia", "januvia100", "januvia 100", "januvia-100"],
        "strength": "100mg", "dosage_form": "tablet",
    },
]
