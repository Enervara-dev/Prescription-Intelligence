# Prescription test fixtures

## Honest status (read this before adding or trusting anything here)

**This repository does not contain any real prescription images, and never
has.** Every existing test (before and after this fixture directory was
added) constructs its OCR input synthetically — either as a hand-typed
string standing in for `medicine_text`, or, in `tests/test_pipeline_end_to_end.py`
and `tests/test_ocr_geometry.py`, as hand-built `OCRWord` objects with
made-up (but structurally realistic) coordinates and confidence values.

**Nothing under this directory is a real-world accuracy benchmark.** A test
passing against a synthetic fixture here says the pipeline behaves
correctly against *that specific, hand-constructed input* — it says nothing
about accuracy against a real photograph of a real prescription, real
handwriting, real paper glare, real skew, or a real printer's font. Do not
report or imply otherwise. Do not add a "confidence: 98% accurate on real
prescriptions" style claim anywhere in this codebase based on these tests.

## Structure

```
fixtures/prescriptions/
  printed/         Synthetic OCR-structure fixtures simulating a clean,
                    well-aligned printed prescription (predictable word
                    heights, small consistent row gaps).
  handwritten/      Synthetic fixtures simulating handwriting-shaped
                    uncertainty: lower per-word confidence, more irregular
                    word heights/gaps — NOT a claim about how Google
                    Vision's document_text_detection actually performs on
                    real handwriting, which has never been measured here
                    (see the OCR/matching forensic audit, section 9: "the
                    architecture is fundamentally dependent on generic OCR
                    for handwriting, with no compensating layer").
  poor_quality/     Synthetic fixtures simulating a low-resolution / poorly
                    lit source: smaller word heights relative to row
                    spacing (the exact condition verified to break the old
                    fixed-pixel-threshold line grouping), and several words
                    below a confidence floor.
  multi_medicine/   Synthetic fixtures with more than one medicine on a
                    page/line, exercising the span-based, multi-medicine
                    scanner and per-medicine instruction-window association.
```

Each fixture is a small JSON file: a list of words, each with
`text, confidence, x, y, width, height` (the same shape as
`app.services.ocr_geometry.OCRWord`), an `expected_medicines` list (each
entry: `name`, `strength`, `dosage`, `frequency`, `duration` — as this
pipeline should extract them from the corresponding catalog rows the test
seeds), and a `notes` field explaining what real-world condition the
fixture is a *stand-in* for and what it deliberately does NOT validate.

## What should happen when real images become available

If real prescription photographs are ever added to this project (with
appropriate consent/authorization to use them for testing):

1. Add the image file itself under the matching category directory.
2. Capture and save the RAW Google Vision `document_text_detection`
   response (or at minimum the word list with bounding boxes/confidence)
   alongside it — this is what lets a future engineer compare "what Vision
   actually saw" against "what the parser produced," exactly as the
   forensic audit's section 2 asks for, instead of only ever inspecting the
   final parsed JSON.
3. The `expected_medicines` field must be filled in by a human who can
   actually read the source image — never fabricated or guessed by an
   automated process. An empty or `null` expected value is more honest than
   a plausible-looking invented one.
4. Update this README's status section — it should stop saying "no real
   images exist" the day that stops being true.
