"""
Interactive verification script to test the end-to-end vision pipeline with a synthesized image.
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.ingest import ingest_pdf
from rag.embedder import store_chunks, chunk_already_exists
from rag.vision import run_image_pipeline

def run_verification():
    print("=== Step 1: Ingest sample PDF ===")
    ROOT_DIR = Path(__file__).parent.parent
    sample_pdf = ROOT_DIR / "data" / "uploaded" / "SafeGuard_Motor_Policy_2024.pdf"
    if sample_pdf.exists():
        res = ingest_pdf(str(sample_pdf))
        new_chunks = [c for c in res["chunks"] if not chunk_already_exists(c["chunk_id"])]
        if new_chunks:
            stored = store_chunks(new_chunks)
            print(f"Stored {stored} chunks from {sample_pdf.name}")
        else:
            print("Chunks already indexed in DB.")
    else:
        print(f"Sample PDF {sample_pdf} not found.")

    print("\n=== Step 2: Create synthetic label image ===")
    img = Image.new("RGB", (400, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "INCIDENT REPORT\nVehicle: SafeGuard Policy\nDamage: Windscreen cracked", fill=(0, 0, 0))
    test_img_path = ROOT_DIR / "tests" / "test_windscreen.png"
    img.save(test_img_path)
    print(f"Created {test_img_path}")

    print("\n=== Step 3: Run pipeline for 'Found' case ===")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    os.environ["DEBUG_VISION"] = "1"
    result = run_image_pipeline(
        image_bytes=img_bytes,
        filename="test_windscreen.png",
        user_question="Is windscreen damage covered under SafeGuard motor policy, and what is the excess or limit?"
    )

    print("\nResult Answer:")
    print(result["answer"])
    print("\nConfidence:", result["confidence"])
    print("Sources Found:", len(result["sources"]))

    print("\n=== Step 4: Run pipeline for 'Not in Documents' case ===")
    # Unrelated image
    img_unrelated = Image.new("RGB", (400, 100), color=(240, 240, 240))
    draw = ImageDraw.Draw(img_unrelated)
    draw.text((20, 40), "Recipe for Italian Risotto", fill=(0, 0, 0))
    unrelated_path = ROOT_DIR / "tests" / "test_recipe.png"
    img_unrelated.save(unrelated_path)

    with open(unrelated_path, "rb") as f:
        recipe_bytes = f.read()

    result_unrelated = run_image_pipeline(
        image_bytes=recipe_bytes,
        filename="test_recipe.png",
        user_question="What ingredients are listed and does this match coverage?"
    )

    print("\nUnrelated Result:")
    print("Answer:", result_unrelated["answer"])
    print("Confidence:", result_unrelated["confidence"])
    print("Sources:", len(result_unrelated["sources"]))

if __name__ == "__main__":
    run_verification()
