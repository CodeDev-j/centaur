from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path

# Setup
file_path = Path("data/inputs/Alphabet_2025.Q1_Earnings.Slides - Copy-05.pdf")
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_ocr = True
pipeline_opts.generate_page_images = True
pipeline_opts.images_scale = 3.0

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_opts)
    }
)

print(f"🚀 Parsing {file_path.name}...")
result = converter.convert(file_path)
doc = result.document

print("\n--- ITEM DUMP (Page 1) ---")
found_tac = False
for item, level in doc.iterate_items():
    if item.prov and item.prov[0].page_no == 1:
        label = item.label.value
        # Check if it has text
        text = getattr(item, "text", "")[:50].replace("\n", " ")
        
        print(f"[{label.upper()}] {text} | BBox: {item.prov[0].bbox}")
        
        if "Total TAC" in text:
            found_tac = True
            print("   >>> ✅ FOUND 'Total TAC' here!")

if not found_tac:
    print("\n❌ CRITICAL: 'Total TAC' was NOT found as a discrete text item.")
    print("   Hypothesis confirmed: Docling merged it into the Chart/Picture item.")
else:
    print("\n✅ 'Total TAC' exists. The issue is strictly the search margins.")