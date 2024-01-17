
import fitz  

def process_uploaded_file(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text_list = []

    for page_num in range(pdf_document.page_count):
        try:
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            text_list.append(text)

        except Exception as e:
            print(f"Exception during processing (Page {page_num + 1}): {str(e)}")

    pdf_document.close()
    for i, text in enumerate(text_list, start=1):
        print(f"Page {i}:\n{text}\n")
    return text_list









