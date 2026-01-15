import streamlit as st
import json
import re
import time
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
import pandas as pd
from jsonschema import validate, ValidationError
from openai import OpenAI
from io import BytesIO

# -------------------------
# Login
# -------------------------
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == st.secrets["USERNAME"] and password == st.secrets["PASSWORD"]:
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login()
    st.stop()

st.title("Welcome! You are logged in.")

# -------------------------
# OpenAI Client (after login)
# -------------------------
def get_openai_client():
    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return st.session_state.client

client = get_openai_client()

# -------------------------
# Invoice JSON Schema
# -------------------------
invoice_schema = {
    "type": "object",
    "properties": {
        "Invoice Number": {"type": ["string", "null"]},
        "Invoice Date": {"type": ["string", "null"]},
        "Vendor Name": {"type": ["string", "null"]},
        "Vendor Address": {"type": ["string", "null"]},
        "Buyer Name": {"type": ["string", "null"]},
        "Buyer Address": {"type": ["string", "null"]},
        "Items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Item Name": {"type": ["string", "null"]},
                    "SAC #": {"type": ["string", "null"]},
                    "Quantity": {"type": ["string", "null"]},
                    "Rate": {"type": ["string", "null"]},
                    "Value": {"type": ["string", "null"]}
                },
                "required": ["Item Name", "Quantity", "Rate", "Value"]
            }
        },
        "Total GST": {"type": ["string", "null"]},
        "Grand Total": {"type": ["string", "null"]}
    },
    "required": ["Invoice Number", "Invoice Date", "Vendor Name", "Buyer Name", "Items"]
}

# -------------------------
# Helper Functions
# -------------------------
def extract_text_from_file(file, max_pages=5, resize_factor=0.5):
    """Extract text from PDF or image using EasyOCR with memory-safe handling"""
    text = ""
    file_ext = file.name.split(".")[-1].lower()
    reader = easyocr.Reader(['en'], gpu=False)

    if file_ext == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            if page_num >= max_pages:
                break
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Resize to reduce memory usage
            img = img.resize((int(img.width*resize_factor), int(img.height*resize_factor)))
            result = reader.readtext(np.array(img))
            text += " ".join([r[1] for r in result]) + "\n"
    elif file_ext in ["jpg", "jpeg", "png", "tiff"]:
        img = Image.open(file)
        img = img.resize((int(img.width*resize_factor), int(img.height*resize_factor)))
        result = reader.readtext(np.array(img))
        text = " ".join([r[1] for r in result])
    else:
        raise ValueError("Unsupported file type. Upload PDF or image.")

    return text.strip()


def extract_invoice(inv_text, retries=2):
    """Call OpenAI to extract structured JSON from invoice text"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert in parsing Indian GST invoices. "
                "Extract ONLY structured JSON, no extra text. "
                "Return null for missing values. "
                "Invoice Number could appear as Invoice No, Inv No, Bill No, Bill Number, Invoice #."
            ),
        },
        {
            "role": "user",
            "content": inv_text + "\n\nPlease extract as JSON with keys: "
                                "Invoice Number, Invoice Date, Vendor Name, Vendor Address, "
                                "Buyer Name, Buyer Address, Items (array with Item Name, SAC #, Quantity, Rate, Value), "
                                "Total GST, Grand Total."
        }
    ]

    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0,
            max_tokens=1500
        )

        text = resp.choices[0].message.content.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'(\{.*\})', text, re.S)
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if data is None:
            time.sleep(1 + 2 * attempt)
            continue

        try:
            validate(instance=data, schema=invoice_schema)
        except ValidationError:
            time.sleep(1 + 2 * attempt)
            continue

        return data

    raise RuntimeError("Failed to extract valid JSON after retries")


def prepare_tabular_data(invoice_json):
    """Flatten invoice JSON into a tabular DataFrame"""
    items = invoice_json.get("Items", [])
    rows = []
    for item in items:
        row = {
            "Invoice Number": invoice_json.get("Invoice Number"),
            "Invoice Date": invoice_json.get("Invoice Date"),
            "Vendor Name": invoice_json.get("Vendor Name"),
            "Vendor Address": invoice_json.get("Vendor Address"),
            "Buyer Name": invoice_json.get("Buyer Name"),
            "Buyer Address": invoice_json.get("Buyer Address"),
            "Item Name": item.get("Item Name"),
            "SAC #": item.get("SAC #"),
            "Quantity": item.get("Quantity"),
            "Rate": item.get("Rate"),
            "Value": item.get("Value"),
            "Total GST": invoice_json.get("Total GST"),
            "Grand Total": invoice_json.get("Grand Total")
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Invoice Data Extractor", layout="wide")
st.title("📄 Invoice Data Extractor (PDF/Image)")

uploaded_file = st.file_uploader("Upload Invoice PDF or Image", type=["pdf","jpg","jpeg","png","tiff", "webp"])

if uploaded_file:
    if st.button("Extract Data"):
        with st.spinner("Extracting invoice data..."):
            try:
                # Memory-safe text extraction
                inv_text = extract_text_from_file(uploaded_file, max_pages=5, resize_factor=0.5)

                result = extract_invoice(inv_text)
                st.success("✅ Extraction Successful!")

                # Display raw JSON
                st.subheader("Raw JSON Output")
                st.json(result)

                # Prepare tabular DataFrame
                df = prepare_tabular_data(result)
                st.subheader("Invoice Details Table")
                st.dataframe(df)

                # Validation Block
                if df['Item Name'].str.contains('Stanley Hammer', na=False).any():
                    st.warning("Unauthorized material detected!")

                # Download CSV
                csv_buffer = BytesIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                st.download_button(
                    label="📥 Download Extracted Data as CSV",
                    data=csv_buffer,
                    file_name="extracted_invoice.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Extraction Failed: {e}")


