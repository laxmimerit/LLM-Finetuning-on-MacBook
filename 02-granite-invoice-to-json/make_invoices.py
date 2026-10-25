"""Write messy invoice text + the exact JSON it should become.

python make_invoices.py 2000 data/train.jsonl [seed]

Each row is {"text": ..., "json": {...}}: the text looks like what pypdf pulls out of a PDF (one table cell per line,
labels and values split apart, page footers, bank details), and the JSON is known exactly because we wrote both.
Names, layouts and numbers are all made up here; nothing comes from the test set.
"""
import json
import random
import sys
from datetime import date, timedelta

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "currency", "subtotal", "tax", "total", "document_type"]

PLACES = ["Silverbrook", "Harbourside", "Redcliff", "Maple Ridge", "Oakfield", "Bluewater", "Stonegate", "Westmarch",
          "Kingsbury", "Lakeshore", "Copperhill", "Riverbend", "Eastwood", "Northgate", "Sunvale", "Ashford",
          "Brookhaven", "Cedar Point", "Highfield", "Marsh Lane", "Pinecrest", "Rosedale", "Tidewater", "Windmere",
          "Ironwood", "Goldcrest", "Thornbury", "Fairhaven", "Larkspur", "Mistral", "Okapi", "Baobab", "Mopani",
          "Kestrel", "Umhlanga", "Rhine Valley", "Seine", "Tamar", "Clyde", "Avon", "Delta", "Summit Ridge"]
TRADES = ["Building Supplies", "Electrical", "Printing", "Catering", "IT Solutions", "Plumbing", "Auto Spares",
          "Medical Supplies", "Security Services", "Cleaning Services", "Signage", "Farm Supplies", "Timber",
          "Steel Works", "Paints", "Logistics", "Couriers", "Laboratory Supplies", "Furniture", "Textiles",
          "Pumps & Valves", "Fasteners", "Consulting", "Hydraulics", "Electronics", "Stationers", "Freight Lines",
          "Clearing Agents", "Haulage", "Packaging Solutions", "Tools & Hardware", "Marine Services"]

LOCALES = {
    # currency, symbols, legal forms, tax label, tax rates, date styles, number style
    "US": ("USD", ["$", "USD ", "US$"], ["Inc", "LLC", "Corp", "Co", "Company"], ["Sales Tax", "Tax", "State Tax"],
           [0.0625, 0.07, 0.0825, 0.06, 0.0475, 0.0], ["mdy", "long_us", "iso"], "us"),
    "GB": ("GBP", ["£", "GBP "], ["Ltd", "Limited", "plc", "LLP"], ["VAT"], [0.20, 0.05, 0.0],
           ["dmy", "long_gb", "dmy_dash"], "us"),
    "DE": ("EUR", ["€", "EUR ", ""], ["GmbH", "AG", "GmbH & Co. KG", "KG"], ["MwSt.", "USt.", "VAT"], [0.19, 0.07],
           ["dmy_dot", "iso"], "eu_dot"),
    "FR": ("EUR", ["€", "EUR "], ["SARL", "SAS", "SA"], ["TVA"], [0.20, 0.10, 0.055], ["dmy", "long_fr"], "eu_space"),
    "NL": ("EUR", ["€", "EUR "], ["B.V.", "N.V."], ["BTW", "VAT"], [0.21, 0.09], ["dmy_dash", "dmy"], "eu_dot"),
    "ZA": ("ZAR", ["R", "R ", "ZAR "], ["(Pty) Ltd", "CC", "Pty Ltd", "Ltd"], ["VAT", "Output VAT"], [0.15, 0.0],
           ["dmy", "ymd_slash", "iso", "long_gb"], "space_groups"),
    "NA": ("NAD", ["N$", "NAD ", "$"], ["(Pty) Ltd", "CC"], ["VAT"], [0.15, 0.0], ["dmy", "ymd_slash", "long_gb"],
           "us"),
    "ZM": ("ZMW", ["K", "ZMW ", "K "], ["Ltd", "Limited"], ["VAT"], [0.16, 0.0], ["dmy", "long_gb"], "us"),
    "IN": ("INR", ["₹", "Rs. ", "INR "], ["Pvt Ltd", "Private Limited", "LLP"], ["GST", "IGST", "CGST + SGST"],
           [0.18, 0.12, 0.05], ["dmy", "dmy_dash", "long_gb"], "in"),
    "AU": ("AUD", ["$", "A$", "AUD "], ["Pty Ltd", "Pty. Ltd."], ["GST"], [0.10, 0.0], ["dmy", "long_gb"], "us"),
    "CA": ("CAD", ["$", "C$", "CAD "], ["Inc.", "Ltd.", "Corp."], ["GST", "HST", "GST/HST"], [0.05, 0.13, 0.15],
           ["iso", "long_us", "mdy"], "us"),
}
GEO = {   # cities, phone code, web domain, country name: the clues a real letterhead carries
    "US": (["Denver", "Austin", "Columbus", "Tampa", "Boise", "Albany"], "+1", ".com", "USA"),
    "GB": (["Leeds", "Bristol", "Glasgow", "Cardiff", "Norwich"], "+44", ".co.uk", "United Kingdom"),
    "DE": (["Hamburg", "Köln", "Leipzig", "Dortmund"], "+49", ".de", "Deutschland"),
    "FR": (["Lyon", "Lille", "Nantes", "Toulouse"], "+33", ".fr", "France"),
    "NL": (["Utrecht", "Eindhoven", "Groningen"], "+31", ".nl", "Nederland"),
    "ZA": (["Johannesburg", "Durban", "Pretoria", "Gqeberha", "Bloemfontein", "Polokwane"], "+27", ".co.za", "South Africa"),
    "NA": (["Windhoek", "Walvis Bay", "Swakopmund", "Oshakati", "Keetmanshoop"], "+264", ".com.na", "Namibia"),
    "ZM": (["Lusaka", "Ndola", "Kitwe", "Livingstone"], "+260", ".co.zm", "Zambia"),
    "IN": (["Pune", "Chennai", "Jaipur", "Kochi", "Indore"], "+91", ".in", "India"),
    "AU": (["Brisbane", "Perth", "Adelaide", "Hobart"], "+61", ".com.au", "Australia"),
    "CA": (["Calgary", "Ottawa", "Halifax", "Winnipeg"], "+1", ".ca", "Canada"),
}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
ITEMS = ["Copier paper A4 80gsm (box of 5 reams)", "LED panel light 600x600", "Labour - site visit", "Annual support licence",
         "Steel angle 50x50x5 6m", "PVC pipe 110mm x 6m", "Cable tie 300mm (pack 100)", "Toner cartridge black",
         "Consulting hours", "Pallet wrap 500mm", "Safety boots size 9", "Hi-vis vest XL", "Diesel filter",
         "Brake pads front set", "Paint 20L white", "Delivery to site", "Server rack 42U", "Network switch 24-port",
         "Ink cartridge colour", "Gloves nitrile (box 100)", "Office chair, mesh back", "Desk 1600x800",
         "Bolts M10 x 40 (box 100)", "Hydraulic hose 1/2\" per m", "Pump impeller", "Courier: overnight parcel",
         "Container handling fee", "Customs entry preparation", "Road freight per load", "Cleaning service - monthly",
         "Printing: flyers A5 (1000)", "Catering: lunch for 20", "Cement 50kg", "Timber 38x114 per m", "Website hosting (12 months)"]


def money(x, style, sym=""):
    s = f"{abs(x):,.2f}"
    if style == "eu_dot":
        s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    elif style == "eu_space":
        s = s.replace(",", " ").replace(".", ",")
    elif style == "space_groups":
        s = s.replace(",", " ") if random.random() < 0.6 else s
    elif style == "in" and abs(x) >= 100000:   # Indian grouping 1,23,456.78
        whole, frac = f"{abs(x):.2f}".split(".")
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2: groups.insert(0, head[-2:]); head = head[:-2]
        if head: groups.insert(0, head)
        s = ",".join(groups + [tail]) + "." + frac
    return sym + s


def fmt_date(d, style):
    return {"mdy": f"{d.month:02d}/{d.day:02d}/{d.year}", "dmy": f"{d.day:02d}/{d.month:02d}/{d.year}",
            "dmy_dash": f"{d.day:02d}-{d.month:02d}-{d.year}", "dmy_dot": f"{d.day:02d}.{d.month:02d}.{d.year}",
            "ymd_slash": f"{d.year}/{d.month:02d}/{d.day:02d}", "iso": d.isoformat(),
            "long_us": f"{MONTHS[d.month - 1]} {d.day}, {d.year}", "long_gb": f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}",
            "long_fr": f"{d.day} {MOIS[d.month - 1]} {d.year}"}[style]


def inv_number(credit):
    r = random.random
    n = random.randint(100, 999999)
    if credit:
        return random.choice([f"CN-{random.randint(2024, 2026)}-{n % 10000:04d}", f"CR{n:06d}", f"CRN-{n % 100000:05d}", f"NC{n:05d}"])
    return random.choice([f"INV-{n:06d}", f"{random.randint(2024, 2026)}-{n % 10000:04d}", f"{n:07d}", f"TX{n:06d}",
                          f"{''.join(random.choices('ABCDEFGHJKLMNPRSTUVWXYZ', k=2))}-{n % 100000:05d}", f"SI/{n % 10000}/{random.randint(24, 26)}"])


def one():
    loc = random.choice(list(LOCALES))
    cur, syms, legal, taxlabels, rates, dstyles, nstyle = LOCALES[loc]
    sym = random.choice(syms)
    vendor = f"{random.choice(PLACES)} {random.choice(TRADES)} {random.choice(legal)}"
    customer = f"{random.choice(PLACES)} {random.choice(TRADES)} {random.choice(legal)}"
    credit = random.random() < 0.12
    d = date(2024, 1, 1) + timedelta(days=random.randint(0, 1000))
    dstyle = random.choice(dstyles)
    number = inv_number(credit)
    rate = random.choice(rates)
    inclusive = rate > 0 and random.random() < 0.15        # prices printed with VAT in them
    n_lines = random.choice([1, 2, 3, 3, 4, 5, 6, 8, 12, 20, 30])
    lines = []
    for _ in range(n_lines):
        q = random.choice([1, 1, 1, 2, 3, 4, 5, 10, 12, 20, 50])
        price = round(random.choice([random.uniform(2, 60), random.uniform(60, 900), random.uniform(900, 9000)]), 2)
        lines.append((random.choice(ITEMS), q, price, round(q * price, 2)))
    lines_sum = round(sum(l[3] for l in lines), 2)
    extras = []                                             # (label, amount) between subtotal and total, tax-free
    if not credit and random.random() < 0.35:
        extras.append((random.choice(["Shipping", "Freight", "Delivery", "S&H", "Shipping & Handling", "Handling", "Fuel surcharge", "Service fee"]),
                       round(random.uniform(10, 400), 2)))
    if not credit and random.random() < 0.25:
        extras.append((random.choice(["Discount", "Trade discount", "Promo code SAVE10", "Less: credit applied", "Early payment discount"]),
                       -round(lines_sum * random.choice([0.02, 0.05, 0.1]), 2)))
    if inclusive:
        subtotal = round(lines_sum / (1 + rate), 2)
        tax = round(lines_sum - subtotal, 2)
    else:
        subtotal = lines_sum
        tax = round(subtotal * rate, 2)
    total = round(subtotal + tax + sum(a for _, a in extras), 2)
    doc_type = "credit_note" if credit else "invoice"
    truth = {"vendor_name": vendor, "invoice_number": number, "invoice_date": d.isoformat(), "currency": cur,
             "subtotal": subtotal, "tax": tax, "total": total, "document_type": doc_type}

    # ---- the text, as a PDF text layer would give it
    sign = (lambda s: s) if not credit else random.choice([lambda s: "-" + s, lambda s: f"({s})", lambda s: s + " CR", lambda s: s])
    M = lambda x, neg=False: sign(money(x, nstyle, sym)) if credit else ("-" if x < 0 else "") + money(x, nstyle, sym)
    title = ("CREDIT NOTE" if credit else random.choice(["INVOICE", "TAX INVOICE", "Invoice", "Tax Invoice", "BILL", "Facture" if loc == "FR" else "INVOICE", "Rechnung" if loc == "DE" else "Invoice"]))
    numlab = random.choice(["Invoice No:", "Invoice #", "Invoice Number", "Inv. No.", "No.", "Document No:", "Our ref:"]) if not credit else random.choice(["Credit Note No:", "CN No.", "Credit note #", "Document No:"])
    datelab = random.choice(["Date:", "Invoice Date", "Date of issue", "Tax point", "Dated", "Datum:" if loc == "DE" else "Date"])
    head = [(numlab, number), (datelab, fmt_date(d, dstyle))]
    noise = []
    if random.random() < 0.6: noise.append((random.choice(["PO Number:", "Order No:", "Your Ref:", "Customer Order"]), f"PO-{random.randint(1000, 99999)}"))
    if random.random() < 0.5:
        due = d + timedelta(days=random.choice([7, 14, 30, 45, 60]))
        noise.append((random.choice(["Due Date:", "Payment due", "Due"]), fmt_date(due, dstyle)))
    if random.random() < 0.4: noise.append((random.choice(["Account No:", "Customer ID", "Acc #"]), str(random.randint(10000, 9999999))))
    if random.random() < 0.3: noise.append(("Terms:", random.choice(["Net 30", "30 days", "COD", "EOM", "14 days nett"])))
    if credit and random.random() < 0.7: noise.append(("Original invoice:", inv_number(False)))
    head += noise
    random.shuffle(head)
    vat_no = f"{random.choice(['VAT Reg No', 'Tax ID', 'VAT No.', 'Reg. No.', 'GSTIN' if loc == 'IN' else 'TIN'])}: {random.randint(10**8, 10**10)}"
    addr = f"{random.randint(1, 400)} {random.choice(['Main Road', 'Station St', 'Industrial Ave', 'Harbour Rd', 'Church Street', 'Long St'])}"
    out = []
    cities, phone, tld, country = GEO[loc]
    city = random.choice(cities) + (f", {country}" if random.random() < 0.3 else "")
    slug = "".join(ch for ch in vendor.split()[0].lower() if ch.isalpha())
    contact = random.choice(["", f"Tel {phone} {random.randint(10, 99)} {random.randint(100, 999)} {random.randint(1000, 9999)}",
                             f"Tel: {phone} {random.randint(100, 999)} {random.randint(1000, 9999)} | accounts@{slug}{tld}", f"www.{slug}{tld}"])
    top = [vendor, addr, city, contact, vat_no]
    if random.random() < 0.3: top = [title] + top
    else: top = top + [title]
    out += [t for t in top if t]
    split = random.random()                                 # how the header box came out of the PDF
    if split < 0.4: out += [f"{a} {b}" for a, b in head]
    elif split < 0.7: out += [a for a, _ in head] + [b for _, b in head]
    else:
        for a, b in head: out += [a, b]
    out += [random.choice(["Bill To:", "Customer:", "Invoice to", "Sold To"]), customer,
            f"{random.randint(1, 400)} {random.choice(['Park Lane', 'Market St', 'Depot Rd', 'Industria'])}", random.choice(cities)]
    if random.random() < 0.3 and cur in ("USD", "NAD", "ZAR"): out.append(f"All amounts in {cur}" if random.random() < 0.5 else f"Currency: {cur}")
    cols = random.choice([["Description", "Qty", "Unit Price", "Amount"], ["Item", "Quantity", "Rate", "Total"],
                          ["Description", "Qty", "Price", "Line Total"], ["Qty", "Description", "Unit", "Amount"]])
    pages = 1 if n_lines < 12 else random.choice([1, 2, 2, 3])
    per = -(-n_lines // pages)
    out += cols
    running = 0.0
    for k, (desc, q, price, amt) in enumerate(lines):
        if k and k % per == 0:
            p = k // per
            out += [f"Carried forward {M(running)}" if random.random() < 0.5 else "Carried forward", *([] if random.random() < 0.5 else [M(running)]),
                    f"Page {p} of {pages}", vendor, f"{numlab} {number}", f"Brought forward {M(running)}", *cols]
        row = [desc, str(q), money(price, nstyle), M(amt)] if cols[0] != "Qty" else [str(q), desc, money(price, nstyle), M(amt)]
        if random.random() < 0.15: row = [" ".join(row)]
        out += row
        running = round(running + amt, 2)
    tl = random.choice(taxlabels)
    pct = random.choice(["", f" {rate * 100:g}%", f" @ {rate * 100:g}%", f" ({rate * 100:g}%)"]) if rate else ""
    totlab = random.choice(["TOTAL", "Total", "Total Due", "Amount Due", "Grand Total", "Total (incl. VAT)" if rate else "Total", "Balance Due"])
    if inclusive:
        block = [("Total (incl. " + tl + ")", total), (tl + " included" + pct, tax), ("Total excl. " + tl, subtotal)]
    else:
        block = [(random.choice(["Subtotal", "Sub Total", "Sub-total", "Net Amount", "Total excl. tax"]), subtotal)]
        block += extras
        block += [(tl + pct if rate else random.choice([f"{tl} 0%", f"{tl} (zero-rated)", f"{tl} exempt"]), tax), (totlab, total)]
    for a, b in block:
        if random.random() < 0.5: out += [a, M(b)]
        else: out.append(f"{a} {M(b)}")
    if random.random() < 0.3 and not credit:
        paid = round(total * random.choice([0.0, 0.5, 1.0]), 2)
        out += [f"Amount paid {M(paid)}", f"Balance {M(round(total - paid, 2))}"]
    if random.random() < 0.6:
        out.append(f"Banking details: {random.choice(['First National', 'City Bank', 'Standard', 'Barclays', 'Commerz'])} "
                   f"Acc: {random.randint(10**8, 10**10)} Branch: {random.randint(100000, 999999)}")
    if random.random() < 0.5: out.append(random.choice(["Thank you for your business!", "E&OE", "Goods remain our property until paid in full."]))
    out.append(f"Page {pages} of {pages}" if pages > 1 or random.random() < 0.5 else "")
    return {"text": "\n".join(x for x in out if x), "json": truth}


if __name__ == "__main__":
    n, path = int(sys.argv[1]), sys.argv[2]
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    with open(path, "w") as f:
        for _ in range(n):
            f.write(json.dumps(one(), ensure_ascii=False) + "\n")
    print(f"wrote {n} invoices to {path}")
