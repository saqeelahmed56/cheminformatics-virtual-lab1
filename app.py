import io
import os
from datetime import date

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors

# ---------------- CONFIG (edit these) ----------------
COLLEGE = "Your College Name"
DEPARTMENT = "Department of Chemistry"
DEVELOPER = "Your Name"
DEVELOPER_ROLE = "Student Developer"
GUIDE = "Your Sir's Name"
GUIDE_ROLE = "Assistant Professor"
LOGO_PATH = "logo.png"  # optional
GITHUB_URL = "https://github.com/your-username/cheminformatics-virtual-lab"
# -----------------------------------------------------

st.set_page_config(page_title="Cheminformatics Virtual Lab", page_icon="🧪", layout="wide")

EXAMPLES = {
    "Ethanol": "CCO",
    "Benzene": "c1ccccc1",
    "Phenol": "Oc1ccccc1",
    "Acetic Acid": "CC(=O)O",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}

PAGES = [
    "🏠 Home",
    "📖 Theory",
    "🧬 Molecular Visualization",
    "📊 Molecular Properties Report",
    "📈 Structure–Property Analysis",
    "📝 Assessment",
    "📄 Final Project Report",
]

QUIZ = [
    ("What is the full form of SMILES?",
     ["Simplified Molecular Input Line Entry System",
      "Standard Molecular Information Language Encoding System",
      "Simple Molecular Identification and Labeling System"], 0),
    ("What does a SMILES string represent?",
     ["A molecular structure using text", "A spectroscopy technique", "A laboratory instrument"], 0),
    ("Which SMILES represents ethanol?", ["CCO", "c1ccccc1", "CC(=O)O"], 0),
    ("Which SMILES represents benzene?", ["c1ccccc1", "CCO", "O=C=O"], 0),
    ("In SMILES, parentheses are mainly used to indicate:",
     ["Branches", "Molecular weight", "Atom colour"], 0),
    ("Which property is commonly associated with molecular lipophilicity?",
     ["LogP", "TPSA", "Ring count"], 0),
    ("TPSA stands for:",
     ["Topological Polar Surface Area", "Total Proton Spin Angle", "Thermal Phase Stability Analysis"], 0),
]


# ---------------- helpers ----------------
def get_mol(smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def bond_counts(mol):
    """Return heavy-atom bonds, sigma, pi and total bonds incl. hydrogens."""
    heavy_bonds = mol.GetNumBonds()
    mh = Chem.AddHs(mol)
    Chem.Kekulize(mh, clearAromaticFlags=True)
    sigma = pi = 0
    for b in mh.GetBonds():
        order = int(round(b.GetBondTypeAsDouble()))
        sigma += 1
        pi += max(order - 1, 0)
    return heavy_bonds, sigma, pi, sigma + pi


def properties(mol) -> dict:
    heavy_bonds, sigma, pi, total = bond_counts(mol)
    return {
        "Molecular Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight (g/mol)": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Descriptors.MolLogP(mol), 3),
        "TPSA (Å²)": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "Atoms (incl. H)": Chem.AddHs(mol).GetNumAtoms(),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Heavy-Atom Bonds": heavy_bonds,
        "Sigma (σ) Bonds": sigma,
        "Pi (π) Bonds": pi,
        "Total Bonds (σ + π)": total,
        "Ring Count": rdMolDescriptors.CalcNumRings(mol),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(mol),
        "Fraction Csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
    }


def draw_2d(mol, size=(450, 350)):
    return Draw.MolToImage(mol, size=size)


def show_3d(mol, style="Ball and Stick", height=420):
    import py3Dmol

    mh = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mh, randomSeed=42) != 0:
        st.warning("3D coordinates could not be generated for this molecule.")
        return
    try:
        AllChem.MMFFOptimizeMolecule(mh)
    except Exception:
        pass
    view = py3Dmol.view(width=600, height=height)
    view.addModel(Chem.MolToMolBlock(mh), "mol")
    if style == "Ball and Stick":
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.28}})
    elif style == "Stick":
        view.setStyle({"stick": {}})
    elif style == "Space Filling":
        view.setStyle({"sphere": {}})
    elif style == "Wireframe":
        view.setStyle({"line": {}})
    else:  # Cartoon (falls back to sticks for small molecules)
        view.setStyle({"cartoon": {"color": "spectrum"}, "stick": {}})
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=height + 10)


def ascii_safe(text: str) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def build_pdf(name, reg, smiles_list) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for smi in smiles_list:
        mol = get_mol(smi)
        if mol is None:
            continue
        pdf.add_page()
        if os.path.exists(LOGO_PATH):
            pdf.image(LOGO_PATH, x=10, y=8, h=18)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(10, 30)
        pdf.cell(0, 8, "Cheminformatics Virtual Laboratory", ln=1, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, ascii_safe(f"{COLLEGE} | {DEPARTMENT}"), ln=1, align="C")
        pdf.ln(4)
        pdf.cell(0, 6, ascii_safe(f"Student: {name or '-'}   Reg. No: {reg or '-'}   Date: {date.today()}"), ln=1)
        pdf.cell(0, 6, ascii_safe(f"SMILES: {smi}"), ln=1)
        pdf.ln(2)
        buf = io.BytesIO()
        draw_2d(mol, (400, 300)).save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, x=10, w=90)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(80, 7, "Property", border=1)
        pdf.cell(60, 7, "Value", border=1, ln=1)
        pdf.set_font("Helvetica", "", 10)
        for k, v in properties(mol).items():
            pdf.cell(80, 6, ascii_safe(k), border=1)
            pdf.cell(60, 6, ascii_safe(v), border=1, ln=1)
    if pdf.page == 0:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, "No valid molecules.")
    return bytes(pdf.output())


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("🧪 Virtual Lab")
    st.caption("Navigation")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.subheader("🎓 Student Information")
    st.session_state["student_name"] = st.text_input("Student Name", st.session_state.get("student_name", ""))
    st.session_state["reg_no"] = st.text_input("Registration Number", st.session_state.get("reg_no", ""))
    st.divider()
    st.subheader("🎯 Learning Objectives")
    st.markdown(
        "Students can learn:\n"
        "- SMILES notation\n- Molecular representation\n- 2D & 3D visualization\n"
        "- Molecular descriptors\n- Structure–property relationships"
    )

if "smiles" not in st.session_state:
    st.session_state["smiles"] = "CCO"

# ---------------- pages ----------------
if page == PAGES[0]:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=320)
    st.title("🧪 Cheminformatics Virtual Laboratory")
    st.subheader(COLLEGE)
    st.write(DEPARTMENT)
    st.divider()
    st.header("Welcome to the Virtual Laboratory")
    st.markdown(
        "In this laboratory you can:\n\n"
        "✅ Learn SMILES notation  \n"
        "✅ Visualize molecules in 2D and 3D  \n"
        "✅ Calculate complete molecular properties  \n"
        "✅ Download molecular reports  \n"
        "✅ Compare multiple molecules  \n"
        "✅ Complete an assessment"
    )
    st.subheader("Example Molecules")
    st.table(pd.DataFrame({"Molecule": list(EXAMPLES), "SMILES": list(EXAMPLES.values())}))
    st.divider()
    st.subheader("👨‍🏫 Developed by")
    st.markdown(f"**{DEVELOPER}** — {DEVELOPER_ROLE}  \nUnder the guidance of **{GUIDE}**, {GUIDE_ROLE}")
    st.caption(f"[Source code on GitHub]({GITHUB_URL})")

elif page == PAGES[1]:
    st.title("📖 Theory")
    st.header("What is Cheminformatics?")
    st.write("Cheminformatics combines:")
    st.markdown("- Chemistry 🧪\n- Computer Science 💻\n- Data Analysis 📊")
    st.write("It uses computational techniques to store, process, visualize and analyze chemical information.")
    st.header("SMILES")
    st.write("**SMILES** means **Simplified Molecular Input Line Entry System**. "
             "It represents a molecular structure using a text string.")
    st.table(pd.DataFrame({"Molecule": ["Ethanol", "Benzene", "Acetic Acid"],
                           "SMILES": ["CCO", "c1ccccc1", "CC(=O)O"]}))
    st.subheader("Basic rules")
    st.markdown(
        "- Atoms are written by their symbols: `C`, `N`, `O`\n"
        "- `=` is a double bond and `#` is a triple bond\n"
        "- `( )` shows a branch\n"
        "- Numbers (`1`) mark ring closures\n"
        "- Lowercase letters (`c`) mean aromatic atoms"
    )

elif page == PAGES[2]:
    st.title("🧬 Molecular Visualization Laboratory")
    st.info("Enter a new SMILES or load an example. The selected molecule is used across the laboratory pages.")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.session_state["smiles"] = st.text_input("Enter SMILES", st.session_state["smiles"])
    with col_b:
        ex = st.selectbox("Choose Example Molecule", list(EXAMPLES))
        if st.button("Load Example"):
            st.session_state["smiles"] = EXAMPLES[ex]
            st.rerun()
    mol = get_mol(st.session_state["smiles"])
    if mol is None:
        st.error("Invalid SMILES. Please check and try again.")
    else:
        st.success("Valid Molecular Structure!")
        st.subheader("2D Molecular Structure")
        st.image(draw_2d(mol))
        st.subheader("Molecular Information")
        p = properties(mol)
        st.write(f"**Input SMILES:** {st.session_state['smiles']}")
        st.write(f"**Canonical SMILES:** {Chem.MolToSmiles(mol)}")
        for k in ["Molecular Formula", "Heavy Atoms", "Atoms (incl. H)",
                  "Heavy-Atom Bonds", "Sigma (σ) Bonds", "Pi (π) Bonds", "Total Bonds (σ + π)"]:
            st.write(f"**{k}:** {p[k]}")
        st.subheader("🌐 Interactive 3D Molecular Structure")
        style = st.selectbox("Select Molecular Model",
                             ["Ball and Stick", "Stick", "Space Filling", "Wireframe", "Cartoon"])
        show_3d(mol, style)

elif page == PAGES[3]:
    st.title("📊 Molecular Properties Report")
    st.session_state["smiles"] = st.text_input("Enter SMILES", st.session_state["smiles"])
    mol = get_mol(st.session_state["smiles"])
    if mol is None:
        st.error("Invalid SMILES.")
    else:
        p = properties(mol)
        st.header("📊 Important Molecular Properties")
        c1, c2, c3 = st.columns(3)
        c1.metric("Molecular Weight", f"{p['Molecular Weight (g/mol)']} g/mol")
        c2.metric("LogP", p["LogP"])
        c3.metric("TPSA", f"{p['TPSA (Å²)']} Å²")
        c1.metric("Ring Count", p["Ring Count"])
        c2.metric("H-Bond Donors", p["H-Bond Donors"])
        c3.metric("H-Bond Acceptors", p["H-Bond Acceptors"])
        c1.metric("Rotatable Bonds", p["Rotatable Bonds"])
        c2.metric("Fraction Csp3", p["Fraction Csp3"])
        st.divider()
        st.subheader("📑 Complete Molecular Property Table")
        df = pd.DataFrame({"Property": list(p), "Value": [str(v) for v in p.values()]})
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download as CSV", df.to_csv(index=False), "properties.csv", "text/csv")

elif page == PAGES[4]:
    st.title("📈 Structure–Property Analysis")
    st.write("Enter one SMILES per line to compare molecules.")
    default = "\n".join(EXAMPLES.values())
    text = st.text_area("SMILES list", default, height=160)
    rows = []
    for smi in [s.strip() for s in text.splitlines() if s.strip()]:
        m = get_mol(smi)
        if m is None:
            st.warning(f"Invalid SMILES skipped: {smi}")
            continue
        p = properties(m)
        rows.append({"SMILES": smi, "Formula": p["Molecular Formula"], "MW": p["Molecular Weight (g/mol)"],
                     "LogP": p["LogP"], "TPSA": p["TPSA (Å²)"], "HBD": p["H-Bond Donors"],
                     "HBA": p["H-Bond Acceptors"], "RotB": p["Rotatable Bonds"], "Rings": p["Ring Count"]})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader("Molecular Weight comparison")
        st.bar_chart(df.set_index("SMILES")["MW"])
        st.subheader("LogP vs TPSA")
        st.scatter_chart(df, x="LogP", y="TPSA")

elif page == PAGES[5]:
    st.title("📝 Cheminformatics Assessment")
    answers = []
    for i, (q, opts, _) in enumerate(QUIZ):
        answers.append(st.radio(f"{i + 1}. {q}", opts, index=None, key=f"q{i}"))
    if st.button("Submit Assessment"):
        if any(a is None for a in answers):
            st.warning("Please answer all questions.")
        else:
            score = sum(1 for a, (_, opts, c) in zip(answers, QUIZ) if a == opts[c])
            st.success(f"Score: {score} / {len(QUIZ)}")
            for i, (a, (q, opts, c)) in enumerate(zip(answers, QUIZ)):
                if a != opts[c]:
                    st.write(f"❌ Q{i + 1}: correct answer is **{opts[c]}**")

else:
    st.title("📄 Final Project Report")
    st.write("Enter one SMILES string per line. The report will include every valid molecule in a single downloadable PDF.")
    name = st.text_input("Student Name for Report", st.session_state.get("student_name", ""))
    reg = st.text_input("Registration Number for Report", st.session_state.get("reg_no", ""))
    text = st.text_area("Enter SMILES (one molecule per line)", "CCO", height=140)
    smiles_list = [s.strip() for s in text.splitlines() if s.strip()]
    valid = [s for s in smiles_list if get_mol(s) is not None]
    invalid = [s for s in smiles_list if s not in valid]
    if valid:
        st.success(f"{len(valid)} valid molecule(s) ready for the report.")
    for s in invalid:
        st.error(f"Invalid SMILES: {s}")
    if valid:
        st.dataframe(pd.DataFrame([{"No.": i + 1, "Input SMILES": s,
                                    "Molecular Formula": properties(get_mol(s))["Molecular Formula"],
                                    "Molecular Weight (g/mol)": properties(get_mol(s))["Molecular Weight (g/mol)"]}
                                   for i, s in enumerate(valid)]),
                     hide_index=True, use_container_width=True)
        if st.button("📄 Generate Single PDF for All Molecules"):
            st.session_state["pdf"] = build_pdf(name, reg, valid)
        if st.session_state.get("pdf"):
            st.download_button("⬇️ Download PDF Report", st.session_state["pdf"],
                               "cheminformatics_report.pdf", "application/pdf")

st.divider()
st.caption(f"🧪 Cheminformatics Virtual Laboratory | Developed using Python, Streamlit and RDKit | {COLLEGE}")
