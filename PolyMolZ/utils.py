import string
import re
from warnings import warn
from rdkit import Chem


def sanitizeSQL(s):
    if re.match(r"[^A-Za-z0-9_.]", s):
        warn("Unallowed characters detected in SQL table name.\n")
    return re.sub(r"[^A-Za-z0-9_.]", "", s)


def brushTeeth(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
