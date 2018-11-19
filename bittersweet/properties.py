from pychem.pychem import Chem, constitution, connectivity, kappa, bcut,\
    moran, geary, molproperty, charge, moe, estate, basak
from .read_file import read_smiles_file

def generate_chemopy_props(mol):
    props = dict()

    try:
        props.update(constitution.GetConstitutional(mol))
        props.update(connectivity.GetConnectivity(mol))
        props.update(kappa.GetKappa(mol))
        props.update(bcut.GetBurden(mol))
        props.update(estate.GetEstate(mol))
        props.update(basak.Getbasak(mol))
        props.update(moran.GetMoranAuto(mol))
        props.update(geary.GetGearyAuto(mol))
        props.update(molproperty.GetMolecularProperty(mol))
        props.update(charge.GetCharge(mol))
        props.update(moe.GetMOE(mol))
    except:
        raise Exception("Properties could not be generated.")

    return props


def get_chemopy_props_from_smilesfile(f):
    smilesf = read_smiles_file(f)
    properties = list()

    try:
        for i, row in smilesf.iterrows():
            mol = Chem.MolFromSmiles(row[1])
            props = {'name': row[0]}
            
            try:
                props.update(generate_chemopy_props(mol))
            except:
                pass
                
            properties.append(props)
    except KeyError:
        raise Exception("Please ensure that the input data is in the correct format.")

    return properties
