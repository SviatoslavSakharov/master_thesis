from analysis.metrics import MoleculeProperties
from pathlib import Path
import os
import numpy as np
import pandas as pd
import ast
import py3Dmol
from rdkit import Chem
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def check_qvina_mols(qvina_scores, processed_mol):
    assert len(qvina_scores) == len(processed_mol), f"len(qvina_scores) = {len(qvina_scores)}, len(processed_mol) = {len(processed_mol)}"
    for i, (scores, mols) in enumerate(zip(qvina_scores, processed_mol)):
        assert len(scores) == len(mols), f"len(scores) = {len(scores)}, len(mols) = {len(mols)}, i = {i}"

def evaluate_folder(args):
    basedir, i, top10 = args
    processed_dir = basedir / "processed"

    processed_mol = []
    file_names = []
    for file in os.listdir(processed_dir):
        mols = Chem.SDMolSupplier(str(processed_dir / file))
        # processed_mol.append([mol for mol in mols if mol is not None])
        processed_mol.append([mol for mol in mols])
        file_names.append(file)


    processed_mol_filt = []
    if top10:
        qvina_scores = read_qvina_scores(basedir)
        check_qvina_mols(qvina_scores, processed_mol)
        file = basedir / "evaluation_top10.txt"
        sorted_idx = [np.argsort(px)[:10] for px in qvina_scores]
        # print(f"sorted_idx: {sorted_idx[0]}")
        for i, mols in enumerate(processed_mol):
            processed_mol_filt.append([mols[idx] for idx in sorted_idx[i] if mols[idx] is not None])
        # Check for each pocket if there are more than 5 molecules
        for i, mols in enumerate(processed_mol_filt):
            assert len(mols) > 5, f"len(mols) = {len(mols)}, i = {i}"
    else:
        file = basedir / "evaluation.txt"
        processed_mol_filt = processed_mol

        
    mol_metrics = MoleculeProperties()
    all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = \
        mol_metrics.evaluate(processed_mol_filt, position=i, desc=basedir.name)

    if not top10:
        ### save metrics to csv file
        for qed in all_qed:
            assert len(qed) == 100
        metrics_new = {"file_names": file_names, "QED": all_qed, "SA": all_sa, "lipinski": all_lipinski}
        df = pd.DataFrame(metrics_new)
        df.to_csv(basedir / "metrics.csv", na_rep='NULL', index=False)

    
    all_qed = [x for px in all_qed for x in px if x is not None]
    all_sa = [x for px in all_sa for x in px if x is not None]
    all_logp = [x for px in all_logp for x in px if x is not None]
    all_lipinski = [x for px in all_lipinski for x in px if x is not None]

    # ### write to file
    ### read pocket_times
    pocket_times = []
    with open(basedir / "pocket_times.txt") as f:
        for line in f:
            pocket_times.append(float(line.split(" ")[1][:-2]))
    
    # ### delete if already exists
    if os.path.exists(file):
        os.remove(file)

    with open(file, "w") as f:
        f.write(f"{sum([len(p) for p in processed_mol_filt])} molecules from {len(processed_mol_filt)} pockets evaluated. \n")
        f.write(f"QED: {np.mean(all_qed):.2f} +- {np.std(all_qed):.2f}\n")
        f.write(f"SA: {np.mean(all_sa):.2f} +- {np.std(all_sa):.2f}\n")
        # f.write(f"logP: {np.mean(all_logp):.2f} +- {np.std(all_logp):.2f}\n")
        f.write(f"lipinski: {np.mean(all_lipinski):.2f} +- {np.std(all_lipinski):.2f}\n")
        f.write(f"Diversity: {np.mean(per_pocket_diversity):.2f} +- {np.std(per_pocket_diversity):.2f}\n")
        f.write(f"pocket times: {np.mean(pocket_times):.2f} +- {np.std(pocket_times):.2f}\n")


def read_qvina_scores(basedir):
    qvina_file = basedir / "qvina" / "qvina2_scores.csv"
    if not qvina_file.exists():
        return None
    df = pd.read_csv(qvina_file, index_col=0)
    # df["scores"] = df["scores"].apply(lambda x: np.array(ast.literal_eval(x.replace("nan, ", "").replace(", nan]", "]"))))
    df["scores"] = df["scores"].apply(lambda x: np.array(ast.literal_eval(x.replace("nan", "100"))))
    scores = df["scores"].values

    return scores

def calculate_qvina_score(basedir):
    print(f"Calculating QVina2 score for {basedir.name}")
    scores = read_qvina_scores(basedir)
    if scores is None:
        return None

    all_scores = [x for px in scores for x in px]
    all_scores = [x for x in all_scores if x != 100]
    sorted_scores = [np.sort(px)[:10] for px in scores]
    top10_all_scores = [x for px in sorted_scores for x in px]

    file = basedir / "evaluation_qvina.txt"
    ### delete if already exists
    if os.path.exists(file):
        os.remove(file)
    with open(file, "w") as f:
        f.write(f"QVina2: {np.mean(all_scores):.2f} +- {np.std(all_scores):.2f}\n")
        f.write(f"QVina2 top10: {np.mean(top10_all_scores):.2f} +- {np.std(top10_all_scores):.2f}\n")
        
        
        

if __name__ == "__main__":
    parent = Path("/home/domainHomes/ssakharov/master_thesis/crossdocked/processed_crossdock_noH_ca_only_temp")
    dirs = [
            # parent / "predictions_r1_t500_ddim10_n100", 
            # parent / "predictions_r1_t500_ddim20_n100",
            # parent / "predictions_r1_t500_ddim50_n100",
            # parent / "predictions_r1_t500_ddim2_n100",
            # parent / "predictions_r1_t500_ddim5_n100",
            # parent / "predictions_r1_t500",
            # parent / "predictions_r10_t50",
            # parent / "predictions_r1_t500_ddim5_n100_quad", 
            # parent / "predictions_r1_t500_ddim10_n100_quad",
            # parent / "predictions_r1_t500_ddim20_n100_quad",
            # parent / "predictions_r1_t500_ddim50_n100_quad",
            # parent / "predictions_r1_t500_n100_second",
            parent / "predictions_r1_t500_ddim_50_nu_0_n100_quad",
            parent / "predictions_r1_t500_ddim_100_nu_0_n100_quad",
            parent / "predictions_r1_t500_ddim_250_nu_0_n100_quad",
            parent / "predictions_r1_t500_ddim_400_nu_0_n100_quad",
    ]
    with Pool(processes=len(dirs)) as pool:
        args = ((_dir, i + 1, False) for i, _dir in enumerate(dirs))
        for _ in tqdm(pool.imap_unordered(evaluate_folder, args), total=len(dirs), position=0, desc='Main Progress'):
            pass
    # for _dir in dirs:
    #     calculate_qvina_score(_dir)
        # evaluate_folder((_dir, 1, False))
        # break

