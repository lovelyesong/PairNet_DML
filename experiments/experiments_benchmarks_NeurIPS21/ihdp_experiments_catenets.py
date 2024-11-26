"""
Utils to replicate IHDP experiments with catenets
"""
import csv
import os
from pathlib import Path
from typing import Optional, Union
from catenets.models.torch import representation_nets as torch_nets
import copy
import torch
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn import clone

from catenets.datasets.dataset_ihdp import (
    get_one_data_set,
    load_raw,
    prepare_ihdp_pairnet_data,
)
from catenets.datasets.torch_dataset import (
    BaseTorchDataset as TorchDS,
)
from catenets.experiment_utils.base import eval_root_mse

from catenets.models.jax import (
    RNET_NAME,
    T_NAME,
    TARNET_NAME,
    CFRNET_NAME,
    PAIRNET_NAME,
    XNET_NAME,
    DRAGON_NAME,
    FLEXTE_NAME,
    DRNET_NAME,
    RNet,
    TARNet,
    CFRNet,
    PairNet,
    FlexTENet,
    DragonNet,
    DRNet,
    TNet,
    XNet,
)

DATA_DIR = Path("catenets/datasets/data/")
RESULT_DIR = Path("results/experiments_benchmarking/ihdp/")

PENALTY_DIFF = 0.01
PENALTY_ORTHOGONAL = 0.1

repr_dir = {
    TARNET_NAME: RESULT_DIR / TARNET_NAME,
}
for v in repr_dir.values():
    if not os.path.isdir(v):
        os.makedirs(v)

SEP = "_"

PARAMS_DEPTH = {"n_layers_r": 3, "n_layers_out": 2}
PARAMS_DEPTH_2 = {
    "n_layers_r": 3,
    "n_layers_out": 2,
    "n_layers_r_t": 3,
    "n_layers_out_t": 2,
}

model_hypers = {
    CFRNET_NAME: {"penalty_disc": 0.1},
    PAIRNET_NAME: {
        "penalty_disc": 0.0,
        "penalty_l2": 1.0,
    },
}

pair_data_args = {
    "det": False,
    "num_cfz": 3,
    "sm_temp": 1.0,
    "dist": "euc",  # cos/euc
    "pcs_dist": True,  # Process distances
    "drop_frac": 0.1,  # distance threshold
    "arbitrary_pairs": False,
    "OT": False,
}


def dict_to_str(dict):
    return SEP.join([f"--{k}{SEP}{v}" for k, v in dict.items()])


ALL_MODELS = {
    # T_NAME: TNet(**PARAMS_DEPTH),
    TARNET_NAME: TARNet(**PARAMS_DEPTH),
    # CFRNET_NAME: CFRNet(**PARAMS_DEPTH),
    PAIRNET_NAME: PairNet(**PARAMS_DEPTH),
    # RNET_NAME: RNet(**PARAMS_DEPTH_2),
    # XNET_NAME: XNet(**PARAMS_DEPTH_2),
    # FLEXTE_NAME: FlexTENet(
    #     penalty_orthogonal=PENALTY_ORTHOGONAL, penalty_l2_p=PENALTY_DIFF, **PARAMS_DEPTH
    # ),
    # DRNET_NAME: DRNet(first_stage_strategy="Tar", **PARAMS_DEPTH_2),
    # DRAGON_NAME: DragonNet(**PARAMS_DEPTH),
}


def do_ihdp_experiments(
    n_exp: Union[int, list] = 100,
    n_reps: int = 1,
    file_name: str = "ihdp_all",
    model_params: Optional[dict] = None,
    models: Optional[dict] = None,
    setting: str = "original",
    save_reps: bool = False,
) -> None:
    if models is None:
        models = ALL_MODELS
        # models = {PAIRNET_NAME: PairNet(**PARAMS_DEPTH)}

    if (setting == "original") or (setting == "C"):
        setting = "C"
    elif (setting == "modified") or (setting == "D"):
        setting = "D"
    else:
        raise ValueError(
            f"Setting should be one of original or modified. You passed {setting}."
        )

    # get file to write in
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # get data
    data_train, data_test = load_raw(DATA_DIR)
    
    out_file = open(RESULT_DIR / f"{file_name}.csv", "w", buffering=1)
    print(f"saving results to {out_file}")
    writer = csv.writer(out_file)
    header = (
        ["exp", "cate_var_in", "cate_var_out", "y_var_in"]
        + [name + "_in" for name in models.keys()]
        + [name + "_out" for name in models.keys()]
    )
    writer.writerow(header)

    if isinstance(n_exp, int):
        experiment_loop = list(range(1, n_exp + 1))
    elif isinstance(n_exp, list):
        experiment_loop = n_exp
    else:
        raise ValueError(
            "n_exp should be either an integer or a list of integers."
        )
    for i_exp in experiment_loop:
        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(
            data_test, i_exp=i_exp, get_po=True
        )

        # NOTE: If setting is D, tau is changed to be additive in the potential outcomes. Not the setting of interest in our paper.
        data_dict, ads_train = prepare_ihdp_pairnet_data(
            i_exp=i_exp,
            model_name=TARNET_NAME,
            data_train=data_exp,
            data_test=data_exp_test,
            setting=setting,
            **pair_data_args,
        )

        X, y, w, cate_true_in, X_t, cate_true_out = (
            data_dict["X"],
            data_dict["y"],
            data_dict["w"],
            data_dict["cate_true_in"],
            data_dict["X_t"],
            data_dict["cate_true_out"],
        )

        # compute some stats
        cate_var_in = np.var(cate_true_in)
        cate_var_out = np.var(cate_true_out)
        y_var_in = np.var(y)

        pehe_in = []
        pehe_out = []

        for model_name, estimator in models.items():
            print(f"model name : {model_name}")
            if model_name == PAIRNET_NAME:
                data_dict, ads_train = prepare_ihdp_pairnet_data(
                    i_exp=i_exp,
                    model_name=PAIRNET_NAME,
                    data_train=data_exp,
                    data_test=data_exp_test,
                    setting=setting,
                    **pair_data_args,
                )

                X, y, w, cate_true_in, X_t, cate_true_out, w_t, y_t, mu0_t, mu1_t = (
                    data_dict["X"],
                    data_dict["y"],
                    data_dict["w"],
                    data_dict["cate_true_in"],
                    data_dict["X_t"],
                    data_dict["cate_true_out"],
                    data_dict["w_t"],
                    data_dict["y_t"],
                    data_dict["mu0_t"],
                    data_dict["mu1_t"],
                ) # mu0_t, mu1_t : where are these from?

            try:
                print(f"Experiment {i_exp}, with {model_name}")
                estimator_temp = clone(estimator)
                estimator_temp.set_params(seed=0)
                if model_name in model_hypers.keys():
                    if model_params is None:
                        model_params = {}
                    model_params.update(model_hypers[model_name])

                if model_params is not None:
                    estimator_temp.set_params(**model_params)

                if model_name in model_hypers.keys():
                    # Delete the keys from the model_params dictionary
                    for key in model_hypers[model_name].keys():
                        del model_params[key]

                # fit estimator
                if model_name in [PAIRNET_NAME]:
                    print('test : pre fit')
                    estimator_temp.agree_fit(ads_train)

                    # estimator mu0_hat, mu1_hat 검증을 위하여
                    # 만약 return_po
                    cate_pred_in, mu0_hat_tr, mu1_hat_tr = estimator_temp.predict(X, return_po=True)
                    # cate_pred_out, mu0_hat_tst, mu1_hat_tst = estimator_temp.predict(X_t, return_po=True)

                    #### [DML estimation] ####
                    # estimator_tmp : outcom regression과 covariate representation (phi)를 위한 학습된함수
                    cate_pred_in, mu0_hat, mu1_hat = estimator_temp.predict(X_t, return_po=True)
                    phi_representation = estimator_temp.getrepr(X_t)  # PairNet에서 학습된 Representation

                    # g_hat (E[Y|X])
                    # g_hat = (1 - w) * mu0 + w * mu1

                    # Propensity Score 모델에 phi_representation 사용
                    propensity_model = LogisticRegression()
                    # propensity_model = XGBRegressor()
                    w_flat = w_t.ravel() if len(w_t.shape) > 1 else w_t
                    propensity_model.fit(phi_representation, w_flat)  # Representation 기반 학습
                    e_hat = propensity_model.predict_proba(phi_representation)[:, 1]  # Propensity score 추정

                    # DML 추정 수행
                    # theta_hat = double_ml(phi_representation, y_t, w_t, mu0_hat, mu1_hat, e_hat)
                    # print(f"[test] DML Causal Effect Estimate for Experiment {i_exp} : {theta_hat}")
                    w_t_flat = w_t.flatten()
                    idx_w0 = np.where(w_t_flat == 0)[0]
                    idx_w1 = np.where(w_t_flat == 1)[0]

                    # w_t == 0
                    phi_representation_w0 = phi_representation[idx_w0]
                    y_t_w0 = y_t[idx_w0]
                    mu0_hat_w0 = mu0_hat[idx_w0]
                    mu1_hat_w0 = mu1_hat[idx_w0]
                    e_hat_w0 = e_hat[idx_w0]
                    w_t_w0 = w_t[idx_w0]

                    # w_t == 1
                    phi_representation_w1 = phi_representation[idx_w1]
                    y_t_w1 = y_t[idx_w1]
                    mu1_hat_w1 = mu1_hat[idx_w1]
                    mu0_hat_w1 = mu0_hat[idx_w1]
                    e_hat_w1 = e_hat[idx_w1]
                    w_t_w1 = w_t[idx_w1]

                    # dml estimator
                    tau1_hat = double_ml(phi_representation_w1, y_t_w1, w_t_w1, mu0_hat_w1, mu1_hat_w1, e_hat_w1)
                    tau0_hat = double_ml(phi_representation_w0, y_t_w0, w_t_w0, mu0_hat_w0, mu1_hat_w0, e_hat_w0)

                    # WAAE
                    flat_arr = w_t.flatten()
                    count_0 = np.sum(flat_arr == 0)
                    count_1 = np.sum(flat_arr == 1)
                    ratio_D0 = count_0 / len(w_t)
                    ratio_D1 = count_1 / len(w_t)

                    waae = (np.abs(tau0_hat.mean() - mu0_t.mean()) * ratio_D0) + (np.abs(tau1_hat.mean() - mu1_t.mean()) * ratio_D1)
                    print(f"WAAE : {waae}")

                    print('!!test!!')

                else:
                    estimator_temp.fit(X=X, y=y, w=w)

                if model_name in [CFRNET_NAME, TARNET_NAME]:
                    cate_pred_in, mu0_tr, mu1_tr = estimator_temp.predict(
                        X, return_po=True
                    )
                    cate_pred_out, mu0_te, mu1_te = estimator_temp.predict(
                        X_t, return_po=True
                    )
                    if save_reps:
                        dump_reps(
                            setting,
                            model_name,
                            i_exp,
                            X,
                            X_t,
                            estimator_temp,
                            mu0_tr,
                            mu1_tr,
                            mu0_te,
                            mu1_te,
                        )
                else:
                    # idhp 데이터의 prediction
                    cate_pred_in = estimator_temp.predict(X)
                    cate_pred_out = estimator_temp.predict(X_t)

                if isinstance(cate_pred_in, torch.Tensor):
                    cate_pred_in = cate_pred_in.detach().numpy()
                if isinstance(cate_pred_out, torch.Tensor):
                    cate_pred_out = cate_pred_out.detach().numpy()

                pehe_in.append(eval_root_mse(cate_pred_in, cate_true_in))
                pehe_out.append(eval_root_mse(cate_pred_out, cate_true_out))
            except:
                print(
                    f"Experiment {i_exp}, with {model_name} failed"
                )
                pehe_in.append(-1)
                pehe_out.append(-1)

        writer.writerow(
            [i_exp, cate_var_in, cate_var_out, y_var_in]
            + pehe_in
            + pehe_out
        )
    out_file.close()


def dump_reps(
    setting, model_name, i_exp, X, X_t, estimator_temp, mu0_tr, mu1_tr, mu0_te, mu1_te
):
    trn_reps = estimator_temp.getrepr(X)
    tst_reps = estimator_temp.getrepr(X_t)

    # concatenate mu0, mu1 to trn_reps
    trn_reps = np.concatenate([trn_reps, mu0_tr, mu1_tr], axis=1)
    tst_reps = np.concatenate([tst_reps, mu0_te, mu1_te], axis=1)

    # Save representations
    np.save(
        repr_dir[model_name] / f"ihdp-{setting}-{i_exp}-trn.npy",
        trn_reps,
    )
    np.save(
        repr_dir[model_name] / f"ihdp-{setting}-{i_exp}-tst.npy",
        tst_reps,
    )


def double_ml(X, y, w, mu0, mu1, e_hat):
    """
    Perform Double Machine Learning (DML) estimation using precomputed g_hat and e_hat.

    Parameters:
    - X: ndarray, covariates (not used but kept for compatibility)
    - y: ndarray, observed outcomes
    - w: ndarray, treatment indicators
    - g_hat: ndarray, predicted E[Y|X]
    - e_hat: ndarray, predicted P(D=1|X)

    Returns:
    - theta_hat: float, treatment effect estimate
    """
    # # Compute doubly robust estimator
    # dr_terms = (w * (y - g_hat) / e_hat) + g_hat
    # theta_hat = dr_terms.mean()

    # IPW terms
    ipw_treated = (w * y) / e_hat
    ipw_control = ((1 - w) * y) / (1 - e_hat)

    # Regression terms
    regression_term = mu1 - mu0

    # Combine components
    dr_estimator =  ipw_treated - ipw_control + regression_term

    # Average over all samples
    ate = dr_estimator.mean()
    return ate
