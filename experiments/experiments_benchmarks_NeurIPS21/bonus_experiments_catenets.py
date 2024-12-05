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
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBRegressor
from sklearn.model_selection import KFold

import numpy as np
from sklearn import clone
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from catenets.datasets.dataset_bonus import (
    get_one_data_set,
    load_raw,
    prepare_bonus_pairnet_data,
)
from catenets.datasets.torch_dataset import (
    BaseTorchDataset as TorchDS,
)
from catenets.experiment_utils.base import eval_root_mse

from catenets.models.jax import (
    # RNET_NAME,
    # T_NAME,
    TARNET_NAME,
    # CFRNET_NAME,
    PAIRNET_NAME,
    # XNET_NAME,
    # DRAGON_NAME,
    # FLEXTE_NAME,
    # DRNET_NAME,
    # RNet,
    TARNet,
    # CFRNet,
    PairNet,
    # FlexTENet,
    # DragonNet,
    # DRNet,
    # TNet,
    # XNet,
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

PARAMS_DEPTH = {"n_layers_r": 2, #3
                "n_layers_out": 1} #2
PARAMS_DEPTH_2 = {
    "n_layers_r": 3,
    "n_layers_out": 2,
    "n_layers_r_t": 3,
    "n_layers_out_t": 2,
}

model_hypers = {
    # CFRNET_NAME: {"penalty_disc": 0.1},
    PAIRNET_NAME: {
        "penalty_disc": 0.0,
        "penalty_l2": 1, # 1.0
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

class MuModel(nn.Module):
    def __init__(self, input_dim):
        super(MuModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
    def forward(self, X):
        return self.net(X)



def do_bonus_experiments(
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
    data_train, data_test = load_raw(DATA_DIR) #DATA_DIR = Path("catenets/datasets/data/")

    # print('test\n', data_train)

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

    ate_pairnet_dml_list = [] # for bootstrap
    ate_pairnet_list = [] # for bootstrap
    ate_pairnet_mumodel_list = [] # for bootstrap
    ate_pairnet_dml_mumodel_list = [] # for bootstrap
    waae_list = []

    for i_exp in experiment_loop:
        # get data
        data_exp = get_one_data_set(data_train, i_exp=i_exp, get_po=True)
        data_exp_test = get_one_data_set(
            data_test, i_exp=i_exp, get_po=True
        )

        # NOTE: If setting is D, tau is changed to be additive in the potential outcomes. Not the setting of interest in our paper.
        data_dict, ads_train = prepare_bonus_pairnet_data(
            i_exp=i_exp,
            model_name=TARNET_NAME,
            data_train=data_exp,
            data_test=data_exp_test,
            setting=setting,
            **pair_data_args,
        )


        X, y, w, cate_true_in, X_t, cate_true_out, y_t = (
            data_dict["X"],
            data_dict["y"],
            data_dict["w"],
            data_dict["cate_true_in"],
            data_dict["X_t"],
            data_dict["cate_true_out"],
            data_dict["y_t"],
        )

        # compute some stats
        cate_var_in = np.var(cate_true_in)
        cate_var_out = np.var(cate_true_out)
        y_var_in = np.var(y)

        pehe_in = []
        pehe_out = []

        pehe_pairdml = []

        for model_name, estimator in models.items():
            if model_name == PAIRNET_NAME:
                data_dict, ads_train = prepare_bonus_pairnet_data(
                    i_exp=i_exp,
                    model_name=PAIRNET_NAME,
                    data_train=data_exp,
                    data_test=data_exp_test,
                    setting=setting,
                    **pair_data_args,
                )

                # X, y, w, cate_true_in, X_t, w_t, cate_true_out, y_t, mu0_t, mu1_t = (
                #     data_dict["X"],
                #     data_dict["y"],
                #     data_dict["w"],
                #     data_dict["cate_true_in"],
                #     data_dict["X_t"],
                #     data_dict["w_t"],
                #     data_dict["cate_true_out"],
                #     data_dict["y_t"],
                #     data_dict["mu0_t"],
                #     data_dict["mu1_t"],
                # )

                X, y, w, cate_true_in, X_t, cate_true_out, w_t, y_t, mu0_t, mu1_t = (
                    data_dict["X"],
                    data_dict["y"],
                    data_dict["w"],
                    data_dict["cate_true_in"],
                    data_dict["X_t"],
                    data_dict["cate_true_out"],
                    data_dict["w_t"], #
                    data_dict["y_t"], #
                    data_dict["mu0_t"], #
                    data_dict["mu1_t"], #
                ) # y_t, mu0_t, mu1_t 추가 추출

            try:
                print(f"Experiment {i_exp}, with {model_name}")
                estimator_temp = clone(estimator) # 먼저 tarnet학습
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
                    estimator_temp.agree_fit(ads_train) # Base_catenet.agree_fit -> Base_catenet.train_func -> PairNet.predict_pairnet

                    #### [DML estimation] ####
                    # estimator_tmp : outcom regression과 covariate representation (phi)를 위한 학습된함수
                    cate_pred_out, mu0_hat, mu1_hat = estimator_temp.predict(X_t, return_po=True)
                    phi_representation = estimator_temp.getrepr(X_t)  # PairNet에서 학습된 Representation

                    # print(f" X_t size : {X_t.shape}")

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


                    ################ CATE residual Neural Net #################
                    mu1_final, mu0_final, mu1_, mu0_ = cate_residualNet(phi_representation,
                                                                        y_t_w1,
                                                                        y_t_w0,
                                                                        cate_pred_out,
                                                                        idx_w0,
                                                                        idx_w1)
                    ############################################################

                    # residual calibration : mu_adjusted
                    # res_cali_mu1 = res_calibration(phi_representation_w1, y_t_w1, mu1_hat_w1) # for treatment
                    # res_cali_mu0 = res_calibration(phi_representation_w0, y_t_w0, mu1_hat_w0) # for control

                    #print(f"res_cali_mu1.mean() : {res_cali_mu1.mean()}, res_cali_mu0.mean() : {res_cali_mu0.mean()} ")


                    # representation learning calibration
                    # repr_cali_mu1 = repr_calibration(phi_representation_w1, y_t_w1, mu1_hat_w1) #(1013,1013)
                    # repr_cali_mu0 = repr_calibration(phi_representation_w0, y_t_w0, mu1_hat_w0) #(517,517)

                    #print(f"repr_cali_mu1.mean() : {repr_cali_mu1.mean()}, repr_cali_mu0.mean() : {repr_cali_mu0.mean()} ")

                    # dml estimator
                    # tau1_hat = double_ml(phi_representation_w1, y_t_w1, w_t_w1, mu0_hat_w1, mu1_hat_w1, e_hat_w1)
                    # tau0_hat = double_ml(phi_representation_w0, y_t_w0, w_t_w0, mu0_hat_w0, mu1_hat_w0, e_hat_w0)


                    #ATE
                    # 1. PairNet+DML ATE ATE : -0.04787540063261986
                    tau_hat =  double_ml(phi_representation, y_t, w_t_flat, mu0_hat, mu1_hat, e_hat)
                    #print(f"PairNet+DML ATE : {tau_hat}")
                    ate_pairnet_dml_list.append(tau_hat)

                    # 2. PairNet ATE : PairNet ATE : -0.17016984522342682
                    cate_pred_out_mean = cate_pred_out.mean()
                    #print(f"PairNet ATE : {cate_pred_out_mean}")
                    ate_pairnet_list.append(cate_pred_out_mean)

                    # 3. PairNet w/ Mu modeling ATE : PairNet w/ Mu modeling ATE : -0.09951705485582352
                    mu_model_ate = (mu1_ - mu0_).mean()
                    #print(f"PairNet w/ Mu modeling ATE : {mu_model_ate}")
                    ate_pairnet_mumodel_list.append(mu_model_ate)

                    # 4. PairNet+DML w/ Mu modeling ATE
                    tau_hat_mumodel = double_ml_tensor(phi_representation, y_t, w_t_flat, mu0_, mu1_, e_hat)
                    #print(f"PairNet+DML w/ Mu modeling ATE : {tau_hat_mumodel}")
                    ate_pairnet_dml_mumodel_list.append(tau_hat_mumodel)

                    # 1. dml estimator separately -> WAAE : 3.08741815876354
                    # tau1_hat = dml_sep(phi_representation_w1, y_t_w1, w_t_w1, mu1_hat_w1, e_hat_w1) # treatment
                    # tau0_hat = dml_sep(phi_representation_w0, y_t_w0, w_t_w0, mu0_hat_w0, e_hat_w0) # control

                    # 2. dml estimator separately with mu_adjusted (res_calibration) -> WAAE : 0.6830219045446648
                    # tau1_hat = dml_sep(phi_representation_w1, y_t_w1, w_t_w1, res_cali_mu1, e_hat_w1) # treatment
                    # tau0_hat = dml_sep(phi_representation_w0, y_t_w0, w_t_w0, res_cali_mu0, e_hat_w0) # control

                    # 3. dml estimator separately with representation calibration -> WAAE : 0.7141657512435007
                    #tau1_hat = dml_sep(phi_representation_w1, y_t_w1, w_t_w1, repr_cali_mu1, e_hat_w1) # treatment
                    #tau0_hat = dml_sep(phi_representation_w0, y_t_w0, w_t_w0, repr_cali_mu0, e_hat_w0) # control

                    # 4. dml estimator separately with MuModel learning -> WAAE : 2.5122521156700874 (100) / WAAE : 1.6746177559094473 (200) / WAAE : 0.6607857349114392 (600)
                    tau1_hat = dml_sep(phi_representation_w1, y_t_w1, w_t_w1, mu1_final, e_hat_w1) # treatment #517
                    tau0_hat = dml_sep(phi_representation_w0, y_t_w0, w_t_w0, mu0_final, e_hat_w0) # control #

                    # WAAE
                    flat_arr = w_t.flatten()
                    count_0 = np.sum(flat_arr == 0)
                    count_1 = np.sum(flat_arr == 1)
                    ratio_D0 = count_0 / len(w_t)
                    ratio_D1 = count_1 / len(w_t)

                    # print(f"w=1: {count_1}, w=0: {count_0}")

                    # PairNet-DML WAAE
                    waae = (np.abs(tau0_hat.mean() - mu0_t.mean()) * ratio_D0) + (np.abs(tau1_hat.mean() - mu1_t.mean()) * ratio_D1)
                    # print(f"WAAE : {waae}")
                    waae_list.append(waae)

                    # # PairNet WAAE : WAAE for PairNet: 1.8299227201860715
                    # pairnet_waae = (np.abs(mu0_hat_w0.mean() - mu0_t.mean()) * ratio_D0) + (np.abs(mu1_hat_w1.mean() - mu1_t.mean()) * ratio_D1)
                    # print(f"WAAE for PairNet: {pairnet_waae}")
                    #
                    # PairNet WAAE with MuModel : WAAE for PairNet: 0.6988605260848999
                    # pairnet_waae_mumodel = (np.abs(mu0_final.mean() - mu0_t.mean()) * ratio_D0) + (np.abs(mu1_final.mean() - mu1_t.mean()) * ratio_D1)
                    # print(f"WAAE for PairNet w/ MuModel: {pairnet_waae_mumodel}")

                    # PEHE
                    #print("PEHE : ", eval_root_mse(tau_hat, cate_true_out))
                    #pehe_pairdml.append(eval_root_mse(tau_hat, cate_true_out))


                else:
                    estimator_temp.fit(X=X, y=y, w=w) # tarnet fit

                if model_name in [TARNET_NAME]:
                    cate_pred_in, mu0_tr, mu1_tr = estimator_temp.predict(
                        X, return_po=True
                    )
                    cate_pred_out, mu0_te, mu1_te = estimator_temp.predict(
                        X_t, return_po=True
                    )
                    if save_reps:
                        dump_reps( # Tarnet 모델 저장
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
                import traceback
                # print(f"Experiment {i_exp}, with {model_name} failed due to: {e}")
                traceback.print_exc()  # 오류의 자세한 stack trace 출력
                pehe_in.append(-1)
                pehe_out.append(-1)

        writer.writerow(
            [i_exp, cate_var_in, cate_var_out, y_var_in]
            + pehe_in
            + pehe_out
            + pehe_pairdml
        )
    out_file.close()

    print(f"Bootstrapped PairNet+DML ATE : {np.array(ate_pairnet_dml_list).mean()}")  # for bootstrap
    print(f"Bootstrapped PairNet ATE : {np.array(ate_pairnet_list).mean()}")
    print(f"Bootstrapped PairNet w/ Mu modeling ATE : {np.array(ate_pairnet_mumodel_list).mean()}")
    print(f"Bootstrapped PairNet+DML w/ Mu modeling ATE : {np.array(ate_pairnet_dml_mumodel_list).mean()}")
    print(f"Bootstrapped PairNet+DML+Mumodel WAAE : {np.array(waae_list).mean()}")



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


def dml_sep(X, y, w, mu, e_hat):
    y = y.reshape(-1, 1)
    w = w.reshape(-1, 1)  # (1530, 1)
    e_hat = e_hat.reshape(-1, 1)  # (1530, 1)
    #mu = mu.reshape(-1, 1)  # Ensure proper shape
    mu = mu.numpy().reshape(-1, 1)

    #print("Shapes in dml_sep:")
    #print(f"y: {y.shape}, w: {w.shape}, mu: {mu.shape}, e_hat: {e_hat.shape}")

    # Calculate Doubly Robust Estimator
    dr = np.where(
        w == 1,  # For treatment group
        (w * (y - mu) / e_hat) + mu,
        # For control group
        ((1 - w) * (y - mu) / (1 - e_hat)) + mu
    )

    return dr


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

    w = w.reshape(-1, 1)  # (1530, 1)
    e_hat = e_hat.reshape(-1, 1)  # (1530, 1)

    # 아래 세개 딥러닝 넣었다 나온 파라미터 쓸 때 꼭 쓰기..tensor -> numpy
    # y = y.reshape(-1, 1)
    # mu0 = mu0.numpy().reshape(-1, 1)
    # mu1 = mu1.numpy().reshape(-1, 1)

    # IPW terms
    # ipw_treated = (w * y) / e_hat
    # ipw_control = ((1 - w) * y) / (1 - e_hat)

    ipw_treated = (w * (y - mu1)) / e_hat
    ipw_control = ((1 - w) * (y - mu0)) / (1 - e_hat)

    # Regression terms
    regression_term = mu1 - mu0  #cate_pred_out

    # Combine components
    dr_estimator =  ipw_treated - ipw_control + regression_term

    # Average over all samples
    ate = dr_estimator.mean()
    return ate #dr_estimator

def double_ml_tensor(X, y, w, mu0, mu1, e_hat):
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

    w = w.reshape(-1, 1)  # (1530, 1)
    e_hat = e_hat.reshape(-1, 1)  # (1530, 1)

    # 아래 세개 딥러닝 넣었다 나온 파라미터 쓸 때 꼭 쓰기..tensor -> numpy
    y = y.reshape(-1, 1)
    mu0 = mu0.numpy().reshape(-1, 1)
    mu1 = mu1.numpy().reshape(-1, 1)

    # IPW terms
    # ipw_treated = (w * y) / e_hat
    # ipw_control = ((1 - w) * y) / (1 - e_hat)

    ipw_treated = (w * (y - mu1)) / e_hat
    ipw_control = ((1 - w) * (y - mu0)) / (1 - e_hat)

    # Regression terms
    regression_term = mu1 - mu0  #cate_pred_out

    # Combine components
    dr_estimator =  ipw_treated - ipw_control + regression_term

    # Average over all samples
    ate = dr_estimator.mean()
    return ate #dr_estimator

def res_calibration(X, y, mu_hat):
    # Step 1: Residuals 계산
    residuals = y - mu_hat  # 실제 값과 예측값 간의 차이

    # Ridge
    # # Step 2: 보정 함수 g(x) 학습
    params = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    ridge_cv = GridSearchCV(Ridge(), params, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X, residuals)
    best_alpha = ridge_cv.best_params_['alpha']
    g_model = Ridge(alpha=best_alpha)  # Regularization 적용
    g_model.fit(X, residuals)

    # Step 3: 보정 값 계산
    g_hat = g_model.predict(X)  # 보정값 g(x)

    # Step 4: Adjusted mu 계산
    mu_adjusted = mu_hat + g_hat

    # mu0_adjusted = mu0_hat + g_hat  # 보정된 mu0
    # mu1_adjusted = mu1_hat + g_hat  # 보정된 mu1

    # Step 5: 새로운 CATE 계산
    #cate_adjusted = mu1_adjusted - mu0_adjusted

    return mu_adjusted

def repr_calibration(X, y, mu_hat):
    residuals = y - mu_hat

    latent_model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)

    latent_model.fit(X, residuals)
    g_hat = latent_model.predict(X)

    g_hat = g_hat.reshape(-1,1)

    mu_adjusted = mu_hat + g_hat

    return mu_adjusted

def cate_residualNet(phi_representation, y_t_w1, y_t_w0, cate_pred_out, idx_w0, idx_w1):
    # 텐서로 변환
    X_tensor = torch.from_numpy(np.array(phi_representation.reshape(-1,200))).float() #1530 x 200
    y1_tensor = torch.from_numpy(np.array(y_t_w1.reshape(-1,1))).float() #517
    y0_tensor = torch.from_numpy(np.array(y_t_w0.reshape(-1,1))).float() #1013
    cate_pred_out_tensor = torch.from_numpy(np.array(cate_pred_out.reshape(-1,1))).float() #1530

    # 샘플 수 확인
    n_samples = X_tensor.shape[0]

    # 모델 초기화
    input_dim = X_tensor.shape[1] #200
    mu1_model = MuModel(input_dim=input_dim)
    mu0_model = MuModel(input_dim=input_dim)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(mu1_model.parameters()) + list(mu0_model.parameters()), lr=0.001)

    # 학습
    n_epochs = 600
    lambda1, lambda2, lambda3 = 0.3, 1.0, 1.0 # 가중치 설정

    for epoch in range(n_epochs):

        mu1_pred = mu1_model(X_tensor)
        mu0_pred = mu0_model(X_tensor)
        X_tensor_1 = X_tensor[idx_w1]  # w=1 데이터
        X_tensor_0 = X_tensor[idx_w0]  # w=0 데이터

        mu1_pred_1 = mu1_model(X_tensor_1)
        mu0_pred_0 = mu0_model(X_tensor_0)

        # Loss 계산
        loss_cate = criterion(mu1_pred - mu0_pred, cate_pred_out_tensor) # CATE 유지
        loss_y1 = criterion(mu1_pred_1, y1_tensor) # mu1과 y1 핏
        loss_y0 = criterion(mu0_pred_0, y0_tensor) # mu0과 y0 핏

        total_loss = lambda1 * loss_cate + lambda2 * loss_y1 + lambda3 * loss_y0

        # 역전파 및 최적화
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4f}")

    X_tensor_0 = X_tensor[idx_w0]
    X_tensor_1 = X_tensor[idx_w1]
    # 최종 결과 확인
    mu1_final = mu1_model(X_tensor_1).detach()
    mu0_final = mu0_model(X_tensor_0).detach()
    #print("Final CATE predictions:", (mu1_final - mu0_final).numpy())

    mu1_ = mu1_model(X_tensor).detach() #1530
    mu0_ = mu0_model(X_tensor).detach() #1530

    return mu1_final, mu0_final, mu1_, mu0_


