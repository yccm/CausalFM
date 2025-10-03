# CausalFM for IV Setting (instrumental variable)



from __future__ import annotations

import random
import warnings
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import partial
from typing import Any, Literal

import einops
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from tabpfn.model.encoders import (
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from tabpfn.model.layer import PerFeatureEncoderLayer

DEFAULT_EMSIZE = 128


@contextmanager
def isolate_torch_rng(seed: int, device: torch.device) -> Generator[None, None, None]:
    torch_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state(device=device)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch_cuda_rng_state, device=device)


class LayerStack(nn.Module):
    """Similar to nn.Sequential, but with support for passing keyword arguments
    to layers and stacks the same layer multiple times.
    """

    def __init__(
        self,
        *,
        layer_creator: Callable[[], nn.Module],
        num_layers: int,
        recompute_each_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([layer_creator() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.min_num_layers_layer_dropout = (
            min_num_layers_layer_dropout
            if min_num_layers_layer_dropout is not None
            else num_layers
        )
        self.recompute_each_layer = recompute_each_layer

    def forward(
        self,
        x: torch.Tensor,
        *,
        half_layers: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if half_layers:
            assert (
                self.min_num_layers_layer_dropout == self.num_layers
            ), "half_layers only works without layer dropout"
            n_layers = self.num_layers // 2
        else:
            n_layers = torch.randint(
                low=self.min_num_layers_layer_dropout,
                high=self.num_layers + 1,
                size=(1,),
            ).item()

        for layer in self.layers[:n_layers]:
            if self.recompute_each_layer and x.requires_grad:
                x = checkpoint(partial(layer, **kwargs), x, use_reentrant=False)
            else:
                x = layer(x, **kwargs)

        return x



class GMMHead(nn.Module):

    def __init__(self, z_dim: int, n_components: int, min_sigma: float = 1e-3, pi_temp: float = 1.0):
        super().__init__()
        self.K = n_components
        self.min_sigma = min_sigma
        self.pi_temp = pi_temp
        self.fc_pi = nn.Linear(z_dim, n_components) 
        self.fc_mu = nn.Linear(z_dim, n_components) 
        self.fc_sigma = nn.Linear(z_dim, n_components) 

    def forward(self, z: torch.Tensor):

        logits = self.fc_pi(z) / self.pi_temp # temperature scaling
        pi = F.softmax(logits, dim=-1) # mixture weight 
        mu = self.fc_mu(z)
        sigma = F.softplus(self.fc_sigma(z)) + self.min_sigma
        return pi, mu, sigma


class PerFeatureTransformerCATE(nn.Module):

    def __init__(
        self,
        *,
        use_gmm_head: bool = True,
        gmm_n_components: int =5,
        gmm_min_sigma: float = 1e-3,
        gmm_pi_temp: float = 1.0,
        x_encoder: nn.Module | None = None,  # For covariates X
        a_encoder: nn.Module | None = None,  # For treatment A  
        y_encoder: nn.Module | None = None,  # For factual outcome Y
        z_encoder: nn.Module | None = None,  # For instrumental variable Z
        ninp: int = DEFAULT_EMSIZE,
        nhead: int = 4,
        nhid: int = DEFAULT_EMSIZE * 4,
        nlayers: int = 10,
        decoder_dict: dict[str, tuple[type[nn.Module] | None, int]] | None = None,
        init_method: str | None = None,
        activation: Literal["gelu", "relu"] = "gelu",
        recompute_layer: bool = False,
        min_num_layers_layer_dropout: int | None = None,
        repeat_same_layer: bool = False,
        dag_pos_enc_dim: int = 0,
        features_per_group: int = 1,
        feature_positional_embedding: (
            Literal[
                "normal_rand_vec",
                "uni_rand_vec", 
                "learned",
                "subspace",
            ]
            | None
        ) = None,
        zero_init: bool = True,
        use_separate_decoder: bool = False,
        nlayers_decoder: int | None = None,
        use_encoder_compression_layer: bool = False,
        precomputed_kv: (
            list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] | None
        ) = None,
        cache_trainset_representation: bool = False,
        seed: int | None = None,
        **layer_kwargs: Any,
    ):


        decoder_dict = {
                     "cate": (None, 1),          
                     "cate_rep": (None, ninp),   
                        }


        super().__init__()

        # Set up encoders for X, A, Y, Z
        if x_encoder is None:
            x_encoder = SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=1,
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
            )
        if a_encoder is None:
            a_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # A + nan indicators
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )
        if z_encoder is None:
            z_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # Z + nan indicators
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )

        if y_encoder is None:
            y_encoder = SequentialEncoder(
                NanHandlingEncoderStep(),
                LinearInputEncoderStep(
                    num_features=2,  # Y + nan indicators
                    emsize=DEFAULT_EMSIZE,
                    replace_nan_by_zero=False,
                    bias=True,
                    out_keys=("output",),
                    in_keys=("main", "nan_indicators"),
                ),
            )


        self.x_encoder = x_encoder  # Covariate encoder
        self.a_encoder = a_encoder  # Treatment encoder  
        self.y_encoder = y_encoder  # Outcome encoder
        self.z_encoder = z_encoder  # Instrumental variable encoder

        self.ninp = ninp
        self.nhead = nhead
        self.nhid = nhid
        self.init_method = init_method
        self.features_per_group = features_per_group
        self.cache_trainset_representation = cache_trainset_representation
        self.cached_embeddings: torch.Tensor | None = None

        layer_creator = lambda: PerFeatureEncoderLayer(
            d_model=ninp,
            nhead=nhead,
            dim_feedforward=nhid,
            activation=activation,
            zero_init=zero_init,
            precomputed_kv=(
                precomputed_kv.pop(0) if precomputed_kv is not None else None
            ),
            **layer_kwargs,
        )
        if repeat_same_layer:
            layer = layer_creator()
            layer_creator = lambda: layer

        nlayers_encoder = nlayers
        if use_separate_decoder and nlayers_decoder is None:
            nlayers_decoder = max((nlayers // 3) * 1, 1)
            nlayers_encoder = max((nlayers // 3) * 2, 1)

        self.transformer_encoder = LayerStack(
            layer_creator=layer_creator,
            num_layers=nlayers_encoder,
            recompute_each_layer=recompute_layer,
            min_num_layers_layer_dropout=min_num_layers_layer_dropout,
        )

        self.transformer_decoder = None
        if use_separate_decoder:
            assert nlayers_decoder is not None
            self.transformer_decoder = LayerStack(
                layer_creator=layer_creator,
                num_layers=nlayers_decoder,
            )

        self.global_att_embeddings_for_compression = None
        if use_encoder_compression_layer:
            assert use_separate_decoder
            num_global_att_tokens_for_compression = 512

            self.global_att_embeddings_for_compression = nn.Embedding(
                num_global_att_tokens_for_compression,
                ninp,
            )

            self.encoder_compression_layer = LayerStack(
                layer_creator=layer_creator,
                num_layers=2,
            )

        
        initialized_decoder_dict = {}
        for decoder_key in decoder_dict:
            decoder_model, decoder_n_out = decoder_dict[decoder_key]
            if decoder_model is None:
                initialized_decoder_dict[decoder_key] = nn.Sequential(
                    nn.Linear(ninp, nhid),
                    nn.GELU(),
                    nn.Linear(nhid, decoder_n_out),
                )
            else:
                initialized_decoder_dict[decoder_key] = decoder_model(
                    ninp,
                    nhid,
                    decoder_n_out,
                )
        self.decoder_dict = nn.ModuleDict(initialized_decoder_dict)

        self.feature_positional_embedding = feature_positional_embedding
        if feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(1_000, ninp)
        elif feature_positional_embedding == "subspace":
            self.feature_positional_embedding_embeddings = nn.Linear(ninp // 4, ninp)

        self.dag_pos_enc_dim = dag_pos_enc_dim
        self.cached_feature_positional_embeddings: torch.Tensor | None = None
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)


        self.use_gmm_head = use_gmm_head
        if self.use_gmm_head:
            self.gmm_head = GMMHead(
                z_dim = self.ninp,
                n_components = gmm_n_components,
                min_sigma = gmm_min_sigma,
                pi_temp = gmm_pi_temp,
            )

    def _pack_eval_io(
        self,
        X_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        X_test: torch.Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Expected input shapes: 
            X_train: [N_tr, Dx],  A_train: [N_tr, 1],  Y_train: [N_tr, 1], Z_train: [N_tr, 1], X_test: [N_te, Dx]
        """
        
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = torch.float32

        X_train = X_train.to(device=device, dtype=dtype)
        A_train = A_train.to(device=device, dtype=dtype)
        Y_train = Y_train.to(device=device, dtype=dtype)
        Z_train = Z_train.to(device=device, dtype=dtype)
        X_test  = X_test.to(device=device,  dtype=dtype)

        n_tr = X_train.shape[0]
        n_te = X_test.shape[0]
        single_eval_pos = int(n_tr)


        X_all = torch.cat([X_train, X_test], dim=0)            
        A_all = torch.cat([A_train, torch.full((n_te, A_train.shape[1]), torch.nan, device=device, dtype=dtype)], dim=0)
        Y_all = torch.cat([Y_train, torch.full((n_te, Y_train.shape[1]), torch.nan, device=device, dtype=dtype)], dim=0)
        Z_all = torch.cat([Z_train, torch.full((n_te, Z_train.shape[1]), torch.nan, device=device, dtype=dtype)], dim=0)

        # add batch dim = 1 -> [S, 1, D]
        X_all = X_all.unsqueeze(1)
        A_all = A_all.unsqueeze(1)
        Y_all = Y_all.unsqueeze(1)
        Z_all = Z_all.unsqueeze(1)

        x = {"main": X_all}   
        a = {"main": A_all}   
        y = {"main": Y_all}   
        z = {"main": Z_all}   
        return x, a, y, z, single_eval_pos

    @torch.no_grad()
    def estimate_cate_iv(
        self,
        X_train: torch.Tensor,
        A_train: torch.Tensor,
        Y_train: torch.Tensor,
        Z_train: torch.Tensor,
        X_test: torch.Tensor,
        *,
        return_gmm: bool = False,
        **forward_kwargs,
    ):
        """
            inputs: X_train, A_train, Y_train, Z_train, X_test 
            returns: dict with predicted CATE on X_test (and optional GMM params)
        """
        x, a, y, z, single_eval_pos = self._pack_eval_io(X_train, A_train, Y_train, Z_train, X_test)
        out = self.forward(x, a, y, z, single_eval_pos=single_eval_pos, **forward_kwargs)

        
        cate = out["cate_mean"][:, single_eval_pos:, :]
        cate = cate.squeeze()
        result = {"cate": cate}

        if return_gmm and ("gmm_pi" in out):
            result.update({
                "gmm_pi":    out["gmm_pi"][0, single_eval_pos:, :],   
                "gmm_mu":    out["gmm_mu"][0, single_eval_pos:, :],   
                "gmm_sigma": out["gmm_sigma"][0, single_eval_pos:, :],
            })
        return result

    def forward(
        self,
        x: torch.Tensor | dict,
        a: torch.Tensor | dict | None = None,
        y: torch.Tensor | dict | None = None,
        z: torch.Tensor | dict | None = None,
        single_eval_pos: int | None = None,
        *,
        style: torch.Tensor | None = None,
        data_dags: list[Any] | None = None,
        categorical_inds: list[int] | None = None,
        half_layers: bool = False,
    ) -> Any | dict[str, torch.Tensor]:

        if isinstance(x, dict):
            assert "main" in set(x.keys()), f"Main must be in input keys: {x.keys()}."
        else:
            x = {"main": x}
        seq_len, batch_size, num_x_features = x["main"].shape # seq_len = n_train_rows + n_test_rows
            
        if isinstance(a, dict):
            assert "main" in set(a.keys()), f"Main must be in input keys: {a.keys()}."
        else:
            a = {"main": a}
        _, _, num_a_features = a["main"].shape

        if isinstance(z, dict):
            assert "main" in set(z.keys()), f"Main must be in input keys: {z.keys()}."
        else:
            z = {"main": z}
        _, _, num_z_features = z["main"].shape

        if isinstance(y, dict):
            assert "main" in set(y.keys())
        else:
            y = {"main": y}
        _, _, num_y_features = y["main"].shape

        training_targets_provided = y["main"] is not None and y["main"].shape[0] > 0

        # Handle y (can be None during inference)
        if y is None:
            y = torch.zeros(
                0,
                batch_size, 
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        # Handle a (can be None during inference)
        if a is None:
            a = torch.zeros(
                0,
                batch_size, 
                device=x["main"].device,
                dtype=x["main"].dtype,
            )

        # Handle z (can be None during inference)
        if z is None:
            z = torch.zeros(
                0,
                batch_size, 
                device=x["main"].device,
                dtype=x["main"].dtype,
            )
        

        # The model will make predictions from the single_eval_pos'th row onwards. 
        
        single_eval_pos_ = int(single_eval_pos)


        # Pad features to multiple of features_per_group
        for k in x:
            num_features_ = x[k].shape[2]

            
            missing_to_next = (
                self.features_per_group - (num_features_ % self.features_per_group)
            ) % self.features_per_group

            if missing_to_next > 0:
                x[k] = torch.cat(
                    (
                        x[k],
                        torch.zeros(
                            seq_len,
                            batch_size,
                            missing_to_next,
                            device=x[k].device,
                            dtype=x[k].dtype,
                        ),
                    ),
                    dim=-1,
                )

        for k in x:
            x[k] = einops.rearrange(
                x[k],
                "s b (f n) -> b s f n",
                n=self.features_per_group,
            )  

        
        for k in y:
            if y[k].ndim == 1:
                y[k] = y[k].unsqueeze(-1)
            if y[k].ndim == 2:
                y[k] = y[k].unsqueeze(-1)  # s b -> s b 1

            y[k] = y[k].transpose(0, 1)  # s b 1 -> b s 1

            if y[k].shape[1] < x["main"].shape[1]:
                assert (
                    y[k].shape[1] == single_eval_pos
                    or y[k].shape[1] == x["main"].shape[1]
                )
                assert k != "main" or y[k].shape[1] == single_eval_pos, (
                    "For main y, y must not be given for target"
                    " time steps (Otherwise the solution is leaked)."
                )
                if y[k].shape[1] == single_eval_pos:
                    y[k] = torch.cat(
                        (
                            y[k],
                            torch.nan
                            * torch.zeros(
                                y[k].shape[0],
                                x["main"].shape[1] - y[k].shape[1],
                                y[k].shape[2],
                                device=y[k].device,
                                dtype=y[k].dtype,
                            ),
                        ),
                        dim=1,
                    )

            y[k] = y[k].transpose(0, 1)  # b s 1 -> s b 1

        # making sure no label leakage ever happens
        y["main"][single_eval_pos:] = torch.nan
        a["main"][single_eval_pos:] = torch.nan
        z["main"][single_eval_pos:] = torch.nan

        # Encode factual outcome Y
        embedded_y = self.y_encoder(
            y,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del y

        if torch.isnan(embedded_y).any():
            raise ValueError(
                f"{torch.isnan(embedded_y).any()=}, make sure to add nan handlers"
                " to the ys that are not fully provided (test set missing)",
            )
        

        embedded_a = self.a_encoder(
            a,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)

        del a

        embedded_z = self.z_encoder(
            z,
            single_eval_pos=single_eval_pos_,
            cache_trainset_representation=self.cache_trainset_representation,
        ).transpose(0, 1)
        del z


        if torch.isnan(embedded_a).any():
            raise ValueError(
                f"{torch.isnan(embedded_a).any()=}, make sure to add nan handlers"
                " to the as that are not fully provided (test set missing)",
            )

        for k in x:
            x[k] = einops.rearrange(x[k], "b s f n -> s (b f) n")
        embedded_x = einops.rearrange(
            self.x_encoder(
                x,
                single_eval_pos=single_eval_pos_,
                cache_trainset_representation=self.cache_trainset_representation,

            ),
            "s (b f) e -> b s f e",
            b=batch_size,
        )
        del x

        embedded_z_grouped = embedded_z.unsqueeze(2)  # (b, s, e) -> (b, s, 1, e)
        embedded_a_grouped = embedded_a.unsqueeze(2)  # (b, s, e) -> (b, s, 1, e)
        embedded_y_grouped = embedded_y.unsqueeze(2)  # (b, s, e) -> (b, s, 1, e)


        embedded_input = torch.cat(
            (embedded_z_grouped, embedded_x, embedded_a_grouped, embedded_y_grouped), dim=2
        )  # (b, s, f_z + f_x + 1 + 1, e)

        if torch.isnan(embedded_input).any():
            raise ValueError(
                f"There should be no NaNs in the encoded x, a, and y."
                f"Your embedded inputs returned the following:"
                f"{torch.isnan(embedded_x).any()=} | {torch.isnan(embedded_a).any()=} | {torch.isnan(embedded_y).any()=}",
            )

        embedded_input = self.add_embeddings_cate(
            embedded_input,
            data_dags=data_dags,
            num_features=num_x_features + num_a_features + num_z_features,
            seq_len=seq_len,
        )

        # Pass through transformer
        encoder_out = self.transformer_encoder(
            embedded_input,
            single_eval_pos=single_eval_pos_,
            half_layers=half_layers,
            cache_trainset_representation=self.cache_trainset_representation,
        )

        
        if self.transformer_decoder:
            assert not half_layers
            assert encoder_out.shape[1] == single_eval_pos_

            test_encoder_out = self.transformer_decoder(
                embedded_input[:, single_eval_pos_:],
                single_eval_pos=0,
                att_src=encoder_out,
            )
            encoder_out = torch.cat([encoder_out, test_encoder_out], 1)


        cate_representation = encoder_out.mean(dim=2)  
        
        gmm_out = None

        out = {}

        if self.use_gmm_head:
            B, S, E = cate_representation.shape
            z_rep = self.decoder_dict["cate_rep"](cate_representation).reshape(B * S, E)

            pi, mu, sigma = self.gmm_head(z_rep)
            K = pi.shape[-1]
            pi   = pi.reshape(B, S, K)
            mu   = mu.reshape(B, S, K)
            sigma= sigma.reshape(B, S, K)

            gmm_out = {"gmm_pi": pi, "gmm_mu": mu, "gmm_sigma": sigma}

        
        if gmm_out is not None:
            out.update(gmm_out)  # add gmm params if enabled
            out["cate_mean"] = (pi * mu).sum(dim=-1, keepdim=True)
        return out
        


    def add_embeddings_cate(
        self,
        embedded_input: torch.Tensor,
        *,
        data_dags: Iterable[nx.DiGraph] | None,
        num_features: int,
        seq_len: int,
    ) -> torch.Tensor:
        """embedding addition for causality"""
        
        with isolate_torch_rng(self.seed, device=embedded_input.device):
            if self.feature_positional_embedding == "normal_rand_vec":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3]),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "uni_rand_vec":
                embs = (
                    torch.rand(
                        (embedded_input.shape[2], embedded_input.shape[3]),
                        device=embedded_input.device,
                        dtype=embedded_input.dtype,
                    )
                    * 2
                    - 1
                )
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "learned":
                w = self.feature_positional_embedding_embeddings.weight
                embs = w[
                    torch.randint(
                        0,
                        w.shape[0],
                        (embedded_input.shape[2],),
                    )
                ]
                embedded_input += embs[None, None]
            elif self.feature_positional_embedding == "subspace":
                embs = torch.randn(
                    (embedded_input.shape[2], embedded_input.shape[3] // 4),
                    device=embedded_input.device,
                    dtype=embedded_input.dtype,
                )
                embs = self.feature_positional_embedding_embeddings(embs)
                embedded_input += embs[None, None]

        return embedded_input


    def reset_save_peak_mem_factor(self, factor: int | None = None) -> None:
        """Sets the save_peak_mem_factor for all layers."""
        for layer in self.transformer_encoder.layers:
            assert hasattr(layer, "save_peak_mem_factor")
            layer.save_peak_mem_factor = factor

    def empty_trainset_representation_cache(self) -> None:
        for layer in (self.transformer_decoder or self.transformer_encoder).layers:
            layer.empty_trainset_representation_cache()
