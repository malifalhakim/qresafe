import os
import torch
import inspect
import logging
import functools
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime
from utils.calib_data import get_calib_dataset, get_fairness_dataset, get_safety_dataset, get_general_dataset
from quantize.scale import apply_scale, apply_clip
from utils.utils import clear_memory, get_best_device
from modules.linear import (
    WQLinear_GEMM,
    WQLinear_GEMV,
    WQLinear_Marlin,
    WQLinear_GEMVFast,
)
from utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)

class AwqQuantizer:
    def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
        protect_safety=False,
        protect_fairness=False,
        beta=1.0,
        tau=0.6,
        model_name="none"
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

        if protect_safety and protect_fairness:
            raise ValueError("Cannot protect both safety and fairness at the same time for now.")
        
        self.protect_safety = protect_safety
        self.protect_fairness = protect_fairness
        self.beta = beta
        self.tau = tau
        self.saved_scores_dir = "./saved_scores"
        self.model_name = model_name

    def _calculate_safescore(self, named_linears):
        """
        Calculate safe score based on adaptation of "FairQuantize" method. 
        Importance_score = H_general - Beta * H_safety, where H is hessian.
        """
        print(f"Calculating safety scores for {len(named_linears)} layers")
        general_data = get_general_dataset(
            dataset_name="databricks/databricks-dolly-15k",
            subset=None,
            split="train",
            use_template=True,
            text_column="response",
            prompt_column="instruction",
            tokenizer=self.tokenizer,
        )
        safety_data = get_safety_dataset(tokenizer=self.tokenizer)

        input_general = []
        target_general = []
        for input_ids, target_ids in general_data:
            input_general.append(input_ids)
            target_general.append(target_ids)
        
        input_safety = []
        target_safety = []
        for input_ids, target_ids in safety_data:
            input_safety.append(input_ids)
            target_safety.append(target_ids)

        device = get_best_device()
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        accumulated_scores = {
            name: torch.zeros_like(module.weight, device='cpu')
            for name, module in named_linears.items()
        }

        num_data = 0
        clear_memory()

        for i in tqdm(range(0, len(input_general)), desc="Calculating Hessians"):
            input_gen = input_general[i].to(device)
            target_gen = target_general[i].to(device)

            num_data += 1
            self.model.train()

            for module in named_linears.values():
                module.weight.requires_grad = True

            self.model.zero_grad()
            outputs_gen = self.model(input_gen)
            logits_gen = outputs_gen.logits

            prediction_logits_gen = logits_gen[:, :-1, :]
            target_labels_gen = target_gen[:, 1:]

            if i == 0:
                print(f"Input Gen Token IDs: {input_gen}")
                print(f"Target Gen Token IDs: {target_gen}")
                print(f"Prediction Logits Shape: {prediction_logits_gen.shape}")
                print(f"Target Labels Shape: {target_labels_gen.shape}")
                print(f"Input Gen Shape: {input_gen.shape}")
                print(f"Logits Shape: {logits_gen.shape}")
                print(f"Target Gen Shape: {target_gen.shape}")

            assert prediction_logits_gen.size(1) == target_labels_gen.size(1), \
                f"Prediction length {prediction_logits_gen.size(1)} does not match target length {target_labels_gen.size(1)}"
            
            loss_gen = criterion(prediction_logits_gen.reshape(-1, prediction_logits_gen.size(-1)), target_labels_gen.reshape(-1))
            loss_gen.backward()

            if i == 0:
                print(f"\nFirst general batch loss: {loss_gen.item():.6f}")
                print(f"Loss requires_grad: {loss_gen.requires_grad}")
                print(f"Logits requires_grad: {logits_gen.requires_grad}")
                print(f"Target Token IDs: {target_labels_gen[0].tolist()}")
                print(f"Target Words: {[self.tokenizer.decode([tid]) for tid in target_labels_gen[0].tolist() if tid != -100]}")

            for name, module in named_linears.items():
                if module.weight.grad is not None:
                    squared_gradient = module.weight.grad.detach().pow(2)
                    accumulated_scores[name] += squared_gradient.cpu()
                    module.weight.grad = None 
                
                module.weight.requires_grad = False

            clear_memory()
        
        importance_scores = {}
        for name, acc_score in tqdm(accumulated_scores.items(), desc="Averaging general scores"):
            importance_scores[name] = acc_score / num_data
        
        self.model.eval()
        clear_memory()
        del accumulated_scores

        accumulated_scores = {
            name: torch.zeros_like(module.weight, device='cpu')
            for name, module in named_linears.items()
        }

        num_data = 0
        clear_memory()

        for i in tqdm(range(0, len(input_safety)), desc="Calculating Hessians"):
            input_safe = input_safety[i].to(device)
            target_safe = target_safety[i].to(device)

            num_data += 1
            self.model.train()

            for module in named_linears.values():
                module.weight.requires_grad = True
            
            self.model.zero_grad()

            outputs_safe = self.model(input_safe)
            logits_safe = outputs_safe.logits

            prediction_logits_safe = logits_safe[:, :-1, :]
            target_labels_safe = target_safe[:, 1:]

            if i == 0:
                print(f"Input Safe Token IDs: {input_safe}")
                print(f"Target Safe Token IDs: {target_safe}")
                print(f"Prediction Logits Shape: {prediction_logits_safe.shape}")
                print(f"Target Labels Shape: {target_labels_safe.shape}")
                print(f"Input Safe Shape: {input_safe.shape}")
                print(f"Logits Shape: {logits_safe.shape}")
                print(f"Target Safe Shape: {target_safe.shape}")

            assert prediction_logits_safe.size(1) == target_labels_safe.size(1), \
                f"Prediction length {prediction_logits_safe.size(1)} does not match target length {target_labels_safe.size(1)}"
            
            loss_safe = criterion(prediction_logits_safe.view(-1, prediction_logits_safe.size(-1)), target_labels_safe.view(-1))
            loss_safe.backward()

            if i == 0:
                print(f"\nFirst safety batch loss: {loss_safe.item():.6f}")
                print(f"Loss requires_grad: {loss_safe.requires_grad}")
                print(f"Logits requires_grad: {logits_safe.requires_grad}")
                print(f"Target Token IDs: {target_labels_safe[0].tolist()}")
                print(f"Target Words: {[self.tokenizer.decode([tid]) for tid in target_labels_safe[0].tolist() if tid != -100]}")

            for name, module in named_linears.items():
                if module.weight.grad is not None:
                    squared_gradient = module.weight.grad.detach().pow(2)
                    accumulated_scores[name] += squared_gradient.cpu()
                    module.weight.grad = None 
                
                module.weight.requires_grad = False

            clear_memory()

        print("Calculating final importance scores...")
        for name, acc_score in tqdm(accumulated_scores.items(), desc="Averaging importance scores"):
            safe_score = acc_score / num_data
            importance_scores[name] = safe_score - self.beta * importance_scores[name]

        del accumulated_scores
        self.model.eval()
        clear_memory()
        return importance_scores
    
    def _calculate_fairscore(self, named_linears):
        """
        Calculate FairScore for each weight in the model.
        """
        general_data = get_general_dataset(
            tokenizer=self.tokenizer,
        )
        fairness_data = get_fairness_dataset(tokenizer=self.tokenizer)

        input_general = []
        for input_ids, _ in general_data:
            input_general.append(input_ids)

        input_stereotypes = []
        target_stereotypes = []
        input_antistereotypes = []
        target_antistereotypes = []
        for input_ids_stereotype, label_ids_stereotype, input_ids_antistereotype, label_ids_antistereotype in fairness_data:
            input_stereotypes.append(input_ids_stereotype)
            target_stereotypes.append(label_ids_stereotype)
            input_antistereotypes.append(input_ids_antistereotype)
            target_antistereotypes.append(label_ids_antistereotype)

        device = get_best_device()
        self.model.to(device)

        criterion = torch.nn.CrossEntropyLoss()

        accumulated_scores = {
            name: torch.zeros_like(module.weight, device='cpu')
            for name, module in named_linears.items()
        }

        num_data = 0
        clear_memory()

        for i in tqdm(range(0, len(input_general)), desc="Calculating Hessians"):
            sentence_gen = input_general[i].to(device)

            num_data += 1
            self.model.train()

            for module in named_linears.values():
                module.weight.requires_grad = True

            self.model.zero_grad()
            outputs_gen = self.model(sentence_gen)
            logits_gen = outputs_gen.logits

            prediction_logits_gen = logits_gen[:, :-1, :]
            target_labels_gen = sentence_gen[:, 1:]

            if i == 0:
                print(f"Input Gen Token IDs: {sentence_gen}")
                print(f"Prediction Logits Shape: {prediction_logits_gen.shape}")
                print(f"Target Labels Shape: {target_labels_gen.shape}")
                print(f"Input Gen Shape: {sentence_gen.shape}")
                print(f"Logits Shape: {logits_gen.shape}")


            assert prediction_logits_gen.size(1) == target_labels_gen.size(1), \
                f"Prediction length {prediction_logits_gen.size(1)} does not match target length {target_labels_gen.size(1)}"
            
            loss_gen = criterion(prediction_logits_gen.reshape(-1, prediction_logits_gen.size(-1)), target_labels_gen.reshape(-1))
            loss_gen.backward()

            if i == 0:
                print(f"\nFirst general batch loss: {loss_gen.item():.6f}")
                print(f"Loss requires_grad: {loss_gen.requires_grad}")
                print(f"Logits requires_grad: {logits_gen.requires_grad}")
                print(f"Target Token IDs: {target_labels_gen[0].tolist()}")
                print(f"Target Words: {[self.tokenizer.decode([tid]) for tid in target_labels_gen[0].tolist() if tid != -100]}")

            for name, module in named_linears.items():
                if module.weight.grad is not None:
                    squared_gradient = module.weight.grad.detach().pow(2)
                    accumulated_scores[name] += squared_gradient.cpu()
                    module.weight.grad = None 
                
                module.weight.requires_grad = False

            clear_memory()

        importance_scores = {}
        for name, acc_score in tqdm(accumulated_scores.items(), desc="Averaging importance scores"):
            importance_scores[name] = acc_score / num_data

        self.model.eval()
        clear_memory()
        del accumulated_scores

        accumulated_scores = {
            name: torch.zeros_like(module.weight, device='cpu')
            for name, module in named_linears.items()
        }

        num_data = 0
        clear_memory()

        for i in tqdm(range(0, len(input_stereotypes)), desc="Calculating Hessians"):
            input_stereotype = input_stereotypes[i].to(device)
            target_stereotype = target_stereotypes[i].to(device)
            input_antistereotype = input_antistereotypes[i].to(device)
            target_antistereotype = target_antistereotypes[i].to(device)

            num_data += 1
            self.model.train()

            for module in named_linears.values():
                module.weight.requires_grad = True

            self.model.zero_grad()

            outputs_stereo = self.model(input_stereotype)
            logits_stereo = outputs_stereo.logits

            prediction_logits_stereo = logits_stereo[:, :-1, :]
            target_labels_stereo = target_stereotype[:, 1:]
            nll_stereo = criterion(prediction_logits_stereo.view(-1, prediction_logits_stereo.size(-1)), target_labels_stereo.view(-1))

            outputs_antistereo = self.model(input_antistereotype)
            logits_antistereo = outputs_antistereo.logits

            prediction_logits_antistereo = logits_antistereo[:, :-1, :]
            target_labels_antistereo = target_antistereotype[:, 1:]
            nll_antistereo = criterion(prediction_logits_antistereo.view(-1, prediction_logits_antistereo.size(-1)), target_labels_antistereo.view(-1))

            if i == 0:
                print(f"Input Fair Token IDs: {input_stereotype} \n| {input_antistereotype}")
                print(f"Target Fair Token IDs: {target_stereotype} \n| {target_antistereotype}")
                print(f"Output stereo Shape: {logits_stereo.shape} \n| Output anti-stereo Shape: {logits_antistereo.shape}")
                print(f"Prediction Logits stereo Shape: {prediction_logits_stereo.shape} \n| Prediction Logits anti-stereo Shape: {prediction_logits_antistereo.shape}")

            loss_fair = torch.abs(nll_stereo - nll_antistereo)
            loss_fair.backward()

            if i == 0:
                print(f"\nFirst fairness batch loss: {loss_fair.item():.6f}")
                print(f"Loss requires_grad: {loss_fair.requires_grad}")
                print(f"Target Token IDs Stereo: {target_labels_stereo[0].tolist()}")
                print(f"Target Words Stereo: {[self.tokenizer.decode([tid]) for tid in target_labels_stereo[0].tolist() if tid != -100]}")
                print(f"Target Token IDs Anti-Stereo: {target_labels_antistereo[0].tolist()}")
                print(f"Target Words Anti-Stereo: {[self.tokenizer.decode([tid]) for tid in target_labels_antistereo[0].tolist() if tid != -100]}")

            for name, module in named_linears.items():
                if module.weight.grad is not None:
                    squared_gradient = module.weight.grad.detach().pow(2)
                    accumulated_scores[name] += squared_gradient.cpu()
                    module.weight.grad = None 
                
                module.weight.requires_grad = False

            clear_memory()

        print("Calculating final importance scores...")
        for name, acc_score in tqdm(accumulated_scores.items(), desc="Averaging importance scores"):
            fair_score = acc_score / num_data
            importance_scores[name] = fair_score - self.beta * importance_scores[name]

        del accumulated_scores
        self.model.eval()
        clear_memory()
        return importance_scores

    def pseudo_quantize_tensor(self, w: torch.Tensor):
        org_w_shape = w.shape
        if self.group_size > 0:
            assert org_w_shape[-1] % self.group_size == 0
            w = w.reshape(-1, self.group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        if self.zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**self.w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (self.w_bit - 1) - 1
            min_int = -(2 ** (self.w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros


    def pseudo_dequantize_tensor(
        self, w: nn.Linear, scales: torch.Tensor, zeros: Optional[torch.Tensor] = None
    ):
        repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
        scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

        if self.zero_point:
            zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
            w = (w.weight.data - zeros) * scales
        else:
            w = w.weight.data * scales

        return w

    def quantize(self):
        all_named_linears = {}
        for i in range(len(self.modules)):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            module_prefix = get_op_name(self.model, self.modules[i]) + "."
            for name, layer in named_linears.items():
                full_name = module_prefix + name
                all_named_linears[full_name] = layer

        if self.protect_safety:
            print(f"Calculating safety-critical weights...")
            if os.path.exists(os.path.join(self.saved_scores_dir, f"{self.model_name}_safety_scores.pt")):
                all_scores = self._load_scores(f"{self.model_name}_safety_scores.pt")
            else:
                all_scores = self._calculate_safescore(all_named_linears)
                self._save_scores(all_scores, f"{self.model_name}_safety_scores.pt")
        elif self.protect_fairness:
            print(f"Calculating fairness-critical weights...")
            if os.path.exists(os.path.join(self.saved_scores_dir, f"{self.model_name}_fairness_scores.pt")):
                all_scores = self._load_scores(f"{self.model_name}_fairness_scores.pt")
            else:
                all_scores = self._calculate_fairscore(all_named_linears)
                self._save_scores(all_scores, f"{self.model_name}_fairness_scores.pt")
        else:
            all_scores = {}

        clear_memory()
        
        if self.protect_safety or self.protect_fairness:
            print("Aggregating scores...")
            all_scores_tensor = self._analyze_scores(all_scores)
            sample_size = min(1_000_000, all_scores_tensor.numel())
            indices = torch.randint(0, all_scores_tensor.numel(), (sample_size,))
            score_samples = all_scores_tensor.view(-1)[indices].cpu()

            print("Finding threshold via sampling and sorting...")
            tau = self.tau
            k = int(score_samples.numel() * tau)
            threshold = torch.topk(score_samples.float(), k, largest=True, sorted=False)[0].min()

            del all_scores_tensor, score_samples
            clear_memory()

            self.critical_threshold = threshold
            self.critical_scores = all_scores

            print(f"Critical threshold: {threshold}")

        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                if torch.cuda.is_available():
                    best_device = "cuda:" + str(i % torch.cuda.device_count())
                else:
                    best_device = get_best_device()

                self.modules[i] = self.modules[i].to(best_device)
                common_device = next(self.modules[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)

            # We need to move the rotary embedding every time we move to a new module.
            # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
            # https://github.com/huggingface/transformers/pull/32617
            self.awq_model.move_embed(self.model, common_device)

            for k, v in self.module_kwargs.items():
                # position embeddings found in tuple
                if isinstance(v, tuple):
                    self.module_kwargs[k] = tuple(
                        item.to(common_device) if isinstance(item, (torch.Tensor, nn.Module)) 
                        else item for item in v
                    )

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)

            clear_memory()
    
        # Clean up scores after quantization is complete
        del self.critical_scores
        self.critical_scores = None
        clear_memory()

    def pack(self):
        for i in tqdm(range(len(self.modules)), desc="Packing"):
            named_linears = get_named_linears(self.modules[i])
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )
            self._apply_quant(self.modules[i], named_linears)
            clear_memory()

    @torch.no_grad()
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            full_layer_name = get_op_name(self.model, linear_layer)
            
            mask = None
            if hasattr(self, 'critical_scores') and self.critical_scores is not None:
                if full_layer_name in self.critical_scores:
                    scores = self.critical_scores[full_layer_name]
                    mask = scores >= self.critical_threshold

                    del self.critical_scores[full_layer_name]
                    del scores
                    clear_memory()

            linear_layer = linear_layer.to(get_best_device()).half()

            # PATH 1: Layer contains critical weights -> Apply Mixed-Precision
            if mask is not None and torch.any(mask):
                print(f"Applying mixed-precision quantization to {full_layer_name}")
                
                mask = mask.to(linear_layer.weight.device)
                w_for_stats = linear_layer.weight.data.clone()

                w_for_stats[mask] = torch.inf
                min_val_unmasked = w_for_stats.amin(dim=-1, keepdim=True)
                
                w_for_stats[mask] = -torch.inf 
                max_val_unmasked = w_for_stats.amax(dim=-1, keepdim=True)

                if self.zero_point:
                    max_int = 2**self.w_bit - 1
                    scales = (max_val_unmasked - min_val_unmasked).clamp(min=1e-5) / max_int
                    zeros = (-torch.round(min_val_unmasked / scales)).clamp_(0, max_int)
                else:
                    max_val_unmasked = torch.abs(w_for_stats).amax(dim=-1, keepdim=True)
                    max_int = 2 ** (self.w_bit - 1) - 1
                    scales = max_val_unmasked.clamp(min=1e-5) / max_int
                    zeros = None

                w_quantized = linear_layer.weight.data.clone()
                if self.zero_point:
                    w_quantized = (torch.clamp(torch.round(w_quantized / scales) + zeros, 0, max_int) - zeros) * scales
                else:
                    min_int = -(2 ** (self.w_bit - 1))
                    w_quantized = torch.clamp(torch.round(w_quantized / scales), min_int, max_int) * scales
                
                
                w_quantized[mask] = linear_layer.weight.data[mask]
                linear_layer.weight.data = w_quantized
                
                del mask, w_for_stats
            
            # PATH 2: Layer is not critical -> Convert to fully quantized WQLinear
            else:
                if mask is not None:
                    del mask
                
                w_quantized, scales, zeros = self.pseudo_quantize_tensor(linear_layer.weight.data)
                
                linear_layer.weight.data = w_quantized

                if self.version == "gemm":
                    scales = scales.t().contiguous()
                    if zeros is not None:
                        zeros = zeros.t().contiguous()
                    q_linear_module = WQLinear_GEMM
                elif self.version == "gemv":
                    q_linear_module = WQLinear_GEMV
                elif self.version == "marlin":
                    q_linear_module = WQLinear_Marlin
                elif self.version == "gemv_fast":
                    q_linear_module = WQLinear_GEMVFast
                else:
                    raise ValueError(f"Unknown version {self.version}")

                q_linear = q_linear_module.from_linear(
                    linear=linear_layer,
                    w_bit=self.w_bit,
                    group_size=self.group_size,
                    init_only=False,
                    scales=scales,
                    zeros=zeros,
                )

                set_op_by_name(module, name, q_linear)

            linear_layer.cpu()
            clear_memory()

    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        if self.n_parallel_calib_samples is None:
            # runs through all samples at once
            module_output = module(x, **module_kwargs)
            if isinstance(module_output, tuple):
                module_output = module_output[0]
        else:
            # memory efficiently runs through all calibration samples
            # but only n_parallel_calib_samples at a time
            module_output = []
            partitioned_inputs = torch.split(x, self.n_parallel_calib_samples)
            for x_partial in partitioned_inputs:
                partial_output = module(x_partial, **module_kwargs)

                if isinstance(partial_output, tuple):
                    partial_output = partial_output[0]

                module_output.append(partial_output.cpu())

            module_output = torch.cat(module_output, dim=0)

        return module_output

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )

    def _compute_best_scale(
        self,
        x: torch.Tensor,
        w_mean: torch.Tensor,
        x_mean: torch.Tensor,
        module2inspect: torch.nn.Module,
        linears2scale: List[nn.Linear],
        fp16_output: torch.Tensor,
        kwargs: Dict={},
    ):
        """
        Compute loss and select best scales

        L(s) = || Q(W * s) (s^-1 * X) - W * X ||
        Q: weight quantization function | pseudo_quantize_tensor(W * s)
        X: inputs from calib dataset    | X
        W: original weights in FP16     | layer
        s: per channel scaling factor   | s^-1 * X
        """
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_mean = x_mean.view(-1).to(device)
        w_mean = w_mean.view(-1).to(device)

        for ratio in range(n_grid):
            # create new scales
            ratio = ratio / n_grid

            # NOTE: s^-1 * x is fused here, according to paper
            if self.duo_scaling:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # avoid scaling values that overflow
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = (
                    self.pseudo_quantize_tensor(fc.weight.data)[0] / scales_view
                )

            # W * X
            int_w_output = self._module_forward(x, module2inspect, kwargs)

            # compute mean squared error (L2 norm)
            loss = self._compute_loss(fp16_output, int_w_output, device)

            history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1:
            logging.debug(history)
            raise Exception

        assert torch.isnan(best_scales).sum() == 0, best_scales

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _compute_loss(
        self,
        fp16_output: torch.Tensor,
        int_w_output: torch.Tensor,
        device: torch.device,
    ):
        loss = 0.0
        fp16_output_flat = fp16_output.view(-1)
        int_w_output_flat = int_w_output.view(-1)
        num_elements = fp16_output_flat.size(0)
        element_size_bytes = fp16_output.element_size()

        # Calculate chunk size dynamically based on max_chunk_memory
        # Divide the max_chunk_memory by twice the element size
        chunk_size = self.max_chunk_memory // (element_size_bytes * 2)
        chunk_size = min(chunk_size, num_elements)

        # Split the computation into chunks
        fp16_chunks = torch.split(fp16_output_flat, chunk_size)
        int_w_chunks = torch.split(int_w_output_flat, chunk_size)

        # Compute the loss for each chunk
        for fp16_chunk, int_w_chunk in zip(fp16_chunks, int_w_chunks):
            chunk_loss = (fp16_chunk.to(device) - int_w_chunk.to(device)).float().pow(2).sum().item()
            loss += chunk_loss

        # Normalize the loss by the total number of elements
        loss /= num_elements

        return loss

    @torch.no_grad()
    def _search_best_clip(self, layer, named_linears, input_feat):
        clip_list = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(get_best_device())
            max_val = self._compute_best_clip(
                named_linears[name].weight, input_feat[name]
            )
            clip_list.append((name, max_val))
            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(
        self,
        w: torch.Tensor,
        input_feat: torch.Tensor,
        n_grid=20,
        max_shrink=0.5,
        n_sample_token=512,
    ):
        assert w.dim() == 2
        org_w_shape = w.shape
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else org_w_shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)

        # Compute input feature step size (minimum 1)
        step_size = max(1, input_feat.shape[1] // n_sample_token)
        input_feat = input_feat[:, ::step_size]
        
        w = w.reshape(org_w_shape[0], 1, -1, group_size)

        oc_batch_size = 256 if org_w_shape[0] % 256 == 0 else 64  # prevent OOM
        assert org_w_shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(org_w_shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = self.pseudo_quantize_tensor(cur_w)[0]
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs

    def _analyze_scores(self, scores):
        """
        Memory-efficient analysis of critical score distribution
        """
        print("\n=== Critical Score Analysis ===")

        try:
            total_weights = sum(score_tensor.numel() for score_tensor in scores.values())
            print(f"Total weights: {total_weights:,}")
            
            global_min = float('inf')
            global_max = float('-inf')

            for score_tensor in scores.values():
                global_min = min(global_min, score_tensor.min().item())
                global_max = max(global_max, score_tensor.max().item())
            
            print(f"Min score: {global_min:.6f}")
            print(f"Max score: {global_max:.6f}")
            
            running_sum = 0.0
            for score_tensor in scores.values():
                running_sum += score_tensor.sum().item()
            mean_score = running_sum / total_weights
            print(f"Mean score: {mean_score:.6f}")
            
            zero_count = 0
            near_zero_count = 0
            for score_tensor in scores.values():
                zero_count += (score_tensor == 0).sum().item()
                near_zero_count += (score_tensor.abs() < 1e-6).sum().item()
            
            print(f"Zero values: {zero_count:,} ({100*zero_count/total_weights:.2f}%)")
            print(f"Near-zero (<1e-6): {near_zero_count:,} ({100*near_zero_count/total_weights:.2f}%)")
            
            print("\n=== Creating Distribution Plot ===")
            sample_size = min(10_000_000, total_weights)
            print(f"Sampling {sample_size:,} weights for visualization...")
            
            sampled_scores = []
            seen = 0

            for score_tensor in scores.values():
                scores_flat = score_tensor.view(-1)
                layer_size = scores_flat.numel()
                
                if seen + layer_size <= sample_size:
                    sampled_scores.append(scores_flat.cpu())
                    seen += layer_size
                else:
                    remaining = sample_size - seen
                    if remaining > 0:
                        indices = torch.randperm(layer_size)[:remaining]
                        sampled_scores.append(scores_flat[indices].cpu())
                        seen = sample_size
                        break
            
            sampled_scores = torch.cat(sampled_scores)
            
            print(f"Computing percentiles on {sampled_scores.numel():,} samples...")
            percentiles = [10, 25, 50, 75, 90, 95, 99, 99.9]
            for p in percentiles:
                val = torch.quantile(sampled_scores.float(), p/100)
                print(f"{p}th percentile: {val.item():.6f}")
            
            plt.figure(figsize=(12, 4))
            
            scores_np = sampled_scores.numpy()
            
            plt.subplot(1, 3, 1)
            plt.hist(scores_np, bins=100, edgecolor='black', alpha=0.7)
            plt.xlabel('Critical Score')
            plt.ylabel('Frequency')
            plt.title(f'Critical Score Distribution (n={len(scores_np):,})')
            plt.yscale('log')
            
            plt.subplot(1, 3, 2)
            plt.hist(scores_np, bins=100, edgecolor='black', cumulative=True, 
                    density=True, alpha=0.7, color='green')
            plt.xlabel('Critical Score')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            non_zero_scores = scores_np[scores_np > 0]
            if len(non_zero_scores) > 0:
                plt.hist(non_zero_scores, bins=100, edgecolor='black', alpha=0.7, color='orange')
                plt.xlabel('Critical Score')
                plt.ylabel('Frequency')
                plt.title('Non-Zero Scores Only')
                plt.xscale('log')
                plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('score_distribution.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved as 'score_distribution.png'")
            plt.close()
            
            print("\n=== Top 10 Layers by Mean Score ===")
            layer_stats = []
            for name, score_tensor in scores.items():
                layer_stats.append({
                    'name': name,
                    'max': score_tensor.max().item(),
                    'mean': score_tensor.mean().item(),
                })
            
            layer_stats.sort(key=lambda x: x['mean'], reverse=True)
            for i, stat in enumerate(layer_stats[:10]):
                print(f"{i+1}. {stat['name']}")
                print(f"   Max: {stat['max']:.6f}, Mean: {stat['mean']:.6f}")
            
        
            del sampled_scores
            clear_memory()
        except Exception as e:
            print(f"Error during score analysis: {e}")

        return torch.cat([score_tensor.view(-1) for score_tensor in scores.values()])

    def _save_scores(self, scores: Dict[str, torch.Tensor], filename: str):
        """
        Save critical scores to a file for later analysis or fasten future runs.
        """
        os.makedirs(self.saved_scores_dir, exist_ok=True)
        save_dict = {
            'scores': {k: v.cpu() for k, v in scores.items()},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'beta': self.beta,
            }
        }

        torch.save(save_dict, os.path.join(self.saved_scores_dir, filename))
        print(f"Importance scores saved to {filename}")
    
    def _load_scores(self, filename: str) -> Dict[str, torch.Tensor]:
        """
        Load importance scores from a file if available.
        """
        filepath = os.path.join(self.saved_scores_dir, filename)
        if os.path.exists(filepath):
            loaded = torch.load(filepath)
            print(f"Loaded importance scores from {filename}")
            return loaded['scores']
        else:
            print(f"No saved scores found at {filename}")
            return {}