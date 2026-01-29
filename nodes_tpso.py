import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules import devices

class TPSONode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "p": ("PROCESSING",),
                "tpso_enabled": ("BOOLEAN", {"default": False}),
                "tpso_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "tpso_lr": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.001}),
                "tpso_lambda": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "tpso_r": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "tpso_kappa": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 0.99, "step": 0.01}),
                "tpso_use_alpha": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model_patches"
    DESCRIPTION = "Training-Free Prompt Semantic Space Optimization (TPSO)"

    def patch(self, unet, p, tpso_enabled=False, tpso_steps=20, tpso_lr=0.1, tpso_lambda=1.0, tpso_r=0.4, tpso_kappa=0.8, tpso_use_alpha=True):
        if not tpso_enabled:
            return (unet,)

        # Ensure boolean
        if isinstance(tpso_use_alpha, str):
            tpso_use_alpha = tpso_use_alpha.lower() in ("true", "1", "yes", "on")

        # LR Heuristic
        real_lr = tpso_lr
        if real_lr < 0.05:
            real_lr = 0.1

        logging.info(f"TPSO: Starting Optimization (Target Kappa: {tpso_kappa}, Steps: {tpso_steps}, LR: {real_lr}, Alpha Scheduler: {tpso_use_alpha})...")
        
        with torch.no_grad():
            original_cond_obj = p.sd_model.get_learned_conditioning(p.prompts)
        
        device = devices.device
        dtype = devices.dtype_unet
        
        original_cond_tensor = None
        if isinstance(original_cond_obj, dict):
            for k in ['crossattn', 'c_crossattn', 'cond']:
                if k in original_cond_obj:
                    original_cond_tensor = original_cond_obj[k]
                    break
            if original_cond_tensor is None:
                for k, v in original_cond_obj.items():
                    if isinstance(v, torch.Tensor) and v.ndim == 3:
                        original_cond_tensor = v
                        break
        else:
            original_cond_tensor = original_cond_obj

        if original_cond_tensor is None or not isinstance(original_cond_tensor, torch.Tensor):
            logging.error("TPSO: Could not extract conditioning tensor!")
            return (unet,)

        batch_size = original_cond_tensor.shape[0]
        
        # --- OPTIMIZATION BLOCK ---
        with torch.inference_mode(False):
            with torch.enable_grad():
                target_cond_fp32 = original_cond_tensor.to(device, dtype=torch.float32).detach()
                epsilon = torch.randn_like(target_cond_fp32) * 1e-2
                epsilon.requires_grad_(True)
                optimizer = optim.Adam([epsilon], lr=real_lr)
                
                kappa = tpso_kappa
                sigma = 0.01
                
                for step in range(tpso_steps):
                    optimizer.zero_grad()
                    optimized_cond = target_cond_fp32 + epsilon
                    
                    v_prime = optimized_cond.view(batch_size, -1)
                    v = target_cond_fp32.view(batch_size, -1)
                    v_prime_norm = F.normalize(v_prime, p=2, dim=1)
                    v_norm = F.normalize(v, p=2, dim=1)
                    cos_sim = (v_prime_norm * v_norm).sum(dim=1)
                    
                    diff = torch.abs(cos_sim - kappa) - sigma
                    l_semantic = torch.clamp(diff, min=0.0).sum()
                    
                    l_div = torch.tensor(0.0, device=device, dtype=torch.float32)
                    if batch_size > 1:
                        sim_matrix = torch.mm(v_prime_norm, v_prime_norm.t())
                        mask = torch.eye(batch_size, device=device).bool()
                        off_diag = sim_matrix[~mask]
                        if off_diag.numel() > 0:
                            l_div = off_diag.mean()
                    
                    loss = l_semantic + tpso_lambda * l_div
                    loss.backward()
                    optimizer.step()
                    
                    if step == 0 or step == tpso_steps - 1:
                         logging.debug(f"TPSO Step {step}: Loss={loss.item():.4f}, CosSim={cos_sim.mean().item():.4f}")

                final_optimized_cond = (target_cond_fp32 + epsilon).to(dtype).detach()

        with torch.no_grad():
            final_v_prime = F.normalize(final_optimized_cond.view(batch_size, -1).float(), p=2, dim=1)
            final_v = F.normalize(original_cond_tensor.view(batch_size, -1).float(), p=2, dim=1)
            actual_sim = (final_v_prime * final_v).sum(dim=1).mean().item()
        
        logging.info(f"TPSO: Optimization Finished. Final Cosine Sim: {actual_sim:.4f} (Target: {kappa})")

        # --- DYNAMIC WRAPPER ---
        # State to store the detected max_t (start sigma)
        wrapper_state = {"max_t": None}
        COND_INDEX = 0

        def unet_wrapper(apply_model, args):
            t = args["timestep"]
            t_curr = t[0].item() if torch.is_tensor(t) else t
            c = args["c"]
            
            # Dynamic Max T Detection (First Step)
            if wrapper_state["max_t"] is None:
                wrapper_state["max_t"] = float(t_curr)
                logging.info(f"TPSO: Detected Start Sigma/Timestep = {t_curr:.2f}")

            # Calculate threshold dynamically based on the REAL max_t
            # threshold = start_t * (1 - r). 
            # E.g. if start=14.6, r=0.4 => threshold = 14.6 * 0.6 = 8.76. 
            # All steps with t > 8.76 will get injected.
            max_t = wrapper_state["max_t"]
            threshold = max_t * (1.0 - tpso_r)
            
            # FORCE DEBUG LOG ONCE
            if not hasattr(unet_wrapper, "has_logged"):
                logging.warning(f"TPSO DEBUG: Wrapper Called! t={t_curr:.2f}. Dynamic Threshold={threshold:.2f} (Max={max_t:.2f})")
                unet_wrapper.has_logged = True

            # Injection Logic
            if t_curr > threshold:
                cond_or_uncond = args.get("cond_or_uncond", [])
                cond_indices = [i for i, x in enumerate(cond_or_uncond) if x == COND_INDEX]
                
                if cond_indices:
                    # Calculate alpha for interpolation (Eq 11 from paper)
                    # alpha = (t - T(1-r)) / (rT) -> (t - threshold) / (max_t - threshold)
                    denom = max_t - threshold
                    alpha = 1.0
                    if denom > 1e-5:
                        alpha = (t_curr - threshold) / denom
                        alpha = max(0.0, min(1.0, alpha))
                    
                    new_c = c.copy()
                    target_keys = ["c_crossattn", "crossattn"]
                    injected = False
                    
                    for key in target_keys:
                        if key in c and isinstance(c[key], torch.Tensor):
                            current_emb = c[key]
                            if current_emb.shape[-1] == final_optimized_cond.shape[-1]:
                                new_emb = current_emb.clone()
                                for i, idx in enumerate(cond_indices):
                                    if idx < current_emb.shape[0]:
                                        opt_idx = i % final_optimized_cond.shape[0]
                                        optimized_val = final_optimized_cond[opt_idx]
                                        
                                        if tpso_use_alpha:
                                            # Interpolate: y* = alpha * y' + (1 - alpha) * y
                                            original_val = current_emb[idx]
                                            new_emb[idx] = alpha * optimized_val + (1.0 - alpha) * original_val
                                        else:
                                            # Direct replacement (User's request)
                                            new_emb[idx] = optimized_val

                                new_c[key] = new_emb
                                injected = True
                                
                                if not hasattr(unet_wrapper, "injected_logged"):
                                    logging.warning(f"TPSO DEBUG: INJECTED into {key} at t={t_curr:.2f} with alpha={alpha:.4f} (Interpolation: {tpso_use_alpha})")
                                    unet_wrapper.injected_logged = True
                    
                    if injected:
                        return apply_model(args["input"], args["timestep"], **new_c)

            return apply_model(args["input"], args["timestep"], **args["c"])

        patched_unet = unet.clone()
        patched_unet.set_model_unet_function_wrapper(unet_wrapper)
        return (patched_unet,)

NODE_CLASS_MAPPINGS = {"TPSONode": TPSONode}
NODE_DISPLAY_NAME_MAPPINGS = {"TPSONode": "TPSO Guidance"}
