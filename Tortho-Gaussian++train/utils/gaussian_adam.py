import torch


class RowWiseGaussianAdam(torch.optim.Adam):
    """Adam with per-row bias-correction ages for Gaussian parameters."""

    birth_step_key = "birth_step"

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=None,
        max_steps=30_000,
    ):
        if amsgrad:
            raise ValueError("RowWiseGaussianAdam does not support AMSGrad")
        if foreach:
            raise ValueError("RowWiseGaussianAdam does not support foreach=True")
        if capturable:
            raise ValueError("RowWiseGaussianAdam does not support capturable=True")
        if differentiable:
            raise ValueError("RowWiseGaussianAdam does not support differentiable=True")
        if fused:
            raise ValueError("RowWiseGaussianAdam does not support fused=True")
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=False,
            foreach=False,
            maximize=maximize,
            capturable=False,
            differentiable=False,
            fused=False,
        )
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("RowWiseGaussianAdam max_steps must be positive")
        self._bias_correction_tables = {}

    @staticmethod
    def _initialize_state(param, state):
        if param.ndim == 0:
            raise ValueError("RowWiseGaussianAdam parameters must have a Gaussian row dimension")
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state[RowWiseGaussianAdam.birth_step_key] = torch.zeros(
            param.shape[0], dtype=torch.int32, device=param.device
        )

    @classmethod
    def _validate_state(cls, param, state):
        required_keys = ("step", "exp_avg", "exp_avg_sq", cls.birth_step_key)
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise ValueError(
                "RowWiseGaussianAdam state is missing {}. "
                "A checkpoint created by the original Adam cannot recover historical Gaussian birth steps."
                .format(", ".join(missing_keys))
            )

        birth_step = state[cls.birth_step_key]
        if birth_step.dtype != torch.int32:
            raise ValueError("RowWiseGaussianAdam birth_step must use torch.int32")
        if birth_step.device != param.device:
            raise ValueError("RowWiseGaussianAdam birth_step must be on the parameter device")
        if birth_step.ndim != 1 or birth_step.shape[0] != param.shape[0]:
            raise ValueError(
                "RowWiseGaussianAdam birth_step shape {} does not match parameter row count {}"
                .format(tuple(birth_step.shape), param.shape[0])
            )
        if state["exp_avg"].shape != param.shape or state["exp_avg_sq"].shape != param.shape:
            raise ValueError("RowWiseGaussianAdam moment tensors must match the parameter shape")

    def load_state_dict(self, state_dict):
        saved_birth_steps = []
        saved_state = state_dict.get("state", {})
        for saved_group in state_dict.get("param_groups", []):
            group_birth_steps = []
            for param_id in saved_group["params"]:
                param_state = saved_state.get(param_id, {})
                group_birth_steps.append(param_state.get(self.birth_step_key))
            saved_birth_steps.append(group_birth_steps)

        result = super().load_state_dict(state_dict)
        for group, group_birth_steps in zip(self.param_groups, saved_birth_steps):
            for param, birth_step in zip(group["params"], group_birth_steps):
                state = self.state.get(param)
                if state:
                    if birth_step is not None:
                        state[self.birth_step_key] = birth_step.detach().to(
                            device=param.device, dtype=torch.int32
                        )
                    self._validate_state(param, state)
        return result

    def _get_bias_correction_tables(self, param, beta1, beta2, required_step):
        device_key = (param.device.type, param.device.index)
        key = (device_key, param.dtype, beta1, beta2)
        tables = self._bias_correction_tables.get(key)
        required_size = max(self.max_steps, int(required_step)) + 1
        if tables is None or tables[0].shape[0] < required_size:
            steps = torch.arange(required_size, dtype=param.dtype, device=param.device)
            bias_correction1 = 1.0 - torch.pow(beta1, steps)
            bias_correction2_sqrt = torch.sqrt(1.0 - torch.pow(beta2, steps))
            tables = (bias_correction1, bias_correction2_sqrt)
            self._bias_correction_tables[key] = tables
        return tables

    @torch.no_grad()
    def step(self, closure=None):
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["amsgrad"] or group["capturable"] or group["differentiable"] or group["fused"]:
                raise ValueError("Unsupported Adam mode loaded into RowWiseGaussianAdam")
            if group["foreach"]:
                raise ValueError("RowWiseGaussianAdam does not support foreach=True")

            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("RowWiseGaussianAdam does not support sparse gradients")
                if torch.is_complex(param):
                    raise RuntimeError("RowWiseGaussianAdam does not support complex parameters")

                state = self.state[param]
                if len(state) == 0:
                    self._initialize_state(param, state)
                else:
                    self._validate_state(param, state)

                grad = -param.grad if group["maximize"] else param.grad
                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                state["step"].add_(1)
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["lr"] == 0:
                    continue

                step = int(state["step"].item())
                effective_step = step - state[self.birth_step_key]
                bias_correction1_table, bias_correction2_sqrt_table = self._get_bias_correction_tables(
                    param, beta1, beta2, step
                )
                bias_correction1 = bias_correction1_table[effective_step]
                bias_correction2_sqrt = bias_correction2_sqrt_table[effective_step]

                broadcast_shape = (param.shape[0],) + (1,) * (param.ndim - 1)
                bias_correction1 = bias_correction1.view(broadcast_shape)
                bias_correction2_sqrt = bias_correction2_sqrt.view(broadcast_shape)
                step_size_neg = -group["lr"] / bias_correction1

                denom = exp_avg_sq.sqrt()
                denom.div_(bias_correction2_sqrt * step_size_neg)
                denom.add_(group["eps"] / step_size_neg)
                param.addcdiv_(exp_avg, denom)

        return loss
