"""Linear conditional Flow Matching path and Euler ODE sampler."""

import torch
from torch.nn import functional as F


class FlowMatchingPipeline:
    """Transport standard Gaussian noise ``z`` to a data image ``x1``.

    x_t = (1 - t) z + t x1
    u_t = dx_t / dt = x1 - z
    """

    def __init__(self, time_scale=1000.0):
        self.time_scale = time_scale

    def model_time(self, t):
        """Map continuous flow time [0,1] to the U-Net embedding scale."""
        return t * self.time_scale

    @staticmethod
    def interpolate(z, x1, t):
        """Return x_t for ``z/x1: (B,C,H,W)`` and ``t: (B,)``."""
        t_view = t[:, None, None, None]
        return (1.0 - t_view) * z + t_view * x1

    def make_training_pair(self, x1):
        batch_size = x1.shape[0]
        t = torch.rand(batch_size, device=x1.device)
        z = torch.randn_like(x1)
        x_t = self.interpolate(z, x1, t)
        target_velocity = x1 - z
        return x_t, t, target_velocity

    def training_loss(self, model, x1):
        x_t, t, target_velocity = self.make_training_pair(x1)
        predicted_velocity = model(x_t, self.model_time(t))
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss, {"t": t.detach().mean()}

    @torch.no_grad()
    def sample(self, model, batch_size, image_shape, steps, device):
        """Euler integration from x(0) ~ N(0,I) to an image estimate x(1)."""
        if steps <= 0:
            raise ValueError("steps must be positive.")

        x = torch.randn(batch_size, *image_shape, device=device)
        dt = 1.0 / steps

        for step in range(steps):
            current_t = step / steps
            t = torch.full((batch_size,), current_t, device=device)
            velocity = model(x, self.model_time(t))
            x = x + dt * velocity

        return x


if __name__ == "__main__":
    pipeline = FlowMatchingPipeline()
    x1 = torch.randn(2, 3, 32, 32)
    z = torch.randn_like(x1)
    batch_size = x1.shape[0]

    assert torch.allclose(pipeline.interpolate(z, x1, torch.zeros(batch_size)), z)
    assert torch.allclose(pipeline.interpolate(z, x1, torch.ones(batch_size)), x1)
    print("Flow path boundary checks passed.")
