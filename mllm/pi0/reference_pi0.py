"""Original pi0 mechanisms. Time follows openpi: t=1 noise -> t=0 actions."""
import torch
from types import SimpleNamespace
from edu_core.backbones import PrefixEncoding
from edu_core.backbones.layers import projected_attention


def action_chunk(actions, now, horizon=8):
    indices = torch.arange(now, now+horizon, device=actions.device).clamp(max=len(actions)-1)
    return actions[indices]


def normalize_values(values, low, high):
    return (2*(values-low)/(high-low).clamp_min(1e-6)-1).clamp(-1, 1)


def unnormalize_values(values, low, high):
    return (values+1)*.5*(high-low)+low


def embed_prefix(model, images, camera_valid, text_ids, text_valid):
    b, cameras, c, h, w = images.shape
    features = model.backbone.encode_vision(images.reshape(b*cameras, c, h, w), projected=True)
    per_image = features.tokens.shape[1]
    visual = features.tokens.reshape(b, cameras*per_image, -1)
    visual_valid = (features.valid_mask.reshape(b, cameras, per_image) & camera_valid[:, :, None]).flatten(1)
    return torch.cat((visual, model.backbone.embed_text(text_ids)), 1), torch.cat((visual_valid, text_valid), 1)


def time_embedding(time, dim):
    fraction = torch.linspace(0, 1, dim//2, device=time.device)
    period = .004*(4./.004)**fraction
    phase = time[:, None]/period[None]*(2*torch.pi)
    return torch.cat((phase.sin(), phase.cos()), -1)


def embed_suffix(model, state, noisy_actions, time):
    state_token = model.state_proj(state)[:, None]
    action_tokens = model.action_in(noisy_actions)
    time_tokens = model.ops.time_embedding(time, model.expert_dim)[:, None].expand_as(action_tokens)
    actions = model.action_time(torch.cat((action_tokens, time_tokens), -1))
    return torch.cat((state_token, actions), 1)


def block_mask(valid, prefix_length):
    length = valid.shape[1]
    groups = torch.zeros(length, device=valid.device, dtype=torch.long)
    groups[prefix_length] = 1  # proprioceptive state is a separate block
    groups[prefix_length+1:] = 2  # actions see one another bidirectionally
    return (groups[None, :] <= groups[:, None])[None] & valid[:, :, None] & valid[:, None, :]


def position_ids(valid):
    return (valid.long().cumsum(-1)-1).clamp_min(0)


def joint_attention(prefix_layer, expert_layer, prefix, suffix, positions, allow):
    p = prefix.shape[1]
    pq, pk, pv = prefix_layer.project(prefix, positions[:, :p])
    sq, sk, sv = expert_layer.project(suffix, positions[:, p:])
    attended = projected_attention(torch.cat((pq, sq), 2), torch.cat((pk, sk), 2), torch.cat((pv, sv), 2), allow)
    return prefix_layer.finish(prefix, attended[:, :p]), expert_layer.finish(suffix, attended[:, p:])


def joint_forward(model, prefix, prefix_valid, suffix):
    valid = torch.cat((prefix_valid, torch.ones(suffix.shape[:2], dtype=torch.bool, device=suffix.device)), 1)
    allow = model.ops.block_mask(valid, prefix.shape[1])
    positions = model.ops.position_ids(valid)
    for p_layer, a_layer in zip(model.backbone.layers, model.expert_layers):
        prefix, suffix = model.ops.joint_attention(p_layer, a_layer, prefix, suffix, positions, allow)
    return model.action_out(model.expert_norm(suffix[:, 1:]))


def suffix_attention(expert_layer, suffix, positions, prefix_kv, allow):
    q, k, v = expert_layer.project(suffix, positions)
    prefix_k, prefix_v = prefix_kv
    attended = projected_attention(q, torch.cat((prefix_k, k), 2), torch.cat((prefix_v, v), 2), allow)
    return expert_layer.finish(suffix, attended)


def cached_forward(model, cache, suffix):
    valid = torch.ones(suffix.shape[:2], dtype=torch.bool, device=suffix.device)
    suffix_allow = model.ops.block_mask(valid, 0)
    prefix_allow = cache.valid_mask[:, None, :].expand(-1, suffix.shape[1], -1)
    allow = torch.cat((prefix_allow, suffix_allow), -1)
    positions = cache.valid_mask.sum(-1)[:, None]+model.ops.position_ids(valid)
    for layer, kv in zip(model.expert_layers, cache.key_values):
        suffix = model.ops.suffix_attention(layer, suffix, positions, kv, allow)
    return model.action_out(model.expert_norm(suffix[:, 1:]))


def sample_time(batch_size, device):
    alpha = torch.tensor(1.5, device=device)
    beta = torch.tensor(1., device=device)
    return torch.distributions.Beta(alpha, beta).sample((batch_size,))*.999+.001


def flow_pair(actions, noise, time):
    t = time[:, None, None]
    return t*noise+(1-t)*actions, noise-actions


def flow_loss(prediction, target):
    return (prediction-target).square().mean()


def training_step(model, batch, optimizer, noise=None, time=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch, noise=noise, time=time)
    loss.backward()
    optimizer.step()
    return loss.detach()


def cache_prefix(model, prefix, valid):
    allow = valid[:, :, None] & valid[:, None, :]
    positions = model.ops.position_ids(valid)
    output = model.backbone.language(prefix, valid, allow_mask=allow, positions=positions,
                                     use_cache=True, output_logits=False)
    return PrefixEncoding(output.key_values, valid, positions)


def euler_step(actions, velocity, step_size):
    return actions+step_size*velocity


def sample_actions(model, batch, initial_noise, steps=10):
    prefix, valid = model.ops.embed_prefix(model, batch["images"], batch["camera_valid"], batch["text_ids"], batch["text_valid"])
    cache = model.ops.cache_prefix(model, prefix, valid)
    state = model.ops.normalize_values(batch["state"], model.state_low, model.state_high)
    x = initial_noise.clone()
    for i in range(steps):
        time = x.new_full((len(x),), 1-i/steps)
        suffix = model.ops.embed_suffix(model, state, x, time)
        velocity = model.ops.cached_forward(model, cache, suffix)
        x = model.ops.euler_step(x, velocity, -1./steps)
    return model.ops.unnormalize_values(x, model.action_low, model.action_high)


def execution_chunk(prediction, steps=3):
    return prediction[:, :steps]


def receding_control(model, observe, execute, noises):
    chunks = []
    for noise in noises:
        observation = observe()
        prediction = model.predict_action(observation, noise)
        chunk = model.ops.execution_chunk(prediction)
        execute(chunk)
        chunks.append(chunk)
    return chunks


def reference_ops():
    names = ("action_chunk", "normalize_values", "unnormalize_values", "embed_prefix", "time_embedding", "embed_suffix",
             "block_mask", "position_ids", "joint_attention", "joint_forward", "suffix_attention", "cached_forward",
             "sample_time", "flow_pair", "flow_loss", "training_step", "cache_prefix", "euler_step", "sample_actions",
             "execution_chunk", "receding_control")
    return SimpleNamespace(**{name: globals()[name] for name in names})


from scaffolding import Pi0Policy


def build_toy_pi0() -> Pi0Policy:
    return Pi0Policy()
