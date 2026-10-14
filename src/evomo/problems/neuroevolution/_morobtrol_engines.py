"""Environment construction for the shared MoRobtrol evaluator."""

# XML asset, control substeps, objective count. Keep task discovery dependency-free.
PLAYGROUND_TASKS = {
    "mo_swimmer": ("swimmer", 4, 2),
    "mo_halfcheetah": ("half_cheetah", 5, 2),
    "mo_hopper_m2": ("hopper", 4, 2),
    "mo_hopper_m3": ("hopper", 4, 3),
}


def make_environment(name, engine, backend, *, objectives=None, env_config=None, observation_key=None):
    native_options = objectives is not None or env_config is not None or observation_key is not None
    if engine == "brax":
        if native_options:
            raise ValueError("objectives, env_config and observation_key are only supported for native Playground tasks")
        from brax import envs

        if name == "mo_swimmer" and backend == "mjx":
            raise ValueError("Brax Swimmer does not support backend='mjx'; use engine='playground' for this task")
        return envs.get_environment(env_name=name, **({} if backend is None else {"backend": backend}))
    if engine != "playground":
        raise ValueError("engine must be 'brax' or 'playground'")
    if backend not in (None, "mjx"):
        raise ValueError("engine='playground' only supports backend='mjx'")
    try:
        if name in PLAYGROUND_TASKS:
            if native_options:
                raise ValueError("Legacy mo_* ports have fixed task definitions; native Playground options do not apply")
            from ._playground_env import MoPlayground

            return MoPlayground(name)
        from ._playground_native import NativePlayground

        return NativePlayground(name, objectives=objectives, env_config=env_config, observation_key=observation_key)
    except ModuleNotFoundError as exc:
        raise ImportError("Playground support requires optional dependencies: pip install 'evomo[playground]'") from exc


def available_playground_tasks():
    try:
        from mujoco_playground import registry
    except ModuleNotFoundError as exc:
        raise ImportError("Install optional dependencies with pip install 'evomo[playground]'") from exc
    return tuple(PLAYGROUND_TASKS) + tuple(registry.ALL_ENVS)


def frames_to_html(frames):
    """Self-contained frame viewer; no video encoder or external assets required."""
    import base64
    import io
    import json
    import uuid

    from PIL import Image

    encoded = []
    for frame in frames:
        buffer = io.BytesIO()
        Image.fromarray(frame).save(buffer, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"))
    identifier = "morobtrol_" + uuid.uuid4().hex
    return (
        f'<div id="{identifier}"><img alt="MoRobtrol rollout" src="{encoded[0]}"/>'
        f'<br/><input aria-label="Frame" type="range" min="0" max="{len(encoded) - 1}" value="0"/></div>'
        f'<script>(() => {{const root=document.getElementById("{identifier}");const frames={json.dumps(encoded)};'
        'root.querySelector("input").oninput=e=>{root.querySelector("img").src=frames[Number(e.target.value)];};})();</script>'
    )
