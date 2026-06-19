"""Bootstrap script passed to `isaac-sim.streaming.sh --exec` so the Kit
app auto-opens the cracked-scene USD after its extensions load. Without
this Kit's streaming experience starts with an empty stage and requires
the user to File > Open manually."""

import asyncio
import omni.kit.app
import omni.usd

_USD_PATH = (
    "/home/thescript/isaac-sim-msoe/assets/environments/"
    "InnovationCenter/InnovationCenter_cracked.usd"
)


async def _open_after_boot():
    # Wait a few frames so core extensions + the viewport are fully
    # registered — opening the stage during the very first update tick
    # sometimes races the renderer and the building shows up invisible.
    app = omni.kit.app.get_app()
    for _ in range(60):
        await app.next_update_async()
    ctx = omni.usd.get_context()
    print(f"[auto-open] opening {_USD_PATH}", flush=True)
    ctx.open_stage(_USD_PATH)
    print("[auto-open] stage open request issued", flush=True)


asyncio.ensure_future(_open_after_boot())
