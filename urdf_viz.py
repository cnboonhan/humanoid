# python3 urdf_viz.py --path .venv/lib/python3.11/site-packages/mani_skill/assets/robots/g1_humanoid/g1.urdf  --port 8081
from __future__ import annotations

import time

import numpy as np
import tyro
from yourdfpy import URDF

import viser
from viser.extras import ViserUrdf


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(path: str, port: int) -> None:

    server = viser.ViserServer(port=port)

    urdf = URDF.load(path, load_collision_meshes=True, build_collision_scene_graph=True)

    # Print all links information
    print(f"Robot name: {urdf.robot}")
    print(f"Number of joints: {len(urdf.joint_map)}")
    print(f"Number of links: {len(urdf.link_map)}")
    print("\nAll links:")
    for i, (link_name, link) in enumerate(urdf.link_map.items()):
        print(f"  {i+1:2d}. {link_name}")
        if link.inertial:
            print(f"      Mass: {link.inertial.mass:.4f} kg")
        if link.visuals:
            print(f"      Visuals: {len(link.visuals)}")
        if link.collisions:
            print(f"      Collisions: {len(link.collisions)}")
    
    print(f"\nActuated joints ({len([j for j in urdf.joint_map.values() if j.type != 'fixed'])}):")
    for i, (joint_name, joint) in enumerate(urdf.joint_map.items()):
        if joint.type != 'fixed':
            print(f"  {i+1:2d}. {joint_name} ({joint.type})")
            if joint.limit:
                print(f"      Limits: [{joint.limit.lower:.3f}, {joint.limit.upper:.3f}]")

    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=True,
        load_collision_meshes=True,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    show_meshes_cb.visible = True 
    show_collision_meshes_cb.visible = True 

    viser_urdf.update_cfg(np.array(initial_config))

    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)