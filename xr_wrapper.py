import xr
import numpy as np
import ctypes
import time

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from typing import NamedTuple


VERTEX_SHADER_SRC = """
    #version 330 core
    layout (location = 0) in vec2 position;
    layout (location = 1) in vec2 texCoords;

    out vec2 TexCoords;

    void main() {
        gl_Position = vec4(position.x, position.y, 0.0, 1.0);
        TexCoords = texCoords;
    }
    """

FRAGMENT_SHADER_SRC = """
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoords;

    uniform sampler2D screenTexture;

    void main() {
        FragColor = texture(screenTexture, TexCoords);
    }
    """


class Position(NamedTuple):
    x: float
    y: float
    z: float


class Orientation(NamedTuple):
    w: float
    x: float
    y: float
    z: float


class Pose(NamedTuple):
    position: Position
    orientation: Orientation


class FullPose(NamedTuple):
    head: Pose | None
    left_hand: Pose | None
    right_hand: Pose | None


def _round(x, precision=None):
    return round(x, precision) if precision else x


def xr_posef_to_world_pose(pose: xr.Posef, precision=None) -> Pose:
    """
        Note: by default we use Local Space reference (see openXR docs) and the unit of measure is meters
        pose is ovrPosef format: right-handed cartisiaan coordinate system,
        flat array of 7 floats as follows:
            1. ovrQuatf: x, y, z, w
            2. ovrVector3f: x, y, z
            (-x is forward, z is left, y is up)
    )
        In isaac world is (+Z up, +X forward), ros is (+Y up, +Z forward) and usd is (+Y up and -Z forward). Defaults to “world”.
        In isaac quaternion is w, x, y, z
    """
    # Old - for OVRLib conversion
    # qx_ovr, qy_ovr, qz_ovr, qw_ovr, px_ovr, py_ovr, pz_ovr = pose
    # orientation = np.array([qw_ovr, qz_ovr, -qx_ovr, qy_ovr])
    # pos = np.array([pz_ovr, -px_ovr, py_ovr])
    new_pose = Pose(
        position=Position(
            x=_round(pose.position.z, precision),
            y=_round(-pose.position.x, precision),
            z=_round(pose.position.y, precision),
        ),
        orientation=Orientation(
            w=_round(pose.orientation.w, precision),
            x=_round(pose.orientation.z, precision),
            y=_round(-pose.orientation.x, precision),
            z=_round(pose.orientation.y, precision),
        ),
    )
    return new_pose


class XrWrapper:
    def __init__(self, resolution):
        self.resolution = resolution
        self.context = None
        self.vao = None
        self.shader_program = None
        self.action_spaces = None
        self.found_count = None
        self.frame_state_generator = None
        self.textures = [None, None]

    def __enter__(self):
        self.context = xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            )
        )
        self.context.__enter__()
        self.vao, self.shader_program = self._init_graphics()
        self._init_textures()
        self.action_spaces = self._create_xr_controller_action_spaces()
        self.found_count = 0
        self.frame_state_generator = self.context.frame_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.context.__exit__(exc_type, exc_val, exc_tb)

    def _init_graphics(self):
        # Compile shaders and program.
        vertex_shader = compileShader(VERTEX_SHADER_SRC, GL_VERTEX_SHADER)
        fragment_shader = compileShader(FRAGMENT_SHADER_SRC, GL_FRAGMENT_SHADER)
        shader_program = compileProgram(vertex_shader, fragment_shader)

        # Define fullscreen quad.
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        # Generate VAO and VBO
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1,
            2,
            GL_FLOAT,
            GL_FALSE,
            4 * vertices.itemsize,
            ctypes.c_void_p(2 * vertices.itemsize),
        )
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        # Use the shader program.
        glUseProgram(shader_program)
        return vao, shader_program

    def _init_textures(self):
        """Initialize OpenGL textures."""
        for i in range(2):  # Assuming two eyes: left (0) and right (1)
            self.textures[i] = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.textures[i])
            # Set texture parameters here if needed, e.g., GL_LINEAR filtering.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            # Initialize texture with empty data
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                self.resolution[0],
                self.resolution[1],
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                None,
            )
            glBindTexture(GL_TEXTURE_2D, 0)

    def update_texture(self, texture_idx, rgb_array: np.array):
        """Update the OpenGL texture with new image data."""
        height, width, channels = rgb_array.shape
        glBindTexture(GL_TEXTURE_2D, self.textures[texture_idx])
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, rgb_array
        )
        # Unbind texture for safety
        glBindTexture(GL_TEXTURE_2D, 0)

    def _load_texture_from_rgb_array(self, rgb_array):
        """Load an image from bytes into an OpenGL texture."""
        height, width, _ = rgb_array.shape
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            rgb_array,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        return texture

    def _create_xr_controller_action_spaces(self) -> list[any]:
        """_summary_

        Args:
            context (xr.ContextObject): the XR context object

        Returns:
            list[any]: list of controller action space. Can be used to to retrieve controller states
        """
        controller_paths = (xr.Path * 2)(
            xr.string_to_path(self.context.instance, "/user/hand/left"),
            xr.string_to_path(self.context.instance, "/user/hand/right"),
        )
        controller_pose_action = xr.create_action(
            action_set=self.context.default_action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="hand_pose",
                localized_action_name="Hand Pose",
                count_subaction_paths=len(controller_paths),
                subaction_paths=controller_paths,
            ),
        )
        suggested_bindings = (xr.ActionSuggestedBinding * 2)(
            xr.ActionSuggestedBinding(
                action=controller_pose_action,
                binding=xr.string_to_path(
                    instance=self.context.instance,
                    path_string="/user/hand/left/input/grip/pose",
                ),
            ),
            xr.ActionSuggestedBinding(
                action=controller_pose_action,
                binding=xr.string_to_path(
                    instance=self.context.instance,
                    path_string="/user/hand/right/input/grip/pose",
                ),
            ),
        )
        xr.suggest_interaction_profile_bindings(
            instance=self.context.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self.context.instance,
                    "/interaction_profiles/khr/simple_controller",
                ),
                count_suggested_bindings=len(suggested_bindings),
                suggested_bindings=suggested_bindings,
            ),
        )
        xr.suggest_interaction_profile_bindings(
            instance=self.context.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self.context.instance,
                    "/interaction_profiles/htc/vive_controller",
                ),
                count_suggested_bindings=len(suggested_bindings),
                suggested_bindings=suggested_bindings,
            ),
        )
        action_spaces = [
            xr.create_action_space(
                session=self.context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=controller_pose_action,
                    subaction_path=controller_paths[0],
                ),
            ),
            xr.create_action_space(
                session=self.context.session,
                create_info=xr.ActionSpaceCreateInfo(
                    action=controller_pose_action,
                    subaction_path=controller_paths[1],
                ),
            ),
        ]
        return action_spaces

    def step(self, rgb_left, rgb_right) -> FullPose:
        head_pose, left_hand_pose, right_hand_pose = None, None, None
        try:
            frame_state = next(self.frame_state_generator)
            t = time.time()
            for view_index, view in enumerate(self.context.view_loop(frame_state)):
                # Clear the color and depth buffers.
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                if view_index == 0:
                    texture = self._load_texture_from_rgb_array(rgb_left)
                    # self.update_texture(texture_idx=0, rgb_array=rgb_left)
                    # glBindTexture(GL_TEXTURE_2D, self.textures[0])
                else:
                    texture = self._load_texture_from_rgb_array(rgb_right)
                    # self.update_texture(texture_idx=1, rgb_array=rgb_right)
                    # glBindTexture(GL_TEXTURE_2D, self.textures[1])
                # Bind the appropriate texture.
                glBindTexture(GL_TEXTURE_2D, texture)

                glUniform1i(
                    glGetUniformLocation(self.shader_program, "screenTexture"), 0
                )

                # Render the fullscreen quad.
                glBindVertexArray(self.vao)
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                glBindVertexArray(0)

                # Unbind the texture.
                glBindTexture(GL_TEXTURE_2D, 0)
            print(f"{time.time() - t}")

            if self.context.session_state == xr.SessionState.FOCUSED:
                active_action_set = xr.ActiveActionSet(
                    action_set=self.context.default_action_set,
                    subaction_path=xr.NULL_PATH,
                )
                xr.sync_actions(
                    session=self.context.session,
                    sync_info=xr.ActionsSyncInfo(
                        count_active_action_sets=1,
                        active_action_sets=ctypes.pointer(active_action_set),
                    ),
                )

                # Get controller poses
                for index, space in enumerate(self.action_spaces):
                    space_location = xr.locate_space(
                        space=space,
                        base_space=self.context.space,
                        time=frame_state.predicted_display_time,
                    )
                    if (
                        space_location.location_flags
                        & xr.SPACE_LOCATION_POSITION_VALID_BIT
                    ):
                        if index == 0:
                            left_hand_pose = xr_posef_to_world_pose(space_location.pose)
                        else:
                            right_hand_pose = xr_posef_to_world_pose(
                                space_location.pose
                            )
                        self.found_count += 1

                # Get headset pose
                view_state, views = xr.locate_views(
                    session=self.context.session,
                    view_locate_info=xr.ViewLocateInfo(
                        view_configuration_type=self.context.view_configuration_type,
                        display_time=frame_state.predicted_display_time,
                        space=self.context.space,
                    ),
                )
                flags = xr.ViewStateFlags(view_state.view_state_flags)
                if flags & xr.ViewStateFlags.POSITION_VALID_BIT:
                    left_eye_pose = views[xr.Eye.LEFT].pose
                    right_eye_pose = views[xr.Eye.RIGHT].pose
                    head_posef = xr.Posef(
                        orientation=left_eye_pose.orientation,
                        position=xr.Vector3f(
                            x=(left_eye_pose.position.x + right_eye_pose.position.x)
                            / 2,
                            y=(left_eye_pose.position.y + right_eye_pose.position.y)
                            / 2,
                            z=(left_eye_pose.position.z + right_eye_pose.position.z)
                            / 2,
                        ),
                    )
                    head_pose = xr_posef_to_world_pose(head_posef, precision=2)
                    self.found_count += 1
                # if self.found_count == 0:
                #     print("No headset or controllers active")
        except StopIteration:
            return None
        return FullPose(
            head=head_pose, left_hand=left_hand_pose, right_hand=right_hand_pose
        )


"""
   For reference:
   ... from xr.ContextObject ...

   def frame_loop(self):
        attach_session_action_sets(
            session=self.session,
            attach_info=SessionActionSetsAttachInfo(
                count_action_sets=len(self.action_sets),
                action_sets=(ActionSet * len(self.action_sets))(
                    *self.action_sets
                )
            ),
        )
        while True:
            window_closed = self.graphics.poll_events()
            if window_closed:
                self.exit_render_loop = True
                break
            self.exit_render_loop = False
            self.poll_xr_events()
            if self.exit_render_loop:
                break
            if self.session_is_running:
                if self.session_state in (
                        SessionState.READY,
                        SessionState.SYNCHRONIZED,
                        SessionState.VISIBLE,
                        SessionState.FOCUSED,
                ):
                    frame_state = wait_frame(self.session)
                    begin_frame(self.session)
                    self.render_layers = []
                    self.graphics.make_current()

                    yield frame_state

                    end_frame(
                        self.session,
                        frame_end_info=FrameEndInfo(
                            display_time=frame_state.predicted_display_time,
                            environment_blend_mode=self.environment_blend_mode,
                            layers=self.render_layers,
                        )
                    )
            else:
                # Throttle loop since xrWaitFrame won't be called.
                time.sleep(0.250)
"""
