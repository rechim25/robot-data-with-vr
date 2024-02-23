import xr
import numpy as np
import ctypes

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


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

#    def frame_loop(self):
#         attach_session_action_sets(
#             session=self.session,
#             attach_info=SessionActionSetsAttachInfo(
#                 count_action_sets=len(self.action_sets),
#                 action_sets=(ActionSet * len(self.action_sets))(
#                     *self.action_sets
#                 )
#             ),
#         )
#         while True:
#             window_closed = self.graphics.poll_events()
#             if window_closed:
#                 self.exit_render_loop = True
#                 break
#             self.exit_render_loop = False
#             self.poll_xr_events()
#             if self.exit_render_loop:
#                 break
#             if self.session_is_running:
#                 if self.session_state in (
#                         SessionState.READY,
#                         SessionState.SYNCHRONIZED,
#                         SessionState.VISIBLE,
#                         SessionState.FOCUSED,
#                 ):
#                     frame_state = wait_frame(self.session)
#                     begin_frame(self.session)
#                     self.render_layers = []
#                     self.graphics.make_current()

#                     yield frame_state

#                     end_frame(
#                         self.session,
#                         frame_end_info=FrameEndInfo(
#                             display_time=frame_state.predicted_display_time,
#                             environment_blend_mode=self.environment_blend_mode,
#                             layers=self.render_layers,
#                         )
#                     )
#             else:
#                 # Throttle loop since xrWaitFrame won't be called.
#                 time.sleep(0.250)


class XrWrapper:
    def __init__(self):
        self.context = None
        self.vao = None
        self.shader_program = None
        self.action_spaces = None
        self.found_count = None
        self.frame_state_generator = None

    def __enter__(self):
        self.context = xr.ContextObject(
            instance_create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            )
        )
        self.context.__enter__()
        self.vao, self.shader_program = self._init_graphics()
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
        return vao, shader_program

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

    def _load_texture_from_rgb_array(self, rgb_array):
        """Load an image from bytes into an OpenGL texture."""
        height, width, channels = rgb_array.shape
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

    def step(self, rgb_left, rgb_right):
        try:
            frame_state = next(self.frame_state_generator)
            for view_index, view in enumerate(self.context.view_loop(frame_state)):
                # Clear the color and depth buffers.
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                # Use the shader program.
                glUseProgram(self.shader_program)

                if view_index == 0:
                    texture = self._load_texture_from_rgb_array(rgb_left)
                else:
                    texture = self._load_texture_from_rgb_array(rgb_right)

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
                        entity = "Headset" if index == 2 else f"Controller {index + 1}"
                        print(entity, space_location.pose)
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
                    headset_pose = views[xr.Eye.LEFT]
                    print(headset_pose, flush=True)

                if self.found_count == 0:
                    print("No headset or controllers active")

                return
        except StopIteration:
            return None
