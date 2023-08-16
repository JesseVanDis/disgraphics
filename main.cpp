#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>

#include "gr/gr.hpp"

static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
static void update_movement();

static constexpr float s_deg_to_rad = 1.0f / 57.2958f;

struct cam
{
	float 		pitch = 0;
	float 		yaw = M_PI / 2.0f;
	grh::vec3 	pos = {0,0,0};
	float 		fov = 90.0f * s_deg_to_rad;

	grh::cam to_grh_cam();
};
cam s_camera;

static bool s_pressed_up 	= false;
static bool s_pressed_down 	= false;
static bool s_pressed_left 	= false;
static bool s_pressed_right = false;

int main()
{
	if (!glfwInit())
	{
		return -1;
	}

	constexpr size_t screen_width = 800;
	constexpr size_t screen_height = 800;

	constexpr size_t scale = 2;

	// https://www.glfw.org/docs/3.3/window_guide.html
	GLFWwindow* window = glfwCreateWindow(screen_width, screen_height, "My Title", nullptr, nullptr);
	if (!window)
	{
		return -1;
	}

	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);

	struct
	{
		std::vector<uint32_t> pixels;//((screen_width / scale) * (screen_height / scale), 0x0);

		inline void operator()(const grh::tri& source_triangle, const gr::draw_hline_ctx& ctx)
		{
			auto begin = std::next(pixels.begin(), ctx.px_x_from + ctx.px_y * ctx.buffer_width);
			auto end = std::next(begin, ctx.px_x_to - ctx.px_x_from);
			std::fill(begin, end, 0xffffffff);
		}
	} draw;
	draw.pixels.resize((screen_width / scale) * (screen_height / scale), 0x0);

	std::vector<grh::tri> triangles;
	//triangles.push_back({{0,0,5}, {2,0.2f,5}, {0,1,5}});
	triangles.push_back({{0,0,5}, {2,0,5}, {1,1,5}});
	//triangles.push_back({{2,1,5}, {2,0.2f,5}, {0,1,5}});

	//s_camera.pitch = 0.840000f;
	//s_camera.yaw = 2.050796f;

	//s_camera.pitch = 1.519999;
	//s_camera.yaw = 1.710796;

	//s_camera.pitch = 1.569999;
	//s_camera.yaw = 1.690796;

	glfwMakeContextCurrent(window);
	while (!glfwWindowShouldClose(window))
	{
		update_movement();

		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2f(-1, -1);
		glPixelZoom(scale, scale);

		//auto camera = grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
		memset(draw.pixels.data(), 0, draw.pixels.size() * sizeof(draw.pixels[0]));
		//printf("%f, %f\n", s_camera.pitch, s_camera.yaw);

		gr::render(triangles, s_camera.to_grh_cam(), draw, screen_width / scale, screen_height / scale);

		glDrawPixels((screen_width / scale), (screen_height / scale), GL_RGBA, GL_UNSIGNED_BYTE, draw.pixels.data()); //draw pixel
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	glfwTerminate();
	return 0;
}


grh::cam cam::to_grh_cam()
{
	const grh::vec3 front = {std::cos(yaw) * std::cos(pitch), std::sin(pitch), std::sin(yaw) * std::cos(pitch)};

	return grh::cam {
			.pos = pos,
			.lookat = {pos.x + front.x, pos.y + front.y, pos.z + front.z},
			.up = {0,1,0},
			.fov = fov
	};
}

static int s_mouse_state = 0;

static void mouse_callback(GLFWwindow* window, const double xpos, const double ypos)
{
	static float s_sensitivity = 0.01f; // change this value to your liking
	static double last_x = xpos;
	static double last_y = ypos;

	const double xoffset = (xpos - last_x) * s_sensitivity;
	const double yoffset = (last_y - ypos) * s_sensitivity; // reversed since y-coordinates go from bottom to top
	last_x = xpos;
	last_y = ypos;

	if(s_mouse_state == 1)
	{
		s_mouse_state = 2;
	}
	else if(s_mouse_state == 2)
	{
		s_camera.pitch 	= static_cast<float>(std::clamp(s_camera.pitch + yoffset, -89.0 * 57.3, 89.0 * 57.3)); // 57.3 is the ratio between radians and degreees
		s_camera.yaw 	= static_cast<float>(s_camera.yaw - xoffset);
	}
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if(action == GLFW_PRESS)
		{
			s_mouse_state = 1;
		}
		else if(action == GLFW_RELEASE)
		{
			s_mouse_state = 0;
		}
	}
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS 	&& key == GLFW_KEY_W)	s_pressed_up 	= true;
	if (action == GLFW_PRESS 	&& key == GLFW_KEY_S)	s_pressed_down 	= true;
	if (action == GLFW_PRESS 	&& key == GLFW_KEY_A)	s_pressed_left 	= true;
	if (action == GLFW_PRESS 	&& key == GLFW_KEY_D)	s_pressed_right = true;
	if (action == GLFW_RELEASE 	&& key == GLFW_KEY_W)	s_pressed_up 	= false;
	if (action == GLFW_RELEASE 	&& key == GLFW_KEY_S)	s_pressed_down 	= false;
	if (action == GLFW_RELEASE 	&& key == GLFW_KEY_A)	s_pressed_left 	= false;
	if (action == GLFW_RELEASE 	&& key == GLFW_KEY_D)	s_pressed_right = false;
}

static void update_movement()
{
	constexpr float speed = 0.05f;
	const grh::vec3 cam_dir = {std::cos(s_camera.yaw) * std::cos(s_camera.pitch), std::sin(s_camera.pitch), std::sin(s_camera.yaw) * std::cos(s_camera.pitch)};
	grh::vec3 cam_tan = {-cam_dir.z, 0, cam_dir.x};
	if(cam_tan.x != 0.0f && cam_tan.z != 0.0f)
	{
		const float l = std::sqrt(cam_tan.x*cam_tan.x + cam_tan.z*cam_tan.z);
		cam_tan.x /= l;
		cam_tan.y /= l;
		cam_tan.z /= l;
	}

	if(s_pressed_up)
	{
		s_camera.pos = {s_camera.pos.x + cam_dir.x * speed,  s_camera.pos.y + cam_dir.y * speed,  s_camera.pos.z + cam_dir.z * speed};
	}
	if(s_pressed_down)
	{
		s_camera.pos = {s_camera.pos.x - cam_dir.x * speed,  s_camera.pos.y - cam_dir.y * speed,  s_camera.pos.z - cam_dir.z * speed};
	}
	if(s_pressed_right)
	{
		s_camera.pos = {s_camera.pos.x - cam_tan.x * speed,  s_camera.pos.y - cam_tan.y * speed,  s_camera.pos.z - cam_tan.z * speed};
	}
	if(s_pressed_left)
	{
		s_camera.pos = {s_camera.pos.x + cam_tan.x * speed,  s_camera.pos.y + cam_tan.y * speed,  s_camera.pos.z + cam_tan.z * speed};
	}
}