#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>

#include "gr/gr.hpp"

static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);

struct cam
{
	float 		pitch = 0;
	float 		yaw = M_PI / 2.0f;
	grh::vec3 	pos = {0,0,0};

	grh::cam to_grh_cam();
};
cam s_camera;

int main()
{
	if (!glfwInit())
	{
		return -1;
	}

	constexpr size_t screen_width = 800;
	constexpr size_t screen_height = 800;

	constexpr size_t scale = 4;

	// https://www.glfw.org/docs/3.3/window_guide.html
	GLFWwindow* window = glfwCreateWindow(screen_width, screen_height, "My Title", nullptr, nullptr);
	if (!window)
	{
		return -1;
	}

	glfwSetCursorPosCallback(window, mouse_callback);

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
	triangles.push_back({{0,0,5}, {2,0.2f,5}, {0,1,5}});

	glfwMakeContextCurrent(window);
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2f(-1, -1);
		glPixelZoom(scale, scale);

		//auto camera = grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
		memset(draw.pixels.data(), 0, draw.pixels.size() * sizeof(draw.pixels[0]));
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
			.fov = 2.0f
	};
}


static void mouse_callback(GLFWwindow* window, const double xpos, const double ypos)
{
	static float s_sensitivity = 0.01f; // change this value to your liking
	static bool s_first_mouse = true;
	static double last_x = true;
	static double last_y = true;

	if (s_first_mouse)
	{
		last_x = xpos;
		last_y = ypos;
		s_first_mouse = false;
	}

	const double xoffset = (xpos - last_x) * s_sensitivity;
	const double yoffset = (last_y - ypos) * s_sensitivity; // reversed since y-coordinates go from bottom to top
	last_x = xpos;
	last_y = ypos;

	s_camera.pitch 	= static_cast<float>(std::clamp(s_camera.pitch + yoffset, -89.0 * 57.3, 89.0 * 57.3)); // 57.3 is the ratio between radians and degreees
	s_camera.yaw 	= static_cast<float>(s_camera.yaw + xoffset);
}
