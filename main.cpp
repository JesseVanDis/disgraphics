#include <iostream>
#include <vector>
#include <GLFW/glfw3.h>

#include "gr/gr.hpp"

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

	std::vector<uint32_t> pixels((screen_width / scale) * (screen_height / scale), 0x0);

	auto render_target = grh::create_render_target((screen_width / scale), (screen_height / scale));

	std::vector<grh::tri> triangles;
	triangles.push_back({{0,0,5}, {2,0.2f,5}, {0,1,5}});

	glfwMakeContextCurrent(window);
	while (!glfwWindowShouldClose(window))
	{
		glClearColor(1.0, 1.0, 1.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		glRasterPos2f(-1, -1);
		glPixelZoom(scale, scale);

		auto camera = grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
		gr::render(render_target, triangles, camera);

		glDrawPixels((screen_width / scale), (screen_height / scale), GL_RGBA, GL_UNSIGNED_BYTE, render_target.rgbd_buffer.get()); //draw pixel
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	glfwTerminate();
	return 0;
}
