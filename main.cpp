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

	struct
	{
		std::vector<uint32_t> pixels;//((screen_width / scale) * (screen_height / scale), 0x0);

		void operator()(const grh::tri& source_triangle, const gr::draw_hline_ctx& ctx)
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

		auto camera = grh::lookat(grh::vec3{0,0,0}, grh::vec3{0,0,1}, grh::vec3{0,1,0}, 2.0f);
		memset(draw.pixels.data(), 0, draw.pixels.size() * sizeof(draw.pixels[0]));
		gr::render(triangles, camera, draw, screen_width / scale, screen_height / scale);

		glDrawPixels((screen_width / scale), (screen_height / scale), GL_RGBA, GL_UNSIGNED_BYTE, draw.pixels.data()); //draw pixel
		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	glfwTerminate();
	return 0;
}
