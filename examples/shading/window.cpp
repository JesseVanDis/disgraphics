#include <string>
#include <GLFW/glfw3.h>
#include <cassert>
#include <optional>
#include "window.hpp"

namespace sim::window
{
	static GLFWwindow* 	s_window = nullptr;
	static size_t 		s_width = 0;
	static size_t 		s_height = 0;
	static size_t 		s_scale = 1;

	static bool s_pressed_w 	= false;
	static bool s_pressed_s 	= false;
	static bool s_pressed_a 	= false;
	static bool s_pressed_d = false;

	static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void update_movement();

	bool create(size_t width, size_t height, std::string_view title, size_t scale)
	{
		if (!glfwInit())
		{
			return false;
		}

		s_width = width;
		s_height = height;
		s_scale = scale;
		std::string title_str(title);
		s_window = glfwCreateWindow(width, height, title_str.c_str(), nullptr, nullptr);
		glfwMakeContextCurrent(s_window);

		if(s_window != nullptr)
		{
			glfwSetCursorPosCallback(s_window, mouse_callback);
			glfwSetMouseButtonCallback(s_window, mouse_button_callback);
			glfwSetKeyCallback(s_window, key_callback);
		}
		else
		{
			return false;
		}
		return true;
	}

	static std::optional<double> s_mouse_x;
	static std::optional<double> s_mouse_y;

	void draw(std::span<const uint32_t> pixels)
	{
		//glRasterPos2f(-1, -1);
		//glPixelZoom(s_scale, s_scale);

		assert(pixels.size() >= (s_width / s_scale) * (s_height / s_scale));
		glDrawPixels((s_width / s_scale), (s_height / s_scale), GL_RGBA, GL_UNSIGNED_BYTE, pixels.data()); //draw pixel

		glfwSwapBuffers(s_window);
		glfwPollEvents();
	}

	void close()
	{
		glfwTerminate();
	}

	bool is_key_pressed_w()
	{
		return s_pressed_w;
	}

	bool is_key_pressed_s()
	{
		return s_pressed_s;
	}

	bool is_key_pressed_a()
	{
		return s_pressed_a;
	}

	bool is_key_pressed_d()
	{
		return s_pressed_d;
	}

	bool is_close_requested()
	{
		return glfwWindowShouldClose(s_window);
	}

	static int s_mouse_state = 0;
	bool is_mouse_pressed()
	{
		return s_mouse_state == 1;
	}

	std::pair<double, double> get_mouse_pos()
	{
		if(s_mouse_x != std::nullopt && s_mouse_y != std::nullopt)
		{
			return {*s_mouse_x, *s_mouse_y};
		}
		return {-1, -1};
	}

	static void mouse_callback(GLFWwindow* window, const double xpos, const double ypos)
	{
		s_mouse_x = xpos;
		s_mouse_y = ypos;
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
		if (action == GLFW_PRESS 	&& key == GLFW_KEY_W) s_pressed_w 	= true;
		if (action == GLFW_PRESS 	&& key == GLFW_KEY_S) s_pressed_s 	= true;
		if (action == GLFW_PRESS 	&& key == GLFW_KEY_A) s_pressed_a 	= true;
		if (action == GLFW_PRESS 	&& key == GLFW_KEY_D) s_pressed_d 	= true;
		if (action == GLFW_RELEASE 	&& key == GLFW_KEY_W) s_pressed_w 	= false;
		if (action == GLFW_RELEASE 	&& key == GLFW_KEY_S) s_pressed_s 	= false;
		if (action == GLFW_RELEASE 	&& key == GLFW_KEY_A) s_pressed_a 	= false;
		if (action == GLFW_RELEASE 	&& key == GLFW_KEY_D) s_pressed_d 	= false;
	}
};

