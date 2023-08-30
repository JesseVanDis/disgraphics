#include <string>
#include <cassert>
#include <optional>
#include <iostream>
#include <GLFW/glfw3.h>
#include "utils_window.hpp"

namespace example::utils
{

	namespace detail
	{
		struct user_data
		{
			bool 					pressed_w = false;
			bool 					pressed_s = false;
			bool 					pressed_a = false;
			bool 					pressed_d = false;
			std::optional<double> 	mouse_x;
			std::optional<double> 	mouse_y;
			int 					mouse_state = 0;
		};

		static void mouse_callback(GLFWwindow* window, const double xpos, const double ypos)
		{
			user_data* data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(window)); // NOLINT
			data->mouse_x = xpos;
			data->mouse_y = ypos;
		}

		static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
		{
			user_data* data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(window)); // NOLINT
			if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_LEFT)
			{
				if(action == GLFW_PRESS)
				{
					data->mouse_state = 1;
				}
				else if(action == GLFW_RELEASE)
				{
					data->mouse_state = 0;
				}
			}
		}

		static void key_callback(GLFWwindow* window, int key, int /* scancode */, int action, int /* mods */)
		{
			user_data* data = reinterpret_cast<user_data*>(glfwGetWindowUserPointer(window)); // NOLINT
			if (action == GLFW_PRESS 	&& key == GLFW_KEY_W) data->pressed_w 	= true;
			if (action == GLFW_PRESS 	&& key == GLFW_KEY_S) data->pressed_s 	= true;
			if (action == GLFW_PRESS 	&& key == GLFW_KEY_A) data->pressed_a 	= true;
			if (action == GLFW_PRESS 	&& key == GLFW_KEY_D) data->pressed_d 	= true;
			if (action == GLFW_RELEASE 	&& key == GLFW_KEY_W) data->pressed_w 	= false;
			if (action == GLFW_RELEASE 	&& key == GLFW_KEY_S) data->pressed_s 	= false;
			if (action == GLFW_RELEASE 	&& key == GLFW_KEY_A) data->pressed_a 	= false;
			if (action == GLFW_RELEASE 	&& key == GLFW_KEY_D) data->pressed_d 	= false;
		}

		struct glfw_context
		{
			glfw_context()
			{
				ok = glfwInit();
			}
			~glfw_context()
			{
				glfwTerminate();
			}
			bool ok = false;
		};

		static std::shared_ptr<glfw_context> s_glfw_context = nullptr;

		struct window::impl
		{
			user_data 						m_window_user_data;
			GLFWwindow* 					m_window = nullptr;
			size_t 							m_width = 0;
			size_t 							m_height = 0;
			size_t 							m_scale = 1;
			std::shared_ptr<glfw_context> 	m_glfw_context;

			// GLFW actually uses globals... so this is pretty hacky
			impl(size_t width, size_t height, std::string_view title, size_t scale)
			{
				if(s_glfw_context == nullptr)
				{
					s_glfw_context = std::make_shared<glfw_context>();
				}
				m_glfw_context = s_glfw_context;

				if (!m_glfw_context->ok)
				{
					std::cerr << "Failed to initialize glfw\n";
					return;
				}

				m_width = width;
				m_height = height;
				m_scale = scale;
				std::string title_str(title);
				m_window = glfwCreateWindow(width, height, title_str.c_str(), nullptr, nullptr);
				glfwMakeContextCurrent(m_window);

				if(m_window != nullptr)
				{
					glfwSetWindowUserPointer(m_window, &m_window_user_data);
					glfwSetCursorPosCallback(m_window, mouse_callback);
					glfwSetMouseButtonCallback(m_window, mouse_button_callback);
					glfwSetKeyCallback(m_window, key_callback);
				}
				else
				{
					std::cerr << "Failed to create glfw window\n";
					return;
				}
				m_is_ok = true;
			}

			~impl()
			{
				glfwTerminate();
			}

			void draw(std::span<const uint32_t> pixels)
			{
				glRasterPos2f(-1, -1);
				glPixelZoom(m_scale, m_scale);

				assert(pixels.size() >= (m_width / m_scale) * (m_height / m_scale));
				glDrawPixels((m_width / m_scale), (m_height / m_scale), GL_RGBA, GL_UNSIGNED_BYTE, pixels.data()); //draw pixel

				glfwSwapBuffers(m_window);
				glfwPollEvents();
			}

			bool is_key_pressed_w() const
			{
				return m_window_user_data.pressed_w;
			}

			bool is_key_pressed_s() const
			{
				return m_window_user_data.pressed_s;
			}

			bool is_key_pressed_a() const
			{
				return m_window_user_data.pressed_a;
			}

			bool is_key_pressed_d() const
			{
				return m_window_user_data.pressed_d;
			}

			bool is_close_requested() const
			{
				return glfwWindowShouldClose(m_window);
			}

			bool is_mouse_pressed() const
			{
				return m_window_user_data.mouse_state == 1;
			}

			std::pair<double, double> get_mouse_pos()
			{
				if(m_window_user_data.mouse_x != std::nullopt && m_window_user_data.mouse_y != std::nullopt)
				{
					return {*m_window_user_data.mouse_x, *m_window_user_data.mouse_y};
				}
				return {-1, -1};
			}

			bool is_ok() const { return m_is_ok; }

			bool m_is_ok = false;
		};

		window::window(size_t width, size_t height, std::string_view title, size_t scale)
			: m_impl(std::make_unique<impl>(width, height, title, scale))
		{}

		window::~window() = default;

		void window::draw(std::span<const uint32_t> pixels) { 			m_impl->draw(pixels); 		}
		bool window::is_key_pressed_w()						{ return 	m_impl->is_key_pressed_w(); }
		bool window::is_key_pressed_s()						{ return 	m_impl->is_key_pressed_s(); }
		bool window::is_key_pressed_a()						{ return 	m_impl->is_key_pressed_a(); }
		bool window::is_key_pressed_d()						{ return 	m_impl->is_key_pressed_d(); }
		bool window::is_mouse_pressed()						{ return 	m_impl->is_mouse_pressed(); }
		std::pair<double, double> window::get_mouse_pos()	{ return m_impl->get_mouse_pos(); 		}
		bool window::is_close_requested()					{ return m_impl->is_close_requested(); 	}

		bool is_ok(window& window) { return window.m_impl->is_ok(); }
	}

	window create_window(size_t width, size_t height, std::string_view title, size_t scale)
	{
		if(auto retval = std::make_unique<detail::window>(width, height, title, scale))
		{
			if(detail::is_ok(*retval))
			{
				return retval;
			}
		}
		return nullptr;
	}
};

