#include <iostream>
#include "utils_camera.hpp"

namespace example::utils
{

	static float s_mouse_sensitivity = 0.01f; // change this value to your liking

	camera::camera(utils::input* input)
		: m_input(input)
	{
	}


	void camera::update()
	{
		update_orientation();
		update_movement();
	}

	dish::cam camera::to_grh() const
	{
		const dish::vec3<float> front = {std::cos(m_yaw) * std::cos(m_pitch), std::sin(m_pitch), std::sin(m_yaw) * std::cos(m_pitch)};

		return dish::cam {
				.pos = m_pos,
				.lookat = {m_pos.x + front.x, m_pos.y + front.y, m_pos.z + front.z},
				.up = {0,1,0},
				.fov = m_fov
		};
	}

	void camera::set_position(const dish::vec3<float>& pos)
	{
		m_pos = pos;
	}

	void camera::set_rot(float pitch_radians, float yaw_radians)
	{
		m_pitch = pitch_radians;
		m_yaw = yaw_radians;
	}

	void camera::print_position() const
	{
		std::cout << "cam pos: " << m_pos.x << ", " << m_pos.y << ", " << m_pos.z << ", cam pitch: " << m_pitch << ", yaw: " << m_yaw << "\n";
	}

	void camera::update_movement()
	{
		constexpr float speed = 0.05f;
		const dish::vec3<float> cam_dir = {std::cos(m_yaw) * std::cos(m_pitch), std::sin(m_pitch), std::sin(m_yaw) * std::cos(m_pitch)};
		dish::vec3<float> cam_tan = {-cam_dir.z, 0, cam_dir.x};
		if(cam_tan.x != 0.0f && cam_tan.z != 0.0f)
		{
			const float l = std::sqrt(cam_tan.x*cam_tan.x + cam_tan.z*cam_tan.z);
			cam_tan.x /= l;
			cam_tan.y /= l;
			cam_tan.z /= l;
		}

		if(m_input->is_key_pressed_w())
		{
			m_pos = {m_pos.x + cam_dir.x * speed,  m_pos.y + cam_dir.y * speed,  m_pos.z + cam_dir.z * speed};
		}
		if(m_input->is_key_pressed_s())
		{
			m_pos = {m_pos.x - cam_dir.x * speed,  m_pos.y - cam_dir.y * speed,  m_pos.z - cam_dir.z * speed};
		}
		if(m_input->is_key_pressed_d())
		{
			m_pos = {m_pos.x - cam_tan.x * speed,  m_pos.y - cam_tan.y * speed,  m_pos.z - cam_tan.z * speed};
		}
		if(m_input->is_key_pressed_a())
		{
			m_pos = {m_pos.x + cam_tan.x * speed,  m_pos.y + cam_tan.y * speed,  m_pos.z + cam_tan.z * speed};
		}
	}

	void camera::update_orientation()
	{
		const bool is_mouse_pressed = m_input->is_mouse_pressed();
		const auto [mouse_x, mouse_y] = m_input->get_mouse_pos();

		if(m_mouse_last_x == -1 && m_mouse_last_y == -1)
		{
			m_mouse_last_x = mouse_x;
			m_mouse_last_y = mouse_y;
		}

		const double xoffset = (mouse_x - m_mouse_last_x) * s_mouse_sensitivity;
		const double yoffset = (m_mouse_last_y - mouse_y) * s_mouse_sensitivity; // reversed since y-coordinates go from bottom to top
		m_mouse_last_x = mouse_x;
		m_mouse_last_y = mouse_y;

		if(m_mouse_button_state == 1)
		{
			m_mouse_button_state = 2;
		}
		else if(m_mouse_button_state == 2)
		{
			m_pitch = static_cast<float>(std::clamp(m_pitch + yoffset, -89.0 * 57.3, 89.0 * 57.3)); // 57.3 is the ratio between radians and degreees
			m_yaw 	= static_cast<float>(m_yaw - xoffset);
		}

		if(is_mouse_pressed && m_mouse_button_state == 0)
		{
			m_mouse_button_state = 1;
		}
		else if(!m_input->is_mouse_pressed())
		{
			m_mouse_button_state = 0;
		}
	}


}
