
#ifndef SIMULATOR_CAMERA_HPP
#define SIMULATOR_CAMERA_HPP

#include <disgraphics.hpp>
#include "utils_window.hpp"

namespace example::utils
{
	class camera
	{
		public:
			explicit camera(utils::input* input);
			void 		update();
			dish::cam 	to_grh() const;
			void 		set_position(const dish::vec3<float>& pos);
			void 		set_rot(float pitch_radians, float yaw_radians);
			void 		print_position() const;

		private:
			void 		update_movement();
			void 		update_orientation();

			utils::input* m_input = nullptr;

			float 		m_pitch = 0;
			float 		m_yaw = M_PI / 2.0f;
			dish::vec3<float> 	m_pos = {0,0,0};
			float 		m_fov = 130.0f / 57.2958f; // 90 degrees in radians

			int 		m_mouse_button_state = 0;
			double 		m_mouse_last_x = -1;
			double 		m_mouse_last_y = -1;
	};

}


#endif //SIMULATOR_CAMERA_HPP
