
#ifndef SIMULATOR_CAMERA_HPP
#define SIMULATOR_CAMERA_HPP

#include <disgraphics.hpp>

namespace sim
{
	class camera
	{
		public:
			void 		update();
			dish::cam 	to_grh();

		private:
			void 		update_movement();
			void 		update_orientation();

			float 		m_pitch = 0;
			float 		m_yaw = M_PI / 2.0f;
			dish::vec3 	m_pos = {0,0,0};
			float 		m_fov = 90.0f / 57.2958f; // 90 degrees in radians

			int 		m_mouse_button_state = 0;
			double 		m_mouse_last_x = -1;
			double 		m_mouse_last_y = -1;
	};

}


#endif //SIMULATOR_CAMERA_HPP
