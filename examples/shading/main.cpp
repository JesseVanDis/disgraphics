#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <concepts>
#include <disgraphics.hpp>
#include "utils_window.hpp"
#include "utils_camera.hpp"


namespace example
{
	constexpr size_t screen_width = 800;
	constexpr size_t screen_height = 800;
	constexpr size_t scale = 1;

	struct vertex
	{
		float x, y, z;
		float u, v;
	};

	struct tri
	{
		vertex p0, p1, p2;
	};

	template<typename DataType>
	struct tex
	{
		tex(std::span<DataType> rgba, size_t width, size_t height) : rgba(rgba), width(width), height(height){}
		tex(std::string_view    rgba, size_t width, size_t height) requires std::is_const_v<DataType> : rgba(reinterpret_cast<DataType*>(rgba.data()), rgba.size() / sizeof(DataType)), width(width), height(height){}

		std::span<DataType> 	rgba;
		size_t 					width = {};
		size_t 					height = {};
	};

	struct shader
	{
		shader(tex<uint32_t> target, tex<const uint32_t> texture)
			: target(target)
			, texture(texture)
		{}

		tex<uint32_t> 			target;
		tex<const uint32_t>	  	texture;

		struct vertex_to_fragment
		{
			float u = 0, v = 0;

			template<unsigned int index>
			static auto& get_field(auto& self) requires (index < 2)
			{
				if constexpr(index == 0) { return self.u; }
				if constexpr(index == 1) { return self.v; }
			}
		};

		inline vertex_to_fragment vertex(const vertex& other) // NOLINT
		{
			return vertex_to_fragment{other.u, other.v};
		}

		inline void fragment(const tri& /* source_triangle */, const dis::frag_vars<vertex_to_fragment>& in) // NOLINT
		{
			uint32_t u_tex = std::min((uint32_t)texture.width,  (uint32_t)std::floor(in.u * (double)texture.width ));
			uint32_t v_tex = std::min((uint32_t)texture.height, (uint32_t)std::floor(in.v * (double)texture.height));
			target.rgba[in.px_index] = texture.rgba[u_tex + v_tex * texture.width];
		}
	};

	bool app()
	{
		auto window = utils::create_window(screen_width, screen_height, "local positioning sim", scale);
		if(window == nullptr)
		{
			return false;
		}
		utils::camera cam(window.get());

		const size_t frame_width  = (screen_width / scale);
		const size_t frame_height = (screen_height / scale);

		// drawing functions
		std::vector<uint32_t> 	frame_pixels(frame_width * frame_height, 0x0);
		tex<uint32_t> 			frame(frame_pixels, frame_width, frame_height);
		tex<const uint32_t> 	texture("\x00\x00\xff\xff" "\x00\xff\x00\xff" "\xff\x00\x00\xff" "\x44\x44\x44\xff", 2, 2);

		// scene
		vertex aruco_mesh[] = {
				{0,0,5,  0,0},
				{2,0,5,  1,0},
				{2,2,5,  1,1},
				{0,2,5,  0,1},
		};
		std::vector<tri> triangles;
		triangles.push_back(tri{aruco_mesh[0], aruco_mesh[1], aruco_mesh[2]});
		triangles.push_back(tri{aruco_mesh[0], aruco_mesh[2], aruco_mesh[3]});

		cam.set_position(dish::vec3<float>{0.128768, 0.391487, 4.58175});
		cam.set_rot(-0.899999, 0.790792);

		// update loop
		while(!window->is_close_requested())
		{
			cam.update();
			//cam.print_position();
			std::fill(frame.rgba.begin(), frame.rgba.end(), 0xff777777);

			dis::render(triangles, cam.to_grh(), shader(frame, texture), screen_width / scale, screen_height / scale);
			window->draw(frame.rgba);
		}

		// exit with ok
		return true;
	}
}

int main()
{
	return example::app() ? 0 : 1;
}

