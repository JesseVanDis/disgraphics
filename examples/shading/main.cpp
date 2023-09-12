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

		constexpr vertex operator + (float other) const { return {x + other, y + other, z + other, u + other, v + other}; }
	};

	struct tri
	{
		vertex p0, p1, p2;
	};

	struct tex
	{
		std::span<const uint32_t> 	rgba;
		size_t 						texture_width = {};
		size_t 						texture_height = {};
	};

	struct draw_ctx
	{
		uint32_t buffer_width;
		uint32_t buffer_height;
		uint32_t px_y, px_x_from, line_length_px;

		float one_over_z;
		float one_over_z_it;

		struct vertex_it
		{
			float u, v;

			template<unsigned int index>
			static auto& get_field(auto& self) requires (index < 2)
			{
				if constexpr(index == 0) { return self.u; }
				if constexpr(index == 1) { return self.v; }
			}

			vertex_it& operator = (const vertex& other)
			{
				u = other.u;
				v = other.v;
				return *this;
			}

			vertex_it& operator += (const vertex_it& other)
			{
				dish::for_each_field(*this, other, [](auto& a, const auto& b){ a += b;});
				return *this;
			}

			vertex_it& operator -= (const vertex_it& other)
			{
				dish::for_each_field(*this, other, [](auto& a, const auto& b){ a -= b;});
				return *this;
			}

			vertex_it& operator *= (dis::arithmetic auto other)
			{
				dish::for_each_field(*this, other, [](auto& a, const auto& b){ a *= b;});
				return *this;
			}

			vertex_it operator * (dis::arithmetic auto other) const
			{
				vertex_it t = *this;
				t *= other;
				return t;
			}

			vertex_it operator - (const vertex_it& other) const
			{
				vertex_it t = *this;
				t -= other;
				return t;
			}

			vertex_it operator + (const vertex_it& other) const
			{
				vertex_it t = *this;
				t += other;
				return t;
			}

		};

		vertex_it begin;
		vertex_it it;
	};

	bool app()
	{
		//todo: Just use an openGL solution that matched this libraries interface for now

		auto window = utils::create_window(screen_width, screen_height, "local positioning sim", scale);
		if(window == nullptr)
		{
			return false;
		}
		utils::camera cam(window.get());

		constexpr size_t texture_width = 2;
		constexpr size_t texture_height = 2;
		const std::array<uint32_t, texture_width * texture_height> texture {0xffff0000, 0xff00ff00, 0xff0000ff, 0xff444444};

		// drawing functions
		struct
		{
			uint32_t 				buffer_width, buffer_height;
			tex						texture;
			std::vector<uint32_t> 	pixels;

			inline void operator()(const tri& source_triangle, const draw_ctx& ctx)
			{
				uint32_t* px = &pixels[ctx.px_x_from + ctx.px_y * buffer_width];

				auto 	current 	= ctx.begin;
				float 	one_over_z 	= ctx.one_over_z;

				for(size_t i=0; i<ctx.line_length_px; i++)
				{
					float z = (1.0f/one_over_z);
					float u = current.u * z;
					float v = current.v * z;

					uint32_t u_tex = std::min((uint32_t)texture.texture_width, (uint32_t)std::floor(u * (double)texture.texture_width));
					uint32_t v_tex = std::min((uint32_t)texture.texture_height, (uint32_t)std::floor(v * (double)texture.texture_width));
					px[i] = texture.rgba[u_tex + v_tex * texture.texture_width];
					current += ctx.it;
					one_over_z += ctx.one_over_z_it;
				}
			}
		} draw;
		draw.buffer_width 	= (screen_width / scale);
		draw.buffer_height	= (screen_height / scale);
		draw.texture 		= {texture, texture_width, texture_height};
		draw.pixels.resize(draw.buffer_width * draw.buffer_height, 0x0);

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
			std::fill(draw.pixels.begin(), draw.pixels.end(), 0xff777777);

			dis::render<draw_ctx>(triangles, cam.to_grh(), draw, screen_width / scale, screen_height / scale);
			window->draw(draw.pixels);
		}

		// exit with ok
		return true;
	}
}

int main()
{
	return example::app() ? 0 : 1;
}

