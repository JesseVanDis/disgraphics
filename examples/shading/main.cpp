#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <disgraphics.hpp>
#include "utils_window.hpp"
#include "utils_camera.hpp"

namespace example
{
	constexpr size_t screen_width = 800;
	constexpr size_t screen_height = 800;

	constexpr size_t scale = 4;


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
		};

		vertex_it begin;
		vertex_it it;
	};

	//void test(const tri& source_triangle, const draw_ctx& ctx)
	//{
	//}

	//void test2(gr::draw_horizontal_line_function<draw_ctx, tri> auto func)
	//{
		//static_assert(gr::draw_horizontal_line_function<decltype(func), draw_ctx, gr::triangle_from_list_t<std::vector<tri>>>);
	//}

	bool app()
	{
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
			std::vector<uint32_t> 	pixels;//((screen_width / scale) * (screen_height / scale), 0x0);

			inline void operator()(const tri& source_triangle, const draw_ctx& ctx)
			{
				uint32_t* px = &pixels[ctx.px_x_from + ctx.px_y * buffer_width];
				float one_over_z = ctx.one_over_z;
				auto it = ctx.begin;

				//uint8_t uu = (uint8_t)std::clamp(it.u * 255.0f, 0.0f, 255.0f);

				float ZZ = (1.0f/one_over_z);
				printf("%f\n", it.u);

				for(size_t i=0; i<ctx.line_length_px; i++)
				{
					float Z = (1.0f/one_over_z);

					uint8_t u = (uint8_t)std::clamp(it.u * 255.0f, 0.0f, 255.0f);
					uint8_t v = (uint8_t)std::clamp(it.v * 255.0f, 0.0f, 255.0f);

					//uint8_t color = u;//std::clamp(Z, 0.0f, 255.0f);

					//px[i] = (u << 24) + (u << 16) + (v << 8) + (v << 0);
					// u = red
					// v = blue
					px[i] = (0 << 24) + (v << 16) + (0 << 8) + (u << 0);

					one_over_z += ctx.one_over_z_it;
					//dis::detail::add(it, ctx.it);
					it += ctx.it;
				}

				//auto begin = std::next(pixels.begin(), ctx.px_x_from + ctx.px_y * buffer_width);
				//auto end = std::next(begin, ctx.line_length_px);
				//std::fill(begin, end, 0xffffffff);
			}
		} draw;
		draw.buffer_width 	= (screen_width / scale);
		draw.buffer_height	= (screen_height / scale);
		draw.texture 		= {texture, texture_width, texture_height};
		draw.pixels.resize(draw.buffer_width * draw.buffer_height, 0x0);

		// scene
		std::vector<tri> triangles;

		triangles.push_back(tri{
				.p0  = {0,0,5,  0,0},
				.p1  = {2,0,5,  1,0},
				.p2  = {2,2,5,  1,1},
		});

		// update loop
		while(!window->is_close_requested())
		{
			cam.update();
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

