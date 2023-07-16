#ifndef CPURAYTRACER_GT_HPP
#define CPURAYTRACER_GT_HPP

#include <cstdint>
#include <memory>

namespace gr
{
	template<typename T>
	concept floating_point = std::floating_point<std::decay_t<T>>;

	template<typename T>
	concept unsigned_integral = std::unsigned_integral<std::decay_t<T>>;

	template<typename T>
	concept unsigned_int_32 = std::same_as<uint32_t, std::decay_t<T>>;

	template<typename T>
	concept vector3 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
		{v.z} -> floating_point;
	};

	template<typename T>
	concept triangle = requires(T v)
	{
		{v.p0} -> vector3;
		{v.p1} -> vector3;
		{v.p2} -> vector3;
	};

	template<typename T>
	concept triangle_list = requires(T v)
	{
		{v.begin()};
		{v.end()};
		{*v.begin()} -> triangle;
		{*v.end()} -> triangle;
	};

	template<typename T>
	concept triangle_color = requires(T v)
	{
		{v.color} -> unsigned_int_32;
	};

	template<typename T>
	concept camera = requires(T v)
	{
		{v.pos} 			-> vector3;
		{v.lookat} 			-> vector3;
		{v.up} 				-> vector3;
		{v.fov} 			-> floating_point;
	};

	template<typename T>
	concept render_target = requires(T v)
	{
		/// 32 byte aligned red,green,blue,depth
		{v.data()} 			-> std::same_as<uint32_t*>;

		{v.width} 			-> unsigned_integral;
		{v.height} 			-> unsigned_integral;
	};

	namespace helpers
	{
		struct vec3
		{
			float x,y,z;
		};

		struct tri
		{
			vec3 p0, p1, p2;
		};

		struct cam
		{
			vec3 pos, lookat, up;
			float fov;
		};

		struct managed_render_target
		{
			std::unique_ptr<uint32_t> 	rgbd_buffer;   // 32 byte aligned red,green,blue,depth
			size_t						width;
			size_t						height;

			uint32_t*					data() { return rgbd_buffer.get(); } // NOLINT
		};

		inline managed_render_target 	create_render_target(size_t num_pixels_width, size_t num_pixels_height);
		cam 							lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov = 2.0f);
	}

	namespace detail
	{
		struct impl;
	}

	struct context
	{
		~context();

		std::unique_ptr<detail::impl> impl;
	};

	context			create_context();

	void			clear(render_target auto& target);

	void 			render(context& context, 	render_target auto& target, const triangle_list auto& triangles, const camera auto& camera);

	/// for performance it is recommended to use a context if you are going to render multiple frames per second
	void 			render(						render_target auto& target, const triangle_list auto& triangles, const camera auto& camera);
}

namespace grh = gr::helpers;


// implementation
#include <span>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace gr
{
	namespace detail
	{
		struct impl
		{
			std::unique_ptr<float> 	managed_depth_buffer;
			size_t					managed_depth_buffer_size = 0;
		};
	}

	context::~context() = default;


	namespace detail
	{
		void draw_triangle(render_target auto& target, const triangle auto& source_triangle,
						   float p0_px_x, float p0_px_y, float p0_depth_z,
						   float p1_px_x, float p1_px_y, float p1_depth_z,
						   float p2_px_x, float p2_px_y, float p2_depth_z)
		{
			int p0_px_x_i = p0_px_x; // NOLINT
			int p0_px_y_i = p0_px_y; // NOLINT
			int p1_px_x_i = p1_px_x; // NOLINT
			int p1_px_y_i = p1_px_y; // NOLINT
			int p2_px_x_i = p2_px_x; // NOLINT
			int p2_px_y_i = p2_px_y; // NOLINT

			if(p0_px_x_i > 0 && p0_px_x_i < target.width && p0_px_y_i > 0 && p0_px_y_i < target.height)
			{
				target.data()[p0_px_x_i + p0_px_y_i * target.width] = 0xffffffff;
			}
			if(p1_px_x_i > 0 && p1_px_x_i < target.width && p1_px_y_i > 0 && p1_px_y_i < target.height)
			{
				target.data()[p1_px_x_i + p1_px_y_i * target.width] = 0xffffffff;
			}
			if(p2_px_x_i > 0 && p2_px_x_i < target.width && p2_px_y_i > 0 && p2_px_y_i < target.height)
			{
				target.data()[p2_px_x_i + p2_px_y_i * target.width] = 0xffffffff;
			}
		}

		void render_rasterize(context& context, render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
		{
			const float target_width_flt 	= static_cast<float>(target.width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(target.height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;
			const glm::mat4x4 perspective 	= glm::perspectiveLH(camera.fov, aspect, 0.1f, 100.0f);
			const glm::mat4x4 lookat 		= glm::lookAtLH(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});

			const glm::mat4x4 projview = lookat * perspective;

			for(const triangle auto& tri : triangles)
			{
				const glm::vec4 p0 = {tri.p0.x, tri.p0.y, tri.p0.z, 1};
				const glm::vec4 p1 = {tri.p1.x, tri.p1.y, tri.p1.z, 1};
				const glm::vec4 p2 = {tri.p2.x, tri.p2.y, tri.p2.z, 1};

				const glm::vec4 p0_projview = projview * p0;
				const glm::vec4 p1_projview = projview * p1;
				const glm::vec4 p2_projview = projview * p2;

				assert(p0_projview.w != 0.0f);

				const float p0x = ((p0_projview.x / p0_projview.w) * 0.5f + 0.5f) * target_width_flt;
				const float p0y = ((p0_projview.y / p0_projview.w) * 0.5f + 0.5f) * target_height_flt;
				const float p0z = (p0_projview.z / p0_projview.w);

				const float p1x = ((p1_projview.x / p1_projview.w) * 0.5f + 0.5f) * target_width_flt;
				const float p1y = ((p1_projview.y / p1_projview.w) * 0.5f + 0.5f) * target_height_flt;
				const float p1z = (p1_projview.z / p1_projview.w);

				const float p2x = ((p2_projview.x / p2_projview.w) * 0.5f + 0.5f) * target_width_flt;
				const float p2y = ((p2_projview.y / p2_projview.w) * 0.5f + 0.5f) * target_height_flt;
				const float p2z = (p2_projview.z / p2_projview.w);

				if(p0z > 0.0f || p1z > 0.0f || p2z > 0.0f)
				{
					draw_triangle(target, tri, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z);
				}
			}
		}

		inline detail::impl& context_detail(gr::context& ctx)
		{
			if(ctx.impl == nullptr)
			{
				ctx.impl = std::make_unique<detail::impl>();
			}
			return *ctx.impl;
		}
	}

	context create_context()
	{
		return context{};
	}

	void clear(render_target auto& target)
	{
		std::span<uint32_t> target_rgb(target.data(), target.width * target.height);
		memset(target_rgb.data(), 0, target_rgb.size_bytes());
	}

	void render(context& context, render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
	{
		detail::render_rasterize(context, target, triangles, camera);
	}

	void render(render_target auto& target, const triangle_list auto& triangles, const camera auto& camera)
	{
		auto ctx = create_context();
		render(ctx, target, triangles, camera);
	}

	namespace helpers
	{
		inline managed_render_target create_render_target(size_t num_pixels_width, size_t num_pixels_height)
		{
			managed_render_target target;
			target.width = num_pixels_width;
			target.height = num_pixels_height;
			target.rgbd_buffer 	= std::unique_ptr<uint32_t>(new(std::align_val_t(32)) uint32_t[target.width * target.height]);
			return target;
		}

		cam lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov)
		{
			return cam{
				.pos = {pos.x, pos.y, pos.z},
				.lookat = {lookat.x, lookat.y, lookat.z},
				.up = {up.x, up.y, up.z},
				.fov = fov
			};
		}
	}
}



#endif //CPURAYTRACER_GT_HPP
