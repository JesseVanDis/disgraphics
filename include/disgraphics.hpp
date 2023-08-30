#ifndef CPURAYTRACER_GT_HPP
#define CPURAYTRACER_GT_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace dis
{
	template<typename T>
	concept floating_point = std::floating_point<std::decay_t<T>>;

	template<typename T>
	concept unsigned_integral = std::unsigned_integral<std::decay_t<T>>;

	template<typename T>
	concept unsigned_int_32 = std::same_as<uint32_t, std::decay_t<T>>;

	template<typename T>
	concept vector2 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
	};

	template<typename T>
	concept vector3 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
		{v.z} -> floating_point;
	};

	template<typename T>
	concept triangle_by_fields = requires(T v)
	{
		{v.p0} -> vector3;
		{v.p1} -> vector3;
		{v.p2} -> vector3;
	};

	template<typename T>
	concept triangle_by_indices = requires(T v)
	{
		{v[0]} -> vector3;
		{v[1]} -> vector3;
		{v[2]} -> vector3;
	};

	template<typename T>
	concept triangle = triangle_by_fields<T> || triangle_by_indices<T>;

	template<typename T>
	concept triangle_list = requires(T v)
	{
		{*v.begin()} -> triangle;
		{*v.end()} -> triangle;
	};

	template<triangle_list T>
	using triangle_from_list_t = decltype(*std::declval<T>().begin());

	template<size_t Index>
	constexpr auto& get_tri_pt(triangle auto& tri) requires (Index < 3)
	{
		if constexpr(requires{tri[Index];})
		{
			return tri[Index];
		}
		else if	constexpr(Index == 0)	return tri.p0;
		else if constexpr(Index == 1)	return tri.p1;
		else							return tri.p2;
	}

	template<size_t Index>
	constexpr const auto& get_tri_pt(const triangle auto& tri) requires (Index < 3)
	{
		if constexpr(requires{tri[Index];})
		{
			return tri[Index];
		}
		else if	constexpr(Index == 0)	return tri.p0;
		else if constexpr(Index == 1)	return tri.p1;
		else							return tri.p2;
	}

	template<triangle T>
	using vertex_from_tri_t = std::decay_t<decltype(get_tri_pt<0>(std::declval<T>()))>;

	template<typename T>
	concept camera = requires(T v)
	{
		{v.pos} 			-> vector3;
		{v.lookat} 			-> vector3;
		{v.up} 				-> vector3;
		{v.fov} 			-> floating_point;
	};

	template<typename T>
	concept draw_horizontal_line_ctx = requires(T v)
	{
		{v.buffer_width} 	-> unsigned_integral;
		{v.buffer_height} 	-> unsigned_integral;
		{v.px_y} 			-> unsigned_integral;
		{v.px_x_from} 		-> unsigned_integral;
		{v.line_length_px} 	-> unsigned_integral;
	};
}

namespace dis::helpers
{
	struct vec2
	{
		float x,y;
	};
	static_assert(vector2<vec2>);

	struct vec3
	{
		float x,y,z;
	};
	static_assert(vector3<vec3>);

	struct vec4
	{
		float x,y,z,w;
	};

	struct tri
	{
		vec3 p0, p1, p2;
	};
	static_assert(triangle<tri>);

	struct cam
	{
		vec3 pos, lookat, up;
		float fov;
	};
	static_assert(camera<cam>);

	using mat4x4 = std::array<std::array<float, 4>, 4>;

	namespace detail
	{
		template<unsigned int index, typename cb_t>
		inline void for_each_field(auto& a, auto& b, cb_t cb)
		{
			using a_t = std::decay_t<decltype(a)>;
			using b_t = std::decay_t<decltype(b)>;
			if constexpr(requires{a_t::template get_field<index>(a);})
			{
				if constexpr(std::is_same_v<a_t, b_t>)
				{
					cb(a_t::template get_field<index>(a), a_t::template get_field<index>(b));
				}
				else
				{
					cb(a_t::template get_field<index>(a), b);
				}
				for_each_field<index+1, cb_t>(a, b, cb);
			}
		}
	}

	template<typename cb_t>
	inline void for_each_field(auto& a, auto& b, cb_t cb)
	{
		using a_t = std::decay_t<decltype(a)>;
		static_assert(requires{a_t::template get_field<0>(std::declval<a_t&>());}, "Given type must feature a static 'get_field<int>(auto& self) function'");
		detail::for_each_field<0>(a, b, cb);
	}
}
namespace dish = dis::helpers;

namespace dis
{

	struct draw_hline_ctx
	{
		uint32_t buffer_width;
		uint32_t buffer_height;
		uint32_t px_y, px_x_from, line_length_px;
	};

	namespace detail::concepts
	{
		struct draw_horizontal_line_ctx_example
		{
			uint32_t buffer_width;
			uint32_t buffer_height;
			uint32_t px_y, px_x_from, line_length_px;
		};
		static_assert(draw_horizontal_line_ctx<draw_horizontal_line_ctx_example>);
	}

	template<typename FuncT, typename draw_ctx_t, typename triangle_t>
	concept draw_horizontal_line_function = requires(FuncT v)
	{
		{v(std::declval<triangle_t>(), std::declval<draw_ctx_t>())};
	};

	template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
	constexpr void 	render(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);

	template<triangle_list triangle_list_t>
	constexpr void 	render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_hline_ctx, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height);
}

namespace dis::helpers
{
	constexpr cam					lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov = 2.0f);
}

// implementation
#include <span>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#if __has_include(<gcem.hpp>)
#include <gcem.hpp>
#define HAS_GCEM
#endif

namespace dis
{
	namespace detail
	{
		struct impl
		{
			std::unique_ptr<float> 	managed_depth_buffer;
			size_t					managed_depth_buffer_size = 0;
		};
	}
	
	namespace detail
	{
        struct line
        {
            const dish::vec3& p0;
            const dish::vec3& p1;
        };

		template<typename T>
		concept line_custom_iterable = requires(T v)
		{
			{v += v}  -> std::same_as<T&>;
		};

		template<typename vertex_t, typename T>
		concept can_assign_vertex = requires(T v)
		{
			{v = std::declval<vertex_t>()} -> std::same_as<T&>;
		};

		template<typename draw_ctx_t>
		concept has_user_defined_iterators = requires(draw_ctx_t v)
		{
			{v.begin} -> line_custom_iterable;
		};

		template<vector3 vertex_t>
		struct screen_space_vertex
		{
			dish::vec3 screen_pos;
			vertex_t  vertex;
		};

		template<typename user_defined_iterators_t>
		struct screen_space_clipped_pt
		{
			dish::vec3 screen_pos;
			user_defined_iterators_t user_defined; // xyz or uv ect... ( they need to be clipped as well )

			screen_space_clipped_pt& operator -= (const screen_space_clipped_pt& other)
			{
				screen_pos.x -= other.screen_pos.x;
				screen_pos.y -= other.screen_pos.y;
				screen_pos.z -= other.screen_pos.z;
				auto temp = user_defined;
				user_defined = other.user_defined;
				user_defined *= -1.0f;
				user_defined += temp;
				return *this;
			}

			screen_space_clipped_pt& operator += (const screen_space_clipped_pt& other)
			{
				screen_pos.x += other.screen_pos.x;
				screen_pos.y += other.screen_pos.y;
				screen_pos.z += other.screen_pos.z;
				user_defined += other.user_defined;
				return *this;
			}

			screen_space_clipped_pt& operator *= (const screen_space_clipped_pt& other)
			{
				screen_pos.x *= other.screen_pos.x;
				screen_pos.y *= other.screen_pos.y;
				screen_pos.z *= other.screen_pos.z;
				user_defined *= other.user_defined;
				return *this;
			}

			screen_space_clipped_pt& operator *= (float v)
			{
				screen_pos.x *= v;
				screen_pos.y *= v;
				screen_pos.z *= v;
				user_defined *= v;
				return *this;
			}

			screen_space_clipped_pt operator - (const screen_space_clipped_pt& other) const
			{
				screen_space_clipped_pt v = *this;
				v -= other;
				return v;
			}

			screen_space_clipped_pt operator + (const screen_space_clipped_pt& other) const
			{
				screen_space_clipped_pt v = *this;
				v += other;
				return v;
			}

			screen_space_clipped_pt operator * (const screen_space_clipped_pt& other) const
			{
				screen_space_clipped_pt v = *this;
				v *= other;
				return v;
			}

			screen_space_clipped_pt operator * (float value) const
			{
				screen_space_clipped_pt v = *this;
				v *= value;
				return v;
			}
		};

		template<typename user_defined_iterators_t>
		using screen_space_clipped_pt_triangle = std::array<screen_space_clipped_pt<user_defined_iterators_t>, 3>;

		template<triangle triangle_t>
		using screen_space_triangle = std::array<screen_space_vertex<vertex_from_tri_t<triangle_t>>, 3>;

		struct line_it_base
		{
			int y_start, height;

			float x_it, x;
			float z_it, z;

			void increment_base()
			{
				x += x_it;
				z += z_it;
			}

			struct set_precalculated
			{
				float y_start_ceiled, height_ceiled, one_over_height_ceiled, sub_pixel;
			};

			set_precalculated set_base(const vector3 auto& p0_screen_pos, const vector3 auto& p1_screen_pos)
			{
				set_precalculated c {
					.y_start_ceiled 		= std::ceil(p0_screen_pos.y),
					.height_ceiled 			= std::ceil(p1_screen_pos.y) - c.y_start_ceiled,
					.one_over_height_ceiled = c.height_ceiled != 0.0f ? (1.0f / c.height_ceiled) : 0.0f,
					.sub_pixel 				= c.y_start_ceiled - p0_screen_pos.y
				};

				//assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN

				y_start    = static_cast<int>(c.y_start_ceiled);
				height     = static_cast<int>(c.height_ceiled);
				x_it 		= (p1_screen_pos.x - p0_screen_pos.x) * c.one_over_height_ceiled;
				x			= p0_screen_pos.x + (x_it * c.sub_pixel);
				z_it 		= (p1_screen_pos.z - p0_screen_pos.z) * c.one_over_height_ceiled;
				z			= p0_screen_pos.z + (z_it * c.sub_pixel);

				return c;
			}
		};

		//struct UV
		//{
		//	float u,v;
		//};

		template<typename user_defined_iterators_t, typename enable = void>
		struct line_it;

		template<typename user_defined_iterators_t>
		struct line_it<user_defined_iterators_t, std::enable_if_t<line_custom_iterable<user_defined_iterators_t>>> : line_it_base
		{
			user_defined_iterators_t user_defined;
			user_defined_iterators_t user_defined_it;

			float one_over_z;

			//UV uv_over_z;
			//UV uv_over_z_it;

			void increment()
			{
				increment_base();
				user_defined += user_defined_it;
				//uv_over_z.u += uv_over_z_it.u;
				//uv_over_z.v += uv_over_z_it.v;
			}

			void set(const screen_space_clipped_pt<user_defined_iterators_t>& p0, const screen_space_clipped_pt<user_defined_iterators_t>& p1)
			{
				const set_precalculated c = set_base(p0.screen_pos, p1.screen_pos);

				using float_t = decltype(c.one_over_height_ceiled);

				const float one_over_z_end 	= 1.0f / p1.screen_pos.z;
				one_over_z 					= 1.0f / p0.screen_pos.z;

				// below is equal to:
				//   start = p0 * one_over_z
				//   end   = p1 * one_over_z_end
				//   v_it  = (end - start) * c.one_over_height_ceiled
				//   v     = start + v_it * c.sub_pixel

				user_defined_iterators_t inv_start;
				inv_start = p0.user_defined;
				inv_start *= -one_over_z;

				user_defined_it = p1.user_defined;
				user_defined_it *= one_over_z_end;
				user_defined_it += inv_start;
				user_defined_it *= c.one_over_height_ceiled;

				inv_start *= (float_t)-1;
				user_defined = user_defined_it;
				user_defined *= c.sub_pixel;
				user_defined += inv_start;

				//const UV uv_over_z_start = {p0.vertex.u * one_over_z, p0.vertex.v * one_over_z};
				//const UV uv_over_z_end = {p1.vertex.u * one_over_z_end, p1.vertex.v * one_over_z_end};

				//uv_over_z_it.u = (uv_over_z_end.u - uv_over_z_start.u) * c.one_over_height_ceiled;
				//uv_over_z_it.v = (uv_over_z_end.v - uv_over_z_start.v) * c.one_over_height_ceiled;
				//uv_over_z.u = uv_over_z_start.u + uv_over_z_it.u * c.sub_pixel;
				//uv_over_z.v = uv_over_z_start.v + uv_over_z_it.v * c.sub_pixel;
			}
		};

		template<typename user_defined_iterators_t>
        struct line_it<user_defined_iterators_t, std::enable_if_t<!line_custom_iterable<user_defined_iterators_t>>> : line_it_base
        {
			void increment()
			{
				increment_base();
			}

			void set(const screen_space_clipped_pt<user_defined_iterators_t>& p0, const screen_space_clipped_pt<user_defined_iterators_t>& p1)
			{
				set_base(p0.screen_pos, p1.screen_pos);
			}
		};

		template<typename VecA, typename VecB>
		using smallest_vec3_t = std::conditional_t<sizeof(VecA) < sizeof(VecB), VecA, VecB>;

		template<draw_horizontal_line_ctx draw_ctx_t>
		inline constexpr void check_context_validity(draw_ctx_t& ctx)
		{
#if 0
			int to = ctx.px_x_from + ctx.line_length_px;
			assert(to 				>= ctx.px_x_from);
			assert(ctx.px_y 		< ctx.buffer_height);
			assert(ctx.px_x_from 	< ctx.buffer_width);
			assert(to				< ctx.buffer_width);
#else
			ctx.px_y 			= std::clamp(ctx.px_y, 		0u, (uint32_t)(ctx.buffer_height-1));
			ctx.px_x_from 		= std::clamp(ctx.px_x_from, 0u, (uint32_t)(ctx.buffer_width-1));
			int to = std::clamp(ctx.px_x_from + ctx.line_length_px, 0u, (uint32_t)(ctx.buffer_width-1));
			if(to <= ctx.px_x_from)
			{
				ctx.line_length_px = 0;
			}
			else
			{
				ctx.line_length_px = to - ctx.px_x_from;
			}

			//ctx.px_y 			= std::min(ctx.px_y, (ctx.buffer_height-1));
			//ctx.px_x_from 		= std::min(ctx.px_x_from, (ctx.buffer_width-1));
			//ctx.line_length_px 	= (ctx.px_x_from + ctx.line_length_px) > (ctx.buffer_width-1) ? ((ctx.buffer_width-1) - ctx.px_x_from) : ctx.line_length_px;
#endif
		}

		template<draw_horizontal_line_ctx draw_ctx_t, typename user_defined_iterators_t, triangle triangle_t>
		constexpr inline void it_line(const triangle_t& source_triangle, draw_ctx_t& ctx, int y, line_it<user_defined_iterators_t>& left, line_it<user_defined_iterators_t>& right, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function)
		{
			ctx.px_y 			= y;
			ctx.px_x_from 		= static_cast<int>(left.x);
			ctx.line_length_px 	= static_cast<int>(right.x) - ctx.px_x_from;

			check_context_validity(ctx);

			if constexpr(has_user_defined_iterators<draw_ctx_t>)
			{
				const float width = right.x - left.x;
				const float one_over_width = 1.0f / width;
				const float sub_texel = std::floor(left.x) - left.x;

				const float one_over_z_it = (right.one_over_z - left.one_over_z) * one_over_width;
				const float one_over_z 	= left.one_over_z + (one_over_z_it * sub_texel);

				const float z = (1.0f/one_over_z);

				//ctx.it 	= (right.user_defined - left.user_defined) * one_over_width;
				ctx.it = left.user_defined;
				ctx.it *= -1.0f;
				ctx.it += right.user_defined;
				ctx.it *= one_over_width;

				//ctx.begin = (left.user_defined + (ctx.it * sub_texel)) * z;
				ctx.begin = ctx.it;
				ctx.begin *= sub_texel;
				ctx.begin += left.user_defined;
				ctx.begin *= z;

				if constexpr(requires{ctx.one_over_z;})
				{
					ctx.one_over_z 		= one_over_z;
					ctx.one_over_z_it 	= one_over_z_it;
				}
			}

			draw_hline_function(source_triangle, ctx);

			left.increment();
			right.increment();
		}

		template<typename user_defined_iterators_t>
		constexpr inline int y_pos_at_intersection(int current_ypos, const line_it<user_defined_iterators_t>& a, const line_it<user_defined_iterators_t>& b)
		{
			if(b.x_it - a.x_it == 0.0f)
			{
				return (current_ypos-1);
			}
			//assert((b.x_it - a.x_it) != 0.0f);
			int lowest_y 		= std::min(a.y_start + a.height, b.y_start + b.height);
			int iterations_left = static_cast<int>(std::floor((a.x - b.x) / (b.x_it - a.x_it)));
			int y_limit 		= current_ypos + iterations_left;
			return std::min(y_limit, lowest_y);
		}

		/// no bounds checking here!
		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t>
		constexpr void draw_triangle_unsafe(const triangle_t& source_triangle, screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			if constexpr(requires{std::declval<draw_ctx_t>().begin;})
			{
				static_assert(has_user_defined_iterators<draw_ctx_t>, "'begin' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
			}

            struct line {const screen_space_clipped_pt<user_defined_iterators_t>& p0, &p1;};

            std::sort(triangle.begin(), triangle.end(), [](const auto& a, const auto& b){return a.screen_pos.y < b.screen_pos.y;});

            // take lines
            const line line_long  = {triangle[0], triangle[2]};
            const line line_top   = {triangle[0], triangle[1]};
            const line line_bot   = {triangle[1], triangle[2]};

            // check whether the long line is on the left or right
            float cross_z = (triangle[1].screen_pos.x - triangle[0].screen_pos.x) * (triangle[2].screen_pos.y - triangle[0].screen_pos.y) - (triangle[2].screen_pos.x - triangle[0].screen_pos.x) * (triangle[1].screen_pos.y - triangle[0].screen_pos.y);

			draw_ctx_t ctx; // NOLINT
			ctx.buffer_width = static_cast<uint32_t>(frame_width);
			ctx.buffer_height = static_cast<uint32_t>(frame_height);

			if(cross_z > 0.0f)
			{
				line_it<user_defined_iterators_t> line_it_long, line_it_top, line_it_bot;

				line_it_long.set(line_long.p0, line_long.p1);
				line_it_top.set(line_top.p0, line_top.p1);
				line_it_bot.set(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_top, draw_hline_function);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_bot, draw_hline_function);
				}
			}
			else
			{
				line_it<user_defined_iterators_t> line_it_long, line_it_top, line_it_bot;

				line_it_long.set(line_long.p0, line_long.p1);
				line_it_top.set(line_top.p0, line_top.p1);
				line_it_bot.set(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					it_line(source_triangle, ctx, y, line_it_top, line_it_long, draw_hline_function);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_bot, line_it_long, draw_hline_function);
				}
			}
		}

		/// does not check for 0 division
		template<vector3 vector3_t>
		constexpr vector3_t normalize(const vector3_t& v)
		{
			const auto l = static_cast<std::decay_t<decltype(v.x)>>(1) / std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
			return {v.x / l, v.y / l, v.z / l};
		}

		/// does not check for 0 division
		template<vector3 vector3_t>
		constexpr vector3_t direction_to(const vector3_t& from, const vector3_t& to)
		{
			return normalize(vector3_t{to.x - from.x, to.y - from.y, to.z - from.z});
		}

		/// does not check for 0 division
		constexpr auto length(const vector3 auto& vec)
		{
			return std::sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
		}

		constexpr dish::vec3 cross_xyz(const vector3 auto& a, const vector3 auto& b)
		{
			return {a.b * b.z - b.b * a.z, a.z * b.a - b.z * a.a, a.a * b.b - b.a * a.b};
		}

		constexpr floating_point auto dot_xyz(const vector3 auto& a, const vector3 auto& b)
		{
			return a.x * b.x + a.y * b.y + a.z * b.z;
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> sub_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x - b.x, a.y - b.y, a.z - b.z};
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> add_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x + b.x, a.y + b.y, a.z + b.z};
		}

		template<vector3 Vec3AT, vector3 Vec3BT>
		constexpr smallest_vec3_t<Vec3AT, Vec3BT> mul_xyz(const Vec3AT& a, const Vec3BT& b)
		{
			return {a.x * b.x, a.y * b.y, a.z * b.z};
		}

		template<vector3 Vec3T>
		constexpr Vec3T mul_xyz(const Vec3T& a, float b)
		{
			return {a.x * b, a.y * b, a.z * b};
		}

		constexpr dish::mat4x4 mul(const dish::mat4x4& m1, const dish::mat4x4& m2)
		{
			dish::mat4x4 result;
			for(size_t k=0; k<4; k++) for(size_t i=0; i<4; i++)
				{
					result[k][i] = m1[0][i] * m2[k][0] + m1[1][i] * m2[k][1] + m1[2][i] * m2[k][2] + m1[3][i] * m2[k][3];
				}
			return result;
		}

		constexpr dish::vec4 mul(const dish::mat4x4& m1, const dish::vec4& m2)
		{
			dish::vec4 result = {};
			result.x = m1[0][0] * m2.x + m1[1][0] * m2.y + m1[2][0] * m2.z + m1[3][0] * m2.w;
			result.y = m1[0][1] * m2.x + m1[1][1] * m2.y + m1[2][1] * m2.z + m1[3][1] * m2.w;
			result.z = m1[0][2] * m2.x + m1[1][2] * m2.y + m1[2][2] * m2.z + m1[3][2] * m2.w;
			result.w = m1[0][3] * m2.x + m1[1][3] * m2.y + m1[2][3] * m2.z + m1[3][3] * m2.w;
			return result;
		}

		constexpr auto& get_tri_p0(triangle auto& tri) { return get_tri_pt<0>(tri); }
		constexpr auto& get_tri_p1(triangle auto& tri) { return get_tri_pt<1>(tri); }
		constexpr auto& get_tri_p2(triangle auto& tri) { return get_tri_pt<2>(tri); }

		constexpr floating_point auto intersect(const vector3 auto& ray_origin, const vector3 auto& ray_dir, const vector3 auto& plane_pos, const vector3 auto& plane_normal)
		{
			const floating_point auto denom = (plane_normal.x * ray_dir.x) + (plane_normal.y * ray_dir.y) + (plane_normal.z * ray_dir.z);
			if ((denom*denom) > static_cast<decltype(denom)>(0.0001 * 0.0001)) // your favorite epsilon
			{
				const vector3 auto 			d = sub_xyz(plane_pos, ray_origin);
				const floating_point auto 	d_dot = dot_xyz(d, plane_normal);
				return d_dot / denom;
			}
			return static_cast<decltype(denom)>(-1.0f);
		}

		//using vertex_t = vertex_from_tri_t<triangle_t>;
		//using screen_space_vertex_t = screen_space_vertex<vertex_t>;
		//constexpr auto tris_capacity = (2*2*2*2)+1;

		//std::array<std::array<screen_space_vertex_t, 3>, tris_capacity>		clipped_tris; // NOLINT

		template<typename user_defined_iterators_t>
		constexpr void clip_user_defined_iterators(const user_defined_iterators_t& from, user_defined_iterators_t& to_mut, float perc)
		{
			user_defined_iterators_t diff = from;
			diff *= -1.0f;
			diff += to_mut;
			to_mut = diff;
			to_mut *= perc;
			to_mut += from;
		}

		template<typename user_defined_iterators_t>
		constexpr int clip_triangle(const vector3 auto& plane_pos, const vector3 auto& plane_normal, screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_in_out, screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_extra_out)
		{
			using point_t = screen_space_clipped_pt<user_defined_iterators_t>;

			const point_t& a_pt = tri_in_out[0];
			const point_t& b_pt = tri_in_out[1];
			const point_t& c_pt = tri_in_out[2];

			const point_t ab_pt = b_pt - a_pt;
			const point_t bc_pt = c_pt - b_pt;
			const point_t ca_pt = a_pt - c_pt;

			const floating_point auto ab_len = length(ab_pt.screen_pos);
			const floating_point auto bc_len = length(bc_pt.screen_pos);
			const floating_point auto ca_len = length(ca_pt.screen_pos);

			const floating_point auto ab_len_inv = 1.0f / ab_len;
			const floating_point auto bc_len_inv = 1.0f / bc_len;
			const floating_point auto ca_len_inv = 1.0f / ca_len;

			const vector3 auto ab_dir = mul_xyz(ab_pt.screen_pos, ab_len_inv); // TODO: This can be precalculated
			const vector3 auto bc_dir = mul_xyz(bc_pt.screen_pos, bc_len_inv); // TODO: This can be precalculated
			const vector3 auto ca_dir = mul_xyz(ca_pt.screen_pos, ca_len_inv); // TODO: This can be precalculated

			const floating_point auto a_to_b_t = intersect(a_pt.screen_pos, ab_dir, plane_pos, plane_normal);
			const floating_point auto b_to_c_t = intersect(b_pt.screen_pos, bc_dir, plane_pos, plane_normal);
			const floating_point auto c_to_a_t = intersect(c_pt.screen_pos, ca_dir, plane_pos, plane_normal);

			const bool intersects_a_to_b = (a_to_b_t > 0 && a_to_b_t < ab_len);
			const bool intersects_b_to_c = (b_to_c_t > 0 && b_to_c_t < bc_len);
			const bool intersects_c_to_a = (c_to_a_t > 0 && c_to_a_t < ca_len);

			const float ab_perc = a_to_b_t * ab_len_inv;
			const float bc_perc = b_to_c_t * bc_len_inv;
			const float ca_perc = c_to_a_t * ca_len_inv;

			point_t intersection_a_to_b_pt = a_pt + (ab_pt * ab_perc);
			point_t intersection_b_to_c_pt = b_pt + (bc_pt * bc_perc);
			point_t intersection_c_to_a_pt = c_pt + (ca_pt * ca_perc);

			const vector3 auto& triangle_tip  = (intersects_a_to_b && intersects_c_to_a) ? a_pt.screen_pos : ((intersects_a_to_b && intersects_b_to_c) ? b_pt.screen_pos : c_pt.screen_pos);
			const vector3 auto& any_other_tip = (intersects_a_to_b && intersects_c_to_a) ? b_pt.screen_pos : ((intersects_a_to_b && intersects_b_to_c) ? c_pt.screen_pos : a_pt.screen_pos);

			if(intersects_a_to_b || intersects_b_to_c || intersects_c_to_a)
			{
				const vector3 auto	tip_to_tip			= sub_xyz(triangle_tip, any_other_tip);
				const bool 			should_cut_to_quad 	= (tip_to_tip.x*plane_normal.x + tip_to_tip.y*plane_normal.y + tip_to_tip.z*plane_normal.z) < 0.0f;

				struct from_to
				{
					std::uint_fast8_t from;
					std::uint_fast8_t to;
					float perc;
				};

				const point_t points[]  	= {a_pt, b_pt, c_pt, intersection_a_to_b_pt, intersection_b_to_c_pt, intersection_c_to_a_pt};

				struct // NOLINT
				{
					std::uint_fast8_t indices[4]; // corresponds to the index of 'points'. so max num is 5  ( points.size() - 1 )
					std::uint_fast8_t num_indices = 2;
				} constexpr s_point_indices[] =
						{
								{ }, 					// 	0, triangle has no intersection after all ?
								{ { 2, 5, 4}, 3 }, 		// 	1, intersection between c->a and c->b, cut out triangle
								{ { 0, 3, 5}, 3 }, 		// 	2, intersection between a->b and a->c, cut out triangle
								{ { 1, 3, 4}, 3 }, 		// 	3, intersection between b->a and b->c, cut out triangle
								{ }, 					// 	4, triangle has no intersection after all ?
								{ { 0, 1, 4, 5}, 4 }, 	// 	5, intersection between c->a and c->b, cut out quad
								{ { 1, 2, 5, 3}, 4 }, 	// 	6, intersection between a->b and a->c, cut out quad
								{ { 2, 0, 3, 4}, 4 }, 	// 	7, intersection between b->a and b->c, cut out quad
						};

				const unsigned int state = ((!intersects_a_to_b ? 0b0001u : 0u) | (!intersects_b_to_c ? 0b0010u : 0u) | (!intersects_c_to_a ? 0b0011u : 0u)) + (should_cut_to_quad ? 0b0100u : 0u);
				const auto& indices = s_point_indices[state];

				unsigned int num_tris = indices.num_indices - 2;
				assert(num_tris == 1 || num_tris == 2);
				std::uint_fast8_t t0_p0_index = indices.indices[((0<<1u)+0u)&0b11u];
				std::uint_fast8_t t0_p1_index = indices.indices[((0<<1u)+1u)&0b11u];
				std::uint_fast8_t t0_p2_index = indices.indices[((0<<1u)+2u)&0b11u];

				tri_in_out[0] = points[t0_p0_index];
				tri_in_out[1] = points[t0_p1_index];
				tri_in_out[2] = points[t0_p2_index];

				if(num_tris == 2)
				{
					std::uint_fast8_t t1_p0_index = indices.indices[((1<<1u)+0u)&0b11u];
					std::uint_fast8_t t1_p1_index = indices.indices[((1<<1u)+1u)&0b11u];
					std::uint_fast8_t t1_p2_index = indices.indices[((1<<1u)+2u)&0b11u];

					tri_extra_out[0] = points[t1_p0_index];
					tri_extra_out[1] = points[t1_p1_index];
					tri_extra_out[2] = points[t1_p2_index];
					return 1;
				}
			}
			else
			{
				return dot_xyz(sub_xyz(c_pt.screen_pos, plane_pos), plane_normal) > 0 ? 0 : -1;
			}
			return 0;
		}

		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height,
									 unsigned_integral auto viewport_x_start, unsigned_integral auto viewport_y_start, unsigned_integral auto viewport_x_end, unsigned_integral auto viewport_y_end)
		{
			using clipped_tri = screen_space_clipped_pt_triangle<user_defined_iterators_t>;
			constexpr auto tris_capacity = (2*2*2*2)+1;

			std::array<clipped_tri, tris_capacity>		clipped_tris; // NOLINT
			std::uint_fast8_t 							clipped_tris_num = 0;

			// first just add the main triangle
			clipped_tris[clipped_tris_num++] = triangle;

			// clipping planes
			// top clipping plane
			{
				const dish::vec3 n = {0, -1, 0};
				const dish::vec3 o = {0, static_cast<float>(viewport_y_end), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y < viewport_y_end, tri[1].screen_pos.y < viewport_y_end, tri[2].screen_pos.y < viewport_y_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// bot clipping plane
			{
				const dish::vec3 n = {0, 1, 0};
				const dish::vec3 o = {0, static_cast<float>(viewport_y_start), 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y > viewport_y_start, tri[1].screen_pos.y > viewport_y_start, tri[2].screen_pos.y > viewport_y_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// left clipping plane
			{
				const dish::vec3 n = {1, 0, 0};
				const dish::vec3 o = {static_cast<float>(viewport_x_start), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x > viewport_x_start, tri[1].screen_pos.x > viewport_x_start, tri[2].screen_pos.x > viewport_x_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// right clipping plane
			{
				const dish::vec3 n = {-1, 0, 0};
				const dish::vec3 o = {static_cast<float>(viewport_x_end), 0, 0};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x < viewport_x_end, tri[1].screen_pos.x < viewport_x_end, tri[2].screen_pos.x < viewport_x_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(o, n, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// draw
			for(std::uint_fast8_t i=0; i<clipped_tris_num; i++)
			{
				draw_triangle_unsafe<draw_ctx_t>(source_triangle, clipped_tris[i], draw_hline_function, frame_width, frame_height);
			}
		}

		template<draw_horizontal_line_ctx draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, draw_horizontal_line_function<draw_ctx_t, triangle_t> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			draw_triangle<draw_ctx_t, triangle_t, user_defined_iterators_t>(source_triangle, triangle, draw_hline_function, frame_width, frame_height, 1u, 1u, (frame_width-1), (frame_height-1));
		}

		constexpr dish::mat4x4 create_perspective(float fovy, float aspect, float z_near, float z_far)
		{
			assert(std::abs(aspect - std::numeric_limits<float>::epsilon()) > 0.0f);

			const float fovy_over_2 = fovy / 2.0f;

#ifndef HAS_GCEM
			const float tanHalfFovy = std::tan(fovy_over_2);
#else

			const float tanHalfFovy = std::is_constant_evaluated() ? gcem::tan(fovy_over_2) : std::tan(fovy_over_2);
#endif

			dish::mat4x4 result = {};
			result[0][0] = 1.0f / (aspect * tanHalfFovy);
			result[1][1] = 1.0f / (tanHalfFovy);
			result[2][2] = (z_far + z_near) / (z_far - z_near);
			result[2][3] = 1.0f;
			result[3][2] = - (2.0f * z_far * z_near) / (z_far - z_near);
			//result[3][3] = 1.0f; maybe ?
			return result;
		}

		template<vector3 vector3_t>
		constexpr dish::mat4x4 create_lookat(const vector3_t& eye, const vector3_t& center, const vector3_t& up)
		{
			const vector3_t f(normalize(sub_xyz(center,eye)));
			const vector3_t s(normalize(cross(up, f)));
			const vector3_t u(cross(f, s));

			dish::mat4x4 result = {};
			result[0][0] = s.x;
			result[1][0] = s.y;
			result[2][0] = s.z;
			result[0][1] = u.x;
			result[1][1] = u.y;
			result[2][1] = u.z;
			result[0][2] = f.x;
			result[1][2] = f.y;
			result[2][2] = f.z;
			result[3][0] = -dot(s, eye);
			result[3][1] = -dot(u, eye);
			result[3][2] = -dot(f, eye);
			result[3][3] = 1.0f;
			return result;
		}

		template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
		constexpr void render_rasterize(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using triangle_t 				= std::decay_t<decltype(*triangles.begin())>;
			using user_defined_iterators_t 	= std::conditional_t<has_user_defined_iterators<draw_ctx_t>, std::decay_t<decltype(std::declval<draw_ctx_t>().begin)>, std::nullptr_t>;
			using clipped_tri_t 				= screen_space_clipped_pt_triangle<user_defined_iterators_t>;


			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;

			const float near_plane 	= 0.1f;
			const float far_plane 	= 100.0f;

			const dish::mat4x4 perspective 	= create_perspective(camera.fov, aspect, near_plane / 10.0f, far_plane);
			const dish::mat4x4 lookat 		= create_lookat(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});
			const dish::mat4x4 projview 		= mul(perspective, lookat);

			const vector3 auto near_clipping_plane_normal 	= direction_to(camera.pos, camera.lookat);
			const vector3 auto near_clipping_plane_pos 		= add_xyz(camera.pos, mul_xyz(near_clipping_plane_normal, near_plane));

			for(const triangle auto& tri : triangles)
			{
				std::array<clipped_tri_t, 2> clipped_tris; // NOLINT
				size_t num_clipped_tris = 0;

				const auto& vertex0 = get_tri_pt<0>(tri);
				const auto& vertex1 = get_tri_pt<1>(tri);
				const auto& vertex2 = get_tri_pt<2>(tri);

				clipped_tris[0][0].user_defined = vertex0;
				clipped_tris[0][1].user_defined = vertex1;
				clipped_tris[0][2].user_defined = vertex2;
				clipped_tris[0][0].screen_pos = {vertex0.x, vertex0.y, vertex0.z};
				clipped_tris[0][1].screen_pos = {vertex1.x, vertex1.y, vertex1.z};
				clipped_tris[0][2].screen_pos = {vertex2.x, vertex2.y, vertex2.z};
				num_clipped_tris++;

				// clip near
				num_clipped_tris += clip_triangle<user_defined_iterators_t>(near_clipping_plane_pos, near_clipping_plane_normal, clipped_tris[0], clipped_tris[1]);

				for(size_t clipped_tri_index=0; clipped_tri_index < num_clipped_tris; clipped_tri_index++)
				{
					clipped_tri_t& clipped_tri = clipped_tris[clipped_tri_index];

					const dish::vec4 p0 = {clipped_tri[0].screen_pos.x, clipped_tri[0].screen_pos.y, clipped_tri[0].screen_pos.z, 1};
					const dish::vec4 p1 = {clipped_tri[1].screen_pos.x, clipped_tri[1].screen_pos.y, clipped_tri[1].screen_pos.z, 1};
					const dish::vec4 p2 = {clipped_tri[2].screen_pos.x, clipped_tri[2].screen_pos.y, clipped_tri[2].screen_pos.z, 1};

					const dish::vec4 p0_projview = mul(projview, p0);
					const dish::vec4 p1_projview = mul(projview, p1);
					const dish::vec4 p2_projview = mul(projview, p2);

					assert(p0_projview.w != 0.0f);

					clipped_tri[0].screen_pos.x = ((p0_projview.x / p0_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tri[0].screen_pos.y = ((p0_projview.y / p0_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tri[0].screen_pos.z = (p0_projview.z / p0_projview.w);

					clipped_tri[1].screen_pos.x = ((p1_projview.x / p1_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tri[1].screen_pos.y = ((p1_projview.y / p1_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tri[1].screen_pos.z = (p1_projview.z / p1_projview.w);

					clipped_tri[2].screen_pos.x = ((p2_projview.x / p2_projview.w) * 0.5f + 0.5f) * target_width_flt;
					clipped_tri[2].screen_pos.y = ((p2_projview.y / p2_projview.w) * 0.5f + 0.5f) * target_height_flt;
					clipped_tri[2].screen_pos.z = (p2_projview.z / p2_projview.w);

					const float cross_z = (clipped_tri[1].screen_pos.x - clipped_tri[0].screen_pos.x) * (clipped_tri[2].screen_pos.y - clipped_tri[0].screen_pos.y) - (clipped_tri[2].screen_pos.x - clipped_tri[0].screen_pos.x) * (clipped_tri[1].screen_pos.y - clipped_tri[0].screen_pos.y);
					const bool backface_culling = cross_z > 0.0f;

					if(backface_culling)
					{
						draw_triangle<draw_ctx_t, triangle_t, user_defined_iterators_t>(tri, clipped_tri, draw_hline_function, frame_width, frame_height);
					}
				}
			}
		}
	}

	template<draw_horizontal_line_ctx draw_ctx_t, triangle_list triangle_list_t>
	constexpr void render(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_ctx_t, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		using triangle_t 				= triangle_from_list_t<triangle_list_t>;
		using vertex_t					= vertex_from_tri_t<triangle_t>;
		if constexpr(requires{std::declval<draw_ctx_t>().begin;})
		{
			using user_defined_iterators_t = std::decay_t<decltype(std::declval<draw_ctx_t>().begin)>;
			static_assert(detail::can_assign_vertex<vertex_t, user_defined_iterators_t>, 	"'begin' member found in 'draw_ctx_t' but cannot assign a tri to it");
			static_assert(detail::has_user_defined_iterators<draw_ctx_t>, 					"'begin' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
		}

		detail::render_rasterize<draw_ctx_t>(triangles, camera, draw_hline_function, frame_width, frame_height);
	}

	template<triangle_list triangle_list_t>
	constexpr void render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, draw_horizontal_line_function<draw_hline_ctx, triangle_from_list_t<triangle_list_t>> auto&& draw_hline_function, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		render<draw_hline_ctx, triangle_list_t>(triangles, camera, draw_hline_function, frame_width, frame_height);
	}

	namespace helpers
	{
		constexpr cam lookat(const vector3 auto& pos, const vector3 auto& lookat, const vector3 auto& up, const std::floating_point auto& fov)
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
