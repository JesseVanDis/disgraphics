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
	concept arithmetic = std::is_arithmetic_v<std::decay_t<T>>;

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
	concept vector4 = requires(T v)
	{
		{v.x} -> floating_point;
		{v.y} -> floating_point;
		{v.z} -> floating_point;
		{v.w} -> floating_point;
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

	//template<typename T>
	//concept draw_horizontal_line_ctx = requires(T v)
	//{
		//{v.buffer_width} 	-> unsigned_integral;
		//{v.buffer_height} 	-> unsigned_integral;
		//{v.px_y} 			-> unsigned_integral;
		//{v.px_x_begin} 		-> unsigned_integral;
		//{v.line_length_px} 	-> unsigned_integral;
	//};

	namespace detail
	{
		template<typename depth_t, typename px_index_t>
		struct base_fragment_vars
		{
			using user_t = std::nullptr_t;

			depth_t 	depth;
			px_index_t 	px_index;
		};

		template<typename depth_t, typename px_index_t, typename user_vars_t>
		struct base_fragment_vars_with_user : user_vars_t
		{
			using user_t = user_vars_t;

			depth_t 	depth;
			px_index_t 	px_index;
		};
	}

	template<typename T>
	concept fragment_vars = requires(T v)
	{
		{v.depth} 			-> floating_point;
		{v.px_index} 		-> unsigned_integral;
	};

	template<typename UserVars = std::nullptr_t>
	using frag_vars = std::conditional_t<std::is_same_v<UserVars, std::nullptr_t>, detail::base_fragment_vars<float, uint32_t>, detail::base_fragment_vars_with_user<float, uint32_t, UserVars>>;
	static_assert(fragment_vars<frag_vars<>>);
}

namespace dis::helpers
{
	template<typename T>
	struct vec2
	{
		T x,y;
	};
	static_assert(vector2<vec2<float>>);

	template<typename T>
	struct vec3
	{
		T x,y,z;
	};
	static_assert(vector3<vec3<float>>);

	template<typename T>
	struct vec4
	{
		T x,y,z,w;
	};
	static_assert(vector4<vec4<float>>);

	template<typename T>
	struct tri
	{
		vec3<T> p0, p1, p2;
	};
	static_assert(triangle<tri<float>>);

	struct cam
	{
		vec3<float> pos, lookat, up;
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
	template<typename user_defined_iterators_t>
	struct draw_ctx
	{
		float 						one_over_z    = 0;
		float 						one_over_z_it = 0;
		user_defined_iterators_t 	vertex;
		user_defined_iterators_t 	vertex_it;
	};

	namespace detail
	{
		template<typename draw_ctx_t>
		using user_defined_iterators_from_draw_ctx = std::decay_t<decltype(std::declval<draw_ctx_t>().vertex)>;

		template<typename shader_t, typename triangle_t>
		using user_defined_iterators_from_shader = std::decay_t<decltype(std::declval<shader_t>().vertex(std::declval<vertex_from_tri_t<triangle_t>>()))>;

	}

	namespace detail::concepts
	{
		template<typename shader_t, typename triangle_t>
		concept shader_scanline = requires(shader_t v)
		{
			{v.vertex(std::declval<vertex_from_tri_t<triangle_t>>())} -> std::same_as<detail::user_defined_iterators_from_shader<shader_t, triangle_t>>;
			{v.scanline(std::declval<triangle_t>(), std::declval<draw_ctx<detail::user_defined_iterators_from_shader<shader_t, triangle_t>>>())};
		};

		template<typename shader_t, typename triangle_t>
		concept shader_fragment = requires(shader_t v)
		{
			{v.vertex(std::declval<vertex_from_tri_t<triangle_t>>())} -> std::same_as<detail::user_defined_iterators_from_shader<shader_t, triangle_t>>;
			{v.fragment(std::declval<triangle_t>(), std::declval<frag_vars<detail::user_defined_iterators_from_shader<shader_t, triangle_t>>>())};
		};
	}

	template<typename shader_t, typename triangle_t>
	concept shader = detail::concepts::shader_scanline<shader_t, triangle_t> || detail::concepts::shader_fragment<shader_t, triangle_t>;

	template<triangle_list triangle_list_t>
	constexpr void 	render(const triangle_list_t& triangles, const camera auto& camera, shader<triangle_from_list_t<triangle_list_t>> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height);

	//template<triangle_list triangle_list_t>
	//constexpr void 	render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, unsigned_integral auto frame_width, unsigned_integral auto frame_height);
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
		template<typename T>
		concept line_custom_iterable_trough_foreach = requires(T v)
		{
			std::decay_t<T>::template get_field<0>(v) = {};
		};

		template<typename T>
		concept line_custom_iterable_trough_operators = requires(T v)
		{
			{v += v}  -> std::same_as<T&>;
		};

		template<typename T>
		concept line_custom_iterable = line_custom_iterable_trough_operators<T> || line_custom_iterable_trough_foreach<T>;

		//template<typename vertex_t, typename T>
		//concept can_assign_vertex = requires(T v)
		//{
		//	{v = std::declval<vertex_t>()} -> std::same_as<T&>;
		//};

		template<typename draw_ctx_t>
		concept has_user_defined_iterators = requires(draw_ctx_t v)
		{
			{v.vertex} -> line_custom_iterable;
		};

		template<typename T>
		struct ops_rref
		{
			explicit ops_rref(const T& v) : v(v){}
			inline T operator * (const auto& other) const;
			inline T operator + (const auto& other) const;
			inline T operator - (const auto& other) const;
			const T& v;
		};

		template<typename T>
		struct ops_lref
		{
			explicit ops_lref(T& v) : v(v){}
			inline T& operator *= (const auto& other);
			inline T& operator += (const auto& other);
			inline T& operator -= (const auto& other);
			inline T operator  *  (const auto& other) const;
			inline T operator  +  (const auto& other) const;
			inline T operator  -  (const auto& other) const;
			T& v;
		};

		template<typename T>
		inline T& ops_lref<T>::operator *= (const auto& other)
		{
			if constexpr(line_custom_iterable_trough_foreach<T>) 	{ dish::for_each_field(v, other, [](auto& a, const auto& b){ a *= b;}); }
			else 													{ return v *= other; }
			return v;
		}

		template<typename T>
		inline T& ops_lref<T>::operator += (const auto& other)
		{
			if constexpr(line_custom_iterable_trough_foreach<T>)	{ dish::for_each_field(v, other, [](auto& a, const auto& b){ a += b;}); }
			else													{ return v += other; }
			return v;
		}

		template<typename T>
		inline T& ops_lref<T>::operator -= (const auto& other)
		{
			if constexpr(line_custom_iterable_trough_foreach<T>)	{ dish::for_each_field(v, other, [](auto& a, const auto& b){ a -= b;}); }
			else if constexpr (requires{v -= other;})				{ return v -= other; }
			else 													{ line_custom_iterable auto temp = other; temp *= -1.0; return v += temp; }
			return v;
		}

		template<typename T>	inline T ops_lref<T>::operator * (const auto& other) const { return ops_rref(v) * other; }
		template<typename T>	inline T ops_lref<T>::operator + (const auto& other) const { return ops_rref(v) + other; }
		template<typename T>	inline T ops_lref<T>::operator - (const auto& other) const { return ops_rref(v) - other; }

		template<typename T>
		inline T ops_rref<T>::operator * (const auto& other) const
		{
			if constexpr(requires{v * other;}) 	{ return v * other; }
			else 								{ T t = v; ops_lref(t) *= other; return t; }
		}

		template<typename T>
		inline T ops_rref<T>::operator + (const auto& other) const
		{
			if constexpr(requires{v + other;}) 	{ return v + other; }
			else 								{ T t = v; ops_lref(t) += other; return t; }
		}

		template<typename T>
		inline T ops_rref<T>::operator - (const auto& other) const
		{
			if constexpr(requires{v * other;}) 	{ return v - other; }
			else 								{ T t = v; ops_lref(t) -= other; return t; }
		}

		template<typename T>
		auto ops(T& v)
		{
			return ops_lref(v);
		}

		template<typename T>
		auto ops(const T& v)
		{
			return ops_rref(v);
		}

		template<typename user_defined_iterators_t>
		struct screen_space_clipped_pt
		{
			dish::vec3<float> screen_pos;
			dish::vec3<float> view_pos;
			user_defined_iterators_t user_defined; // xyz or uv ect... ( they need to be clipped as well )

			screen_space_clipped_pt& operator -= (const screen_space_clipped_pt& other)
			{
				screen_pos.x 		-= other.screen_pos.x;
				screen_pos.y 		-= other.screen_pos.y;
				screen_pos.z 		-= other.screen_pos.z;
				view_pos.x   		-= other.view_pos.x;
				view_pos.y   		-= other.view_pos.y;
				view_pos.z   		-= other.view_pos.z;
				ops(user_defined) 	-= other.user_defined;
				return *this;
			}

			screen_space_clipped_pt& operator += (const screen_space_clipped_pt& other)
			{
				screen_pos.x 		+= other.screen_pos.x;
				screen_pos.y 		+= other.screen_pos.y;
				screen_pos.z 		+= other.screen_pos.z;
				view_pos.x   		+= other.view_pos.x;
				view_pos.y   		+= other.view_pos.y;
				view_pos.z   		+= other.view_pos.z;
				ops(user_defined) 	+= other.user_defined;
				return *this;
			}

			screen_space_clipped_pt& operator *= (const screen_space_clipped_pt& other)
			{
				screen_pos.x 		*= other.screen_pos.x;
				screen_pos.y 		*= other.screen_pos.y;
				screen_pos.z 		*= other.screen_pos.z;
				view_pos.x   		*= other.view_pos.x;
				view_pos.y   		*= other.view_pos.y;
				view_pos.z   		*= other.view_pos.z;
				ops(user_defined) 	*= other.user_defined;
				return *this;
			}

			screen_space_clipped_pt& operator *= (floating_point auto v)
			{
				screen_pos.x    	*= v;
				screen_pos.y    	*= v;
				screen_pos.z    	*= v;
				view_pos.x      	*= v;
				view_pos.y      	*= v;
				view_pos.z      	*= v;
				ops(user_defined) 	*= v;
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

		struct line_it_base
		{
			using flt_t = float;

			int y_start, height;
			flt_t x_it, x;

			void increment_base()
			{
				x += x_it;
			}

			struct set_precalculated
			{
				flt_t y_start_ceiled, height_ceiled, one_over_height_ceiled, sub_pixel;
			};

			template<typename user_defined_iterators_t>
			set_precalculated set_base(const screen_space_clipped_pt<user_defined_iterators_t>& p0, const screen_space_clipped_pt<user_defined_iterators_t>& p1)
			{
				set_precalculated c {
					.y_start_ceiled 		= std::ceil(p0.screen_pos.y),
					.height_ceiled 			= std::ceil(p1.screen_pos.y) - c.y_start_ceiled,
					.one_over_height_ceiled = c.height_ceiled != flt_t(0) ? (flt_t(1) / c.height_ceiled) : flt_t(0),
					.sub_pixel 				= c.y_start_ceiled - p0.screen_pos.y
				};

				//assert(height_ceiled != 0.0f); // this is going to be a division over 0 ! // TODO: handle this to avoid NaN
				y_start    	= static_cast<int>(c.y_start_ceiled);
				height     	= static_cast<int>(c.height_ceiled);
				x_it 		= (p1.screen_pos.x - p0.screen_pos.x) * c.one_over_height_ceiled;
				x			= p0.screen_pos.x + (x_it * c.sub_pixel);
				return c;
			}
		};

		template<typename user_defined_iterators_t, typename enable = void>
		struct line_it;

		template<typename user_defined_iterators_t>
		struct line_it<user_defined_iterators_t, std::enable_if_t<line_custom_iterable<user_defined_iterators_t>>> : line_it_base
		{
			flt_t one_over_z;
			flt_t one_over_z_it;

			user_defined_iterators_t user_defined;
			user_defined_iterators_t user_defined_it;

			void increment()
			{
				increment_base();
				one_over_z += one_over_z_it;
				ops(user_defined) += user_defined_it;
			}

			void set(const screen_space_clipped_pt<user_defined_iterators_t>& p0, const screen_space_clipped_pt<user_defined_iterators_t>& p1)
			{
				const set_precalculated c = set_base(p0, p1);

				// begin and end values
				const flt_t one_over_z_start	= flt_t(1) / p0.view_pos.z;
				const flt_t one_over_z_end 		= flt_t(1) / p1.view_pos.z;

				const auto 	one_over_uv_start	= ops(p0.user_defined) * one_over_z_start;
				const auto 	one_over_uv_end		= ops(p1.user_defined) * one_over_z_end;

				// set iterators
				one_over_z_it   = ops(ops(one_over_z_end ) - one_over_z_start ) * c.one_over_height_ceiled;
				user_defined_it = ops(ops(one_over_uv_end) - one_over_uv_start) * c.one_over_height_ceiled;

				// sub pixel
				one_over_z   = ops(one_over_z_start ) + (ops(one_over_z_it)   * c.sub_pixel);
				user_defined = ops(one_over_uv_start) + (ops(user_defined_it) * c.sub_pixel);
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
				set_base(p0, p1);
			}
		};

		template<typename VecA, typename VecB>
		using smallest_vec3_t = std::conditional_t<sizeof(VecA) < sizeof(VecB), VecA, VecB>;

		inline constexpr void check_context_validity(auto& px_y, auto& px_x_begin, auto& line_length_px, auto buffer_width, auto buffer_height)
		{
#if 0
			int to = px_x_begin + line_length_px;
			assert(to 				>= px_x_begin);
			assert(px_y 		< ctx.buffer_height);
			assert(px_x_begin 	< ctx.buffer_width);
			assert(to				< ctx.buffer_width);
			assert(line_length_px	< 100000);
#else
			px_y 			= std::clamp(px_y, 		0u, (uint32_t)(buffer_height-1));
			px_x_begin   	= std::clamp(px_x_begin, 0u, (uint32_t)(buffer_width-1));
			int to = std::clamp(px_x_begin + line_length_px, 0u, (uint32_t)(buffer_width-1));
			if(to <= px_x_begin || line_length_px > 100000)
			{
				line_length_px = 0;
			}
			else
			{
				line_length_px = to - px_x_begin;
			}

			//px_y 					= std::min(px_y, (ctx.buffer_height-1));
			//px_x_begin 		= std::min(px_x_begin, (ctx.buffer_width-1));
			//line_length_px 	= (px_x_begin + line_length_px) > (ctx.buffer_width-1) ? ((ctx.buffer_width-1) - px_x_begin) : line_length_px;
#endif
		}

		template<typename draw_ctx_t, typename user_defined_iterators_t, triangle triangle_t, shader<triangle_t> shader_t>
		constexpr inline void it_line(const triangle_t& source_triangle, draw_ctx_t& ctx, int y, line_it<user_defined_iterators_t>& left, line_it<user_defined_iterators_t>& right, shader_t&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using flt_t = line_it<user_defined_iterators_t>::flt_t;

			const flt_t left_x_floored 		= std::floor(left.x);
			const flt_t width 				= right.x - left.x;
			const flt_t one_over_width 		= flt_t(1) / width;
			const flt_t sub_texel 			= (left_x_floored) - left.x;

			uint32_t px_y 			= y;
			uint32_t px_x_begin 	= static_cast<uint32_t>(left_x_floored); // NOLINT
			uint32_t line_length_px = static_cast<int>(right.x) - px_x_begin;
			auto     one_over_z_it	= ops(ops(right.one_over_z)   - left.one_over_z  ) * one_over_width;
			auto     vertex_it      = ops(ops(right.user_defined) - left.user_defined) * one_over_width;
			auto     one_over_z		= ops(ops(one_over_z_it) * sub_texel) + left.one_over_z;
			auto     vertex         = ops(ops(vertex_it)     * sub_texel) + left.user_defined;
			uint32_t px_index_begin = px_x_begin + px_y * frame_width;

			check_context_validity(px_y, px_x_begin, line_length_px, frame_width, frame_height);

			if constexpr(requires{ctx.px_y;})			ctx.px_y 			= px_y;
			if constexpr(requires{ctx.px_x_begin;})		ctx.px_x_begin 		= px_x_begin;
			if constexpr(requires{ctx.line_length_px;})	ctx.line_length_px 	= line_length_px;
			if constexpr(requires{ctx.one_over_z_it;})	ctx.one_over_z_it 	= one_over_z_it;
			if constexpr(requires{ctx.vertex_it;})		ctx.vertex_it 		= vertex_it;
			if constexpr(requires{ctx.one_over_z;})		ctx.one_over_z 		= one_over_z;
			if constexpr(requires{ctx.vertex;})			ctx.vertex 			= vertex;
			if constexpr(requires{ctx.px_index_begin;})	ctx.px_index_begin 	= px_index_begin;

			constexpr bool is_shader_scanline = detail::concepts::shader_scanline<shader_t, triangle_t>;
			constexpr bool is_shader_fragment = detail::concepts::shader_fragment<shader_t, triangle_t>;
			static_assert(is_shader_scanline + is_shader_fragment <= 1); // shader can not both have a 'scanline' AND 'fragment' function.

			if constexpr(is_shader_scanline)
			{
				shader.scanline(source_triangle, ctx);
			}
			else if constexpr(is_shader_fragment)
			{
				// scanline 'wrapper' here then...
				dis::frag_vars<user_defined_iterators_t> it;
				it.px_index = px_index_begin;
				for(size_t i=0; i<line_length_px; i++)
				{
					flt_t z = flt_t(1)/one_over_z;
					it.depth = z;
					if constexpr(!std::is_same_v<user_defined_iterators_t, std::nullptr_t>)
					{
						user_defined_iterators_t& user_vars = it;
						user_vars = vertex;
						ops(user_vars) *= z;
					}
					shader.fragment(source_triangle, it);
					ops(vertex) += vertex_it;
					one_over_z  += one_over_z_it;
					it.px_index++;
				}
			}

			// increment edges
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
		template<typename draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t>
		constexpr void draw_triangle_unsafe(const triangle_t& source_triangle, screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, shader<triangle_t> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			if constexpr(requires{std::declval<draw_ctx_t>().vertex;})
			{
				static_assert(has_user_defined_iterators<draw_ctx_t>, "'vertex' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
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
			if constexpr(requires{ctx.frame_width;})  ctx.frame_width  = static_cast<decltype(ctx.frame_width)>(frame_width);
			if constexpr(requires{ctx.frame_height;}) ctx.frame_height = static_cast<decltype(ctx.frame_height)>(frame_height);

			if(cross_z > 0.0f)
			{
				line_it<user_defined_iterators_t> line_it_long, line_it_top, line_it_bot;

				line_it_long.set(line_long.p0, line_long.p1);
				line_it_top.set(line_top.p0, line_top.p1);
				line_it_bot.set(line_bot.p0, line_bot.p1);

				int y=line_it_long.y_start;
				for(; y<line_it_long.y_start+line_it_top.height; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_top, shader, frame_width, frame_height);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_long, line_it_bot, shader, frame_width, frame_height);
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
					it_line(source_triangle, ctx, y, line_it_top, line_it_long, shader, frame_width, frame_height);
				}

				const int yLimit = y_pos_at_intersection(y, line_it_long, line_it_bot);
				for(; y<yLimit; y++)
				{
					it_line(source_triangle, ctx, y, line_it_bot, line_it_long, shader, frame_width, frame_height);
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

		template<floating_point flt_t>
		constexpr dish::vec4<flt_t> mul(const dish::mat4x4& m1, const dish::vec4<flt_t>& m2)
		{
			dish::vec4<flt_t> result = {};
			result.x = m1[0][0] * m2.x + m1[1][0] * m2.y + m1[2][0] * m2.z + m1[3][0] * m2.w;
			result.y = m1[0][1] * m2.x + m1[1][1] * m2.y + m1[2][1] * m2.z + m1[3][1] * m2.w;
			result.z = m1[0][2] * m2.x + m1[1][2] * m2.y + m1[2][2] * m2.z + m1[3][2] * m2.w;
			result.w = m1[0][3] * m2.x + m1[1][3] * m2.y + m1[2][3] * m2.z + m1[3][3] * m2.w;
			return result;
		}

		template<floating_point flt_t>
		constexpr dish::vec4<flt_t> mul(const dish::mat4x4& m1, const dish::vec3<flt_t>& m2, flt_t m2_w)
		{
			dish::vec4<flt_t> result = {};
			result.x = m1[0][0] * m2.x + m1[1][0] * m2.y + m1[2][0] * m2.z + m1[3][0] * m2_w;
			result.y = m1[0][1] * m2.x + m1[1][1] * m2.y + m1[2][1] * m2.z + m1[3][1] * m2_w;
			result.z = m1[0][2] * m2.x + m1[1][2] * m2.y + m1[2][2] * m2.z + m1[3][2] * m2_w;
			result.w = m1[0][3] * m2.x + m1[1][3] * m2.y + m1[2][3] * m2.z + m1[3][3] * m2_w;
			return result;
		}

		constexpr auto& get_tri_p0(triangle auto& tri) { return get_tri_pt<0>(tri); }
		constexpr auto& get_tri_p1(triangle auto& tri) { return get_tri_pt<1>(tri); }
		constexpr auto& get_tri_p2(triangle auto& tri) { return get_tri_pt<2>(tri); }

		/*
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
		 */

		template<typename flt_t>
		struct plane
		{
			dish::vec3<flt_t> pos, normal;
		};

		template<typename flt_t>
		struct viewport
		{
			flt_t x_start, y_start, x_end, y_end;
		};

		namespace detail::clip_triangle
		{

			template<typename flt_t>
			constexpr bool should_cut_to_quad(const plane<flt_t>& plane, const dish::vec3<flt_t>& a, const dish::vec3<flt_t>& b, const dish::vec3<flt_t>& c, bool intersects_a_to_b, bool intersects_b_to_c, bool intersects_c_to_a)
			{
				const vector3 auto& triangle_tip  		= (intersects_a_to_b && intersects_c_to_a) ? a : ((intersects_a_to_b && intersects_b_to_c) ? b : c);
				const vector3 auto& any_other_tip 		= (intersects_a_to_b && intersects_c_to_a) ? b : ((intersects_a_to_b && intersects_b_to_c) ? c : a);
				const vector3 auto	tip_to_tip			= sub_xyz(triangle_tip, any_other_tip);

				return (tip_to_tip.x*plane.normal.x + tip_to_tip.y*plane.normal.y + tip_to_tip.z*plane.normal.z) < 0.0f;
			}

			template<typename flt_t>
			static flt_t intersect(const plane<flt_t>& plane, const dish::vec3<flt_t>& ray_origin, const dish::vec3<flt_t>& ray_dir)
			{
				const floating_point auto denom = (plane.normal.x * ray_dir.x) + (plane.normal.y * ray_dir.y) + (plane.normal.z * ray_dir.z);
				if ((denom*denom) > static_cast<decltype(denom)>(0.0001 * 0.0001)) // your favorite epsilon
				{
					const vector3 auto 			d = sub_xyz(plane.pos, ray_origin);
					const floating_point auto 	d_dot = dot_xyz(d, plane.normal);
					return d_dot / denom;
				}
				return static_cast<decltype(denom)>(-1.0f);
			}

			template<typename flt_t>
			constexpr int is_behind_clipping_plane(const plane<flt_t>& plane, const dish::vec3<flt_t>& pt)
			{
				return dot_xyz(sub_xyz(pt, plane.pos), plane.normal) > 0 ? 0 : -1;
			}


			template<typename flt_t>
			constexpr flt_t intersect_top(const viewport<flt_t>& viewport, const dish::vec3<flt_t>& ray_origin, const dish::vec3<flt_t>& ray_dir)
			{
				return intersect(plane{ {0, viewport.y_end, 0}, {0, -1, 0} }, ray_origin, ray_dir);
			}

			template<typename flt_t>
			constexpr flt_t intersect_bottom(const viewport<flt_t>& viewport, const dish::vec3<flt_t>& ray_origin, const dish::vec3<flt_t>& ray_dir)
			{
				return intersect(plane{{0, viewport.y_end, 0}, {0, -1, 0}}, ray_origin, ray_dir);
			}
		}

		template<typename user_defined_iterators_t, typename intersect_func_t, typename is_behind_func_t, typename should_cut_to_quad_func_t>
		constexpr int clip_triangle_ext(screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_in_out,
										screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_extra_out,
										intersect_func_t intersect_func,
										is_behind_func_t is_behind_func,
										should_cut_to_quad_func_t should_cut_to_quad_func,
										const auto& func_argument)
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

			//static_assert(std::is_same_v<std::decay_t<decltype(a_pt.screen_pos)>, std::decay_t<decltype(ab_dir)>>);
			const floating_point auto a_to_b_t = intersect_func(func_argument, a_pt.screen_pos, ab_dir);
			const floating_point auto b_to_c_t = intersect_func(func_argument, b_pt.screen_pos, bc_dir);
			const floating_point auto c_to_a_t = intersect_func(func_argument, c_pt.screen_pos, ca_dir);

			const bool intersects_a_to_b = (a_to_b_t > 0 && a_to_b_t < ab_len);
			const bool intersects_b_to_c = (b_to_c_t > 0 && b_to_c_t < bc_len);
			const bool intersects_c_to_a = (c_to_a_t > 0 && c_to_a_t < ca_len);

			const floating_point auto ab_perc = a_to_b_t * ab_len_inv;
			const floating_point auto bc_perc = b_to_c_t * bc_len_inv;
			const floating_point auto ca_perc = c_to_a_t * ca_len_inv;

			point_t intersection_a_to_b_pt = a_pt + (ab_pt * ab_perc);
			point_t intersection_b_to_c_pt = b_pt + (bc_pt * bc_perc);
			point_t intersection_c_to_a_pt = c_pt + (ca_pt * ca_perc);

			if(intersects_a_to_b || intersects_b_to_c || intersects_c_to_a)
			{
				const bool should_cut_to_quad = should_cut_to_quad_func(func_argument, a_pt.screen_pos, b_pt.screen_pos, c_pt.screen_pos, intersects_a_to_b, intersects_b_to_c, intersects_c_to_a);
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
				return is_behind_func(func_argument, c_pt.screen_pos);
			}
			return 0;
		}

		template<typename user_defined_iterators_t, typename flt_t>
		constexpr int clip_triangle(const plane<flt_t>& plane, screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_in_out, screen_space_clipped_pt_triangle<user_defined_iterators_t>& tri_extra_out)
		{
			using namespace detail::clip_triangle;
			return clip_triangle_ext(tri_in_out, tri_extra_out, detail::clip_triangle::intersect<flt_t>, is_behind_clipping_plane<flt_t>, should_cut_to_quad<flt_t>, plane);
		}

		template<typename draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t, typename viewport_flt_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, shader<triangle_t> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height, const viewport<viewport_flt_t>& viewport)
		{
			using clipped_tri = screen_space_clipped_pt_triangle<user_defined_iterators_t>;
			constexpr auto tris_capacity = (2*2*2*2)+1;

			clipped_tri				clipped_tris[tris_capacity]; // NOLINT
			std::uint_fast8_t 		clipped_tris_num = 0;

			// first just add the main triangle
			clipped_tris[clipped_tris_num++] = triangle;

			// clipping planes
			// top clipping plane
			{
				const plane<viewport_flt_t> plane{{0, viewport.y_end, 0}, {0, -1, 0}};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y < viewport.y_end, tri[1].screen_pos.y < viewport.y_end, tri[2].screen_pos.y < viewport.y_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(plane, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// bot clipping plane
			{
				const plane<viewport_flt_t> plane{{0, viewport.y_start, 0}, {0, 1, 0}};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.y > viewport.y_start, tri[1].screen_pos.y > viewport.y_start, tri[2].screen_pos.y > viewport.y_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(plane, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// left clipping plane
			{
				const plane<viewport_flt_t> plane{{viewport.x_start, 0, 0}, {1, 0, 0}};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x > viewport.x_start, tri[1].screen_pos.x > viewport.x_start, tri[2].screen_pos.x > viewport.x_start};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(plane, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// right clipping plane
			{
				const plane<viewport_flt_t> plane{{viewport.x_end, 0, 0}, {-1, 0, 0}};

				for(std::uint_fast8_t i=clipped_tris_num; i--;)
				{
					clipped_tri& tri = clipped_tris[i];

					const bool 					in_screen[3] 		= {tri[0].screen_pos.x < viewport.x_end, tri[1].screen_pos.x < viewport.x_end, tri[2].screen_pos.x < viewport.x_end};
					const std::uint_fast8_t 	num_pts_in_screen 	= in_screen[0] + in_screen[1] + in_screen[2];

					if(num_pts_in_screen == 0) // not in screen. delete
					{
						std::swap(clipped_tris[i], clipped_tris[clipped_tris_num-1]);
						clipped_tris_num--;
					}
					else if(num_pts_in_screen != 3) // otherwise nothing to clip. leave it
					{
						clipped_tris_num += clip_triangle<user_defined_iterators_t>(plane, tri, clipped_tris[clipped_tris_num]);
					}
				}
			}

			// draw
			for(std::uint_fast8_t i=0; i<clipped_tris_num; i++)
			{
				draw_triangle_unsafe<draw_ctx_t>(source_triangle, clipped_tris[i], shader, frame_width, frame_height);
			}
		}

		template<typename draw_ctx_t, triangle triangle_t, typename user_defined_iterators_t>
		constexpr void draw_triangle(const triangle_t& source_triangle, const screen_space_clipped_pt_triangle<user_defined_iterators_t>& triangle, shader<triangle_t> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			viewport<float> viewport{1.0f, 1.0f, (float)(frame_width-1), (float)(frame_height-1)};
			draw_triangle<draw_ctx_t, triangle_t, user_defined_iterators_t>(source_triangle, triangle, shader, frame_width, frame_height, viewport);
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

		template<typename draw_ctx_t, triangle_list triangle_list_t>
		constexpr void render_rasterize(const triangle_list_t& triangles, const camera auto& camera, shader<triangle_from_list_t<triangle_list_t>> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
		{
			using triangle_t 				= std::decay_t<decltype(*triangles.begin())>;
			using user_defined_iterators_t 	= std::conditional_t<has_user_defined_iterators<draw_ctx_t>, std::decay_t<decltype(std::declval<draw_ctx_t>().vertex)>, std::nullptr_t>;
			using clipped_tri_t 			= screen_space_clipped_pt_triangle<user_defined_iterators_t>;

			const float target_width_flt 	= static_cast<float>(frame_width);  // NOLINT
			const float target_height_flt 	= static_cast<float>(frame_height); // NOLINT
			const float aspect = target_width_flt / target_height_flt;

			const float near_plane 		= 0.0005f;
			const float far_plane 		= 50.0f;
			const float near_clip_plane = 0.1f;

			const dish::mat4x4 perspective 	= create_perspective(camera.fov, aspect, near_plane, far_plane);
			const dish::mat4x4 lookat 		= create_lookat(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});
			const dish::mat4x4 projview 	= mul(perspective, lookat);

			const vector3 auto near_clipping_plane_normal 	= direction_to(camera.pos, camera.lookat);
			const vector3 auto near_clipping_plane_pos 		= add_xyz(camera.pos, mul_xyz(near_clipping_plane_normal, near_plane));

			for(const triangle auto& tri : triangles)
			{
				std::array<clipped_tri_t, 2> clipped_tris; // NOLINT
				size_t num_clipped_tris = 0;

				const auto& vertex0 = get_tri_pt<0>(tri);
				const auto& vertex1 = get_tri_pt<1>(tri);
				const auto& vertex2 = get_tri_pt<2>(tri);

				clipped_tris[0][0].user_defined = shader.vertex(vertex0);
				clipped_tris[0][1].user_defined = shader.vertex(vertex1);
				clipped_tris[0][2].user_defined = shader.vertex(vertex2);
				clipped_tris[0][0].screen_pos = {vertex0.x, vertex0.y, vertex0.z};
				clipped_tris[0][1].screen_pos = {vertex1.x, vertex1.y, vertex1.z};
				clipped_tris[0][2].screen_pos = {vertex2.x, vertex2.y, vertex2.z};
				num_clipped_tris++;

				// clip near
				const plane<float> plane{near_clipping_plane_pos, near_clipping_plane_normal};

				num_clipped_tris += clip_triangle<user_defined_iterators_t>(plane, clipped_tris[0], clipped_tris[1]);

				for(size_t clipped_tri_index=0; clipped_tri_index < num_clipped_tris; clipped_tri_index++)
				{
					clipped_tri_t& clipped_tri = clipped_tris[clipped_tri_index];

					using flt_t = decltype(clipped_tri[0].screen_pos.x);

					const vector4 auto p0_view = mul(lookat, clipped_tri[0].screen_pos, flt_t(1));
					const vector4 auto p1_view = mul(lookat, clipped_tri[1].screen_pos, flt_t(1));
					const vector4 auto p2_view = mul(lookat, clipped_tri[2].screen_pos, flt_t(1));

					const vector4 auto p0_projview = mul(perspective, p0_view);
					const vector4 auto p1_projview = mul(perspective, p1_view);
					const vector4 auto p2_projview = mul(perspective, p2_view);

					/*
					const glm::mat4 perspective4 	= glm::perspective(camera.fov, aspect, near_plane, far_plane);
					const glm::mat4 lookat4 		= glm::lookAt(glm::vec3{camera.pos.x, camera.pos.y, camera.pos.z}, glm::vec3{camera.lookat.x, camera.lookat.y, camera.lookat.z}, glm::vec3{camera.up.x, camera.up.y, camera.up.z});

					auto p0_test_glm = glm::project(glm::vec3{clipped_tri[0].screen_pos.x, clipped_tri[0].screen_pos.y, clipped_tri[0].screen_pos.z}, lookat4, perspective4, glm::vec4(0.0f,0.0f,(float)frame_width, (float)frame_height));
					auto p1_test_glm = glm::project(glm::vec3{clipped_tri[1].screen_pos.x, clipped_tri[1].screen_pos.y, clipped_tri[1].screen_pos.z}, lookat4, perspective4, glm::vec4(0.0f,0.0f,(float)frame_width, (float)frame_height));
					auto p2_test_glm = glm::project(glm::vec3{clipped_tri[2].screen_pos.x, clipped_tri[2].screen_pos.y, clipped_tri[2].screen_pos.z}, lookat4, perspective4, glm::vec4(0.0f,0.0f,(float)frame_width, (float)frame_height));

					dish::vec3<float> p0_test = {p0_test_glm.x, p0_test_glm.y, p0_test_glm.z};
					dish::vec3<float> p1_test = {p1_test_glm.x, p1_test_glm.y, p1_test_glm.z};
					dish::vec3<float> p2_test = {p2_test_glm.x, p2_test_glm.y, p2_test_glm.z};
					 */

							//const vector4 auto p0_projview = mul(projview, clipped_tri[0].screen_pos, flt_t(1));
					//const vector4 auto p1_projview = mul(projview, clipped_tri[1].screen_pos, flt_t(1));
					//const vector4 auto p2_projview = mul(projview, clipped_tri[2].screen_pos, flt_t(1));

					assert(p0_projview.w != flt_t(0));

					clipped_tri[0].screen_pos.x = ((p0_projview.x / p0_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_width_flt;
					clipped_tri[0].screen_pos.y = ((p0_projview.y / p0_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_height_flt;
					clipped_tri[0].screen_pos.z = (p0_projview.z / p0_projview.w);

					clipped_tri[1].screen_pos.x = ((p1_projview.x / p1_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_width_flt;
					clipped_tri[1].screen_pos.y = ((p1_projview.y / p1_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_height_flt;
					clipped_tri[1].screen_pos.z = (p1_projview.z / p1_projview.w);

					clipped_tri[2].screen_pos.x = ((p2_projview.x / p2_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_width_flt;
					clipped_tri[2].screen_pos.y = ((p2_projview.y / p2_projview.w) * flt_t(0.5) + flt_t(0.5)) * target_height_flt;
					clipped_tri[2].screen_pos.z = (p2_projview.z / p2_projview.w);

					clipped_tri[0].view_pos.x = p0_view.x;
					clipped_tri[0].view_pos.y = p0_view.y;
					clipped_tri[0].view_pos.z = p0_view.z;

					clipped_tri[1].view_pos.x = p1_view.x;
					clipped_tri[1].view_pos.y = p1_view.y;
					clipped_tri[1].view_pos.z = p1_view.z;

					clipped_tri[2].view_pos.x = p2_view.x;
					clipped_tri[2].view_pos.y = p2_view.y;
					clipped_tri[2].view_pos.z = p2_view.z;

					const floating_point auto cross_z = (clipped_tri[1].screen_pos.x - clipped_tri[0].screen_pos.x) * (clipped_tri[2].screen_pos.y - clipped_tri[0].screen_pos.y) - (clipped_tri[2].screen_pos.x - clipped_tri[0].screen_pos.x) * (clipped_tri[1].screen_pos.y - clipped_tri[0].screen_pos.y);
					const bool backface_culling = cross_z > 0;

					if(backface_culling)
					{
						draw_triangle<draw_ctx_t, triangle_t, user_defined_iterators_t>(tri, clipped_tri, shader, frame_width, frame_height);
					}
				}
			}
		}
	}

	template<triangle_list triangle_list_t>
	constexpr void render(const triangle_list_t& triangles, const camera auto& camera, shader<triangle_from_list_t<triangle_list_t>> auto&& shader, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	{
		using shader_t 					= std::decay_t<decltype(shader)>;
		using triangle_t 				= triangle_from_list_t<triangle_list_t>;
		using vertex_t					= vertex_from_tri_t<triangle_t>;
		using draw_ctx_t 				= draw_ctx<detail::user_defined_iterators_from_shader<shader_t, triangle_t>>;

		if constexpr(requires{std::declval<draw_ctx_t>().vertex;})
		{
			//using user_defined_iterators_t = detail::user_defined_iterators_from_draw_ctx<draw_ctx_t>;
			//static_assert(detail::can_assign_vertex<vertex_t, user_defined_iterators_t>, 	"'vertex' member found in 'draw_ctx_t' but cannot assign a tri to it");
			static_assert(detail::has_user_defined_iterators<draw_ctx_t>, 					"'vertex' member found in 'draw_ctx_t' but does not satisfy the 'has_user_defined_iterators' conditions");
		}

		detail::render_rasterize<draw_ctx_t>(triangles, camera, shader, frame_width, frame_height);
	}

	//template<triangle_list triangle_list_t>
	//constexpr void render_nonshaded(const triangle_list_t& triangles, const camera auto& camera, unsigned_integral auto frame_width, unsigned_integral auto frame_height)
	//{
		//render<draw_hline_ctx, triangle_list_t>(triangles, camera, shader, frame_width, frame_height);
	//}

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
